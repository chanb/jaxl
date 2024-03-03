from types import SimpleNamespace
from typing import Any, Dict, Tuple

import chex
import jax
import jax.random as jrandom
import numpy as np
import timeit

from jaxl.constants import *
from jaxl.learners.learner import OfflineLearner
from jaxl.losses import get_loss_function, make_aggregate_loss
from jaxl.models import get_model, get_optimizer, get_update_function
from jaxl.utils import l2_norm, parse_dict


"""
Supervised learner. We assume traditional setting where we have input/output 
pairs sampled i.i.d. from a data distribution. We support seq-to-seq style
learning, provided that the correct buffers and losses are provided.
XXX: Feel free to add new components as needed.
"""


class SupervisedLearner(OfflineLearner):
    """
    Supervised learner class that extends the ``OfflineLearner`` class.
    This is the general learner for supervised learning agents.
    """

    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        super().__init__(config, model_config, optimizer_config)

        if getattr(self._config, CONST_BUFFER_CONFIG, False):

            def sample():
                input, carry, output, _ = self._buffer.sample(self._config.batch_size)
                return input, carry, output

        else:

            def sample():
                try:
                    inputs, carries, outputs, _ = next(self._train_loader)
                except StopIteration:
                    self._train_loader = iter(self._train_dataloader)
                    inputs, carries, outputs, _ = next(self._train_loader)
                inputs = inputs.numpy()
                carries = carries.numpy()
                outputs = outputs.numpy()
                return inputs, carries, outputs

        self.sample = sample

        self._initialize_losses()
        self.train_step = jax.jit(self.make_train_step())

    def _initialize_model_and_opt(self, input_dim: chex.Array, output_dim: chex.Array):
        """
        Construct the model and the optimizer.

        :param input_dim: input dimension of the data point
        :param output_dim: output dimension of the data point
        :type input_dim: chex.Array
        :type output_dim: chex.Array

        """
        self._model = get_model(input_dim, output_dim, self._model_config)

        model_key = jrandom.PRNGKey(self._config.seeds.model_seed)
        dummy_x = self._generate_dummy_x(input_dim)
        params = self._model.init(model_key, dummy_x)
        self._optimizer, opt_state = get_optimizer(
            self._optimizer_config, self._model, params
        )
        self._model_dict = {CONST_MODEL: params, CONST_OPT_STATE: opt_state}

    def _initialize_losses(self):
        """
        Construct the losses.
        """
        losses = {}
        for loss, loss_setting in zip(self._config.losses, self._config.loss_settings):
            loss_setting_ns = parse_dict(loss_setting)
            losses[loss] = (
                get_loss_function(
                    self._model,
                    loss,
                    loss_setting_ns,
                    num_classes=self._buffer.output_dim[-1],
                ),
                loss_setting_ns.coefficient,
            )
        self._loss = jax.jit(make_aggregate_loss(losses))

    def make_train_step(self):
        """
        Makes the training step for model update.
        """

        update_function = get_update_function(self._model)

        def _train_step(
            model_dict: Dict[str, Any],
            train_x: chex.Array,
            train_carry: chex.Array,
            train_y: chex.Array,
            *args,
            **kwargs,
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            """
            The training step that computes the loss and performs model update.

            :param model_dict: the model state and optimizer state
            :param train_x: the training inputs
            :param train_carry: the training hidden states for memory-based models
            :param train_y: the training targets
            :param *args:
            :param **kwargs:
            :type model_dict: Dict[str, Any]
            :type train_x: chex.Array
            :type train_carry: chex.Array
            :type train_y: chex.Array
            :return: the updated model state and optimizer state, and auxiliary information
            :rtype: Tuple[Dict[str, Any], Dict[str, Any]]
            """
            (agg_loss, aux), grads = jax.value_and_grad(self._loss, has_aux=True)(
                model_dict[CONST_MODEL],
                train_x,
                train_carry,
                train_y,
            )
            aux[CONST_AGG_LOSS] = agg_loss

            params, opt_state = update_function(
                self._optimizer,
                grads,
                model_dict[CONST_OPT_STATE],
                model_dict[CONST_MODEL],
                aux[self._config.losses[0]][CONST_AUX][CONST_UPDATES],
            )

            aux[CONST_GRAD_NORM] = {CONST_MODEL: l2_norm(grads)}

            return {CONST_MODEL: params, CONST_OPT_STATE: opt_state}, aux

        return _train_step

    def compute_loss(
        self, xs: chex.Array, ys: chex.Array
    ) -> Tuple[chex.Array, Dict[str, Any]]:
        """
        Computes the loss.

        :param xs: the batch of inputs
        :param ys: the batch of outputs
        :type xs: chex.Array
        :type ys: chex.Array
        :return: the loss and auxiliary information
        :rtype: Tuple[chex.Array, Dict[str, Any]]

        """
        return self._loss(self._model_dict[CONST_MODEL], xs, ys)

    def update(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Updates the model.

        :param *args:
        :param **kwargs:
        :return: the update information
        :rtype: Dict[str, Any]

        """
        auxes = []
        total_update_time = 0
        for _ in range(self._num_updates_per_epoch):
            tic = timeit.default_timer()
            auxes.append({})
            inputs, carries, outputs = self.sample()

            self.model_dict, aux = self.train_step(
                self._model_dict, inputs, carries, outputs
            )
            total_update_time += timeit.default_timer() - tic
            assert np.isfinite(aux[CONST_AGG_LOSS]), f"Loss became NaN\naux: {aux}"

            auxes[-1][CONST_AUX] = aux

        auxes = jax.tree_util.tree_map(lambda *args: np.mean(args), *auxes)
        aux[CONST_LOG] = {
            f"losses/{CONST_AGG_LOSS}": auxes[CONST_AUX][CONST_AGG_LOSS].item(),
            f"time/{CONST_UPDATE_TIME}": total_update_time,
            f"{CONST_GRAD_NORM}/model": auxes[CONST_AUX][CONST_GRAD_NORM][
                CONST_MODEL
            ].item(),
        }

        for loss_key in self._config.losses:
            aux[CONST_LOG][f"losses/{loss_key}"] = auxes[CONST_AUX][loss_key][
                CONST_LOSS
            ].item()

        return aux
