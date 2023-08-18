from types import SimpleNamespace
from typing import Any, Dict, Tuple

import chex
import jax
import jax.random as jrandom
import numpy as np
import optax

from jaxl.constants import *
from jaxl.learners.learner import OfflineLearner
from jaxl.losses import get_loss_function, make_aggregate_loss
from jaxl.models import get_model, get_optimizer, get_update_function
from jaxl.utils import parse_dict


"""
In-context learner. We assume traditional setting where we have input/output 
pairs sampled i.i.d. from a data distribution. We support seq-to-seq style
learning, provided that the correct buffers and losses are provided.
XXX: Feel free to add new components as needed.
"""


class InContextLearner(OfflineLearner):
    """
    In-context learner class that extends the ``OfflineLearner`` class.
    This is the general learner for in-context learning agents.
    """

    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        super().__init__(config, model_config, optimizer_config)
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
        dummy_input = self._generate_dummy_x(input_dim)
        dummy_output = self._generate_dummy_x(output_dim)
        params = self._model.init(model_key, dummy_input, dummy_output)
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
            context_inputs,
            context_outputs,
            queries,
            outputs,
            *args,
            **kwargs,
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            """
            The training step that computes the loss and performs model update.

            :param model_dict: the model state and optimizer state
            :param context_inputs: the context inputs
            :param context_outputs: the context inputs
            :param queries: the queries
            :param outputs: the outputs
            :param *args:
            :param **kwargs:
            :type model_dict: Dict[str, Any]
            :type context_inputs: chex.Array
            :type context_outputs: chex.Array
            :type queries: chex.Array
            :type outputs: chex.Array
            :return: the updated model state and optimizer state, and auxiliary information
            :rtype: Tuple[Dict[str, Any], Dict[str, Any]]
            """
            (agg_loss, aux), grads = jax.value_and_grad(self._loss, has_aux=True)(
                model_dict[CONST_MODEL],
                queries,
                {
                    CONST_CONTEXT_INPUT: context_inputs,
                    CONST_CONTEXT_OUTPUT: context_outputs,
                },
                outputs,
            )
            aux[CONST_AGG_LOSS] = agg_loss

            params, opt_state = update_function(
                self._optimizer,
                grads,
                model_dict[CONST_OPT_STATE],
                model_dict[CONST_MODEL],
            )

            return {CONST_MODEL: params, CONST_OPT_STATE: opt_state}, aux

        return _train_step

    def update(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Updates the model.

        :param *args:
        :param **kwargs:
        :return: the update information
        :rtype: Dict[str, Any]

        """
        try:
            context_inputs, context_outputs, queries, outputs = next(self._train_loader)
        except StopIteration:
            self._train_loader = iter(self._train_dataloader)
            context_inputs, context_outputs, queries, outputs = next(self._train_loader)
        context_inputs = context_inputs.numpy()
        context_outputs = context_outputs.numpy()
        queries = queries.numpy()
        outputs = outputs.numpy()

        self.model_dict, aux = self.train_step(
            self._model_dict, context_inputs, context_outputs, queries, outputs
        )
        assert np.isfinite(aux[CONST_AGG_LOSS]), f"Loss became NaN\naux: {aux}"

        aux[CONST_LOG] = {CONST_AGG_LOSS: aux[CONST_AGG_LOSS]}

        return aux
