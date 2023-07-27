from types import SimpleNamespace
from typing import Any, Dict, Tuple, Union

import _pickle as pickle
import chex
import jax
import math
import numpy as np
import optax
import os
import timeit

from jaxl.constants import *
from jaxl.learners.supervised import SupervisedLearner
from jaxl.learners.utils import gather_per_leaf_l2_norm
from jaxl.models import get_update_function
from jaxl.utils import l2_norm, RunningMeanStd


"""
Standard Behavioural Cloning.
"""


class BC(SupervisedLearner):
    """
    Behavioural Cloning (BC) algorithm. This extends `SupervisedLearner`.
    """

    #: The running statistics for the observations.
    _obs_rms: Union[bool, RunningMeanStd]

    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        super().__init__(config, model_config, optimizer_config)

        self._obs_rms = False
        if getattr(config, CONST_OBS_RMS, False):
            self._obs_rms = RunningMeanStd(shape=self._buffer.input_dim)

            # We compute the statistics offline by iterating through the whole buffer once
            num_batches = math.ceil(len(self._buffer) / self._config.batch_size)
            last_batch_remainder = len(self._buffer) % self._config.batch_size
            for batch_i in range(num_batches):
                if batch_i + 1 == num_batches:
                    batch_size = last_batch_remainder
                    sample_idxes = np.arange(
                        batch_i * self._config.batch_size,
                        batch_i * self._config.batch_size + last_batch_remainder,
                    )
                else:
                    batch_size = self._config.batch_size
                    sample_idxes = np.arange(
                        batch_i * self._config.batch_size,
                        (batch_i + 1) * self._config.batch_size,
                    )
                obss, _, _, _, _, _, _, _, _, _ = self._buffer.sample(
                    batch_size, sample_idxes
                )
                self._obs_rms.update(obss)

    def _initialize_model_and_opt(self, input_dim: chex.Array, output_dim: chex.Array):
        """
        Construct the model and the optimizer.

        :param input_dim: input dimension of the data point
        :param output_dim: output dimension of the data point
        :type input_dim: chex.Array
        :type output_dim: chex.Array

        """
        super()._initialize_model_and_opt(input_dim, output_dim)
        self._model_dict = {
            CONST_MODEL: {CONST_POLICY: self._model_dict[CONST_MODEL]},
            CONST_OPT_STATE: {CONST_POLICY: self._model_dict[CONST_OPT_STATE]},
        }

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
                model_dict[CONST_MODEL][CONST_POLICY],
                train_x,
                train_carry,
                train_y,
            )
            aux[CONST_AGG_LOSS] = agg_loss
            params, opt_state = update_function(
                self._optimizer,
                grads,
                model_dict[CONST_OPT_STATE][CONST_POLICY],
                model_dict[CONST_MODEL][CONST_POLICY],
            )
            aux[CONST_GRAD_NORM] = {CONST_POLICY: l2_norm(grads)}
            return {
                CONST_MODEL: {CONST_POLICY: params},
                CONST_OPT_STATE: {CONST_POLICY: opt_state},
            }, aux

        return _train_step

    def update(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Updates the policy.

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
            obss, h_states, acts_e, _, _, _, _, _, _, _ = self._buffer.sample(
                self._config.batch_size
            )
            self.model_dict, aux = self.train_step(
                self._model_dict, obss, h_states, acts_e
            )
            total_update_time += timeit.default_timer() - tic
            assert np.isfinite(aux[CONST_AGG_LOSS]), f"Loss became NaN\naux: {aux}"

            auxes[-1][CONST_AUX] = aux

        auxes = jax.tree_util.tree_map(lambda *args: np.mean(args), *auxes)
        aux[CONST_LOG] = {
            f"losses/{CONST_AGG_LOSS}": auxes[CONST_AUX][CONST_AGG_LOSS].item(),
            f"time/{CONST_UPDATE_TIME}": total_update_time,
            f"{CONST_GRAD_NORM}/pi": auxes[CONST_AUX][CONST_GRAD_NORM][
                CONST_POLICY
            ].item(),
        }

        gather_per_leaf_l2_norm(
            aux[CONST_LOG], "pi", self.model_dict[CONST_MODEL][CONST_POLICY]
        )

        for loss_key in self._config.losses:
            aux[CONST_LOG][f"losses/{loss_key}"] = auxes[CONST_AUX][loss_key][
                CONST_LOSS
            ].item()

        return aux

    @property
    def obs_rms(self):
        """
        Running statistics for observations.
        """
        return self._obs_rms

    def checkpoint(self, final=False) -> Dict[str, Any]:
        """
        Returns the parameters to checkpoint

        :param final: whether or not this is the final checkpoint
        :type final: bool (DefaultValue = False)
        :return: the checkpoint parameters
        :rtype: Dict[str, Any]

        """
        params = super().checkpoint(final=final)
        if self.obs_rms:
            params[CONST_OBS_RMS] = self.obs_rms.get_state()
        return params

    def load_checkpoint(self, params: Dict[str, Any]):
        """
        Loads a model state from a saved checkpoint.

        :param params: the checkpointed parameters
        :type params: Dict[str, Any]

        """
        super().load_checkpoint(params)
        if self.obs_rms:
            self._obs_rms.set_state(params[CONST_OBS_RMS])
