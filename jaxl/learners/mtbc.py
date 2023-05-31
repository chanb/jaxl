from types import SimpleNamespace
from typing import Any, Dict, Tuple, Union

import _pickle as pickle
import chex
import jax
import jax.random as jrandom
import math
import numpy as np
import optax
import os
import timeit

from jaxl.buffers import get_buffer
from jaxl.constants import *
from jaxl.learners.learner import OfflineLearner
from jaxl.models import (
    get_model,
    get_optimizer,
    get_policy,
    policy_output_dim,
)
from jaxl.utils import l2_norm, RunningMeanStd


"""
Naive Multitask Behavioural Cloning.
"""


class MTBC(OfflineLearner):
    """
    Naive Multitask Behavioural Cloning (BC) algorithm. This extends `OfflineLearner`.
    """

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
            self._obs_rms = RunningMeanStd(shape=self._buffer[0].input_dim)

            for buffer in self._buffer:
                # We compute the statistics offline by iterating through the whole buffer once
                num_batches = math.ceil(len(buffer) / self._config.batch_size)
                last_batch_remainder = len(buffer) % self._config.batch_size
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
                    obss, _, _, _, _, _, _, _, _, _ = buffer.sample(
                        batch_size, sample_idxes
                    )
                    self._obs_rms.update(obss)

    def _initialize_buffer(self):
        """
        Construct the buffers
        """
        self._buffer = [
            get_buffer(buffer_config, self._config.seeds.buffer_seed)
            for buffer_config in self._config.buffer_configs
        ]
        input_dims = [buffer.input_dim for buffer in self._buffer]
        assert all(
            input_dims[:-1] == input_dims[1:]
        ), "We assume the observation space to be the same for all tasks."

    # TODO: Generate dummy for both input and representation
    def _generate_dummy_x(self) -> chex.Array:
        """
        Generates an arbitrary input based on the buffer's input space.
        This is mainly for initializing the model.

        :return: an arbitrary input
        :rtype: chex.Array

        """
        return np.zeros((1, 1, *self._buffer.input_dim))

    # TODO: Shared representation
    def _initialize_model_and_opt(self, input_dim: chex.Array, output_dim: chex.Array):
        """
        Construct the model and the optimizer.

        :param input_dim: input dimension of the data point
        :param output_dim: output dimension of the data point
        :type input_dim: chex.Array
        :type output_dim: chex.Array

        """
        act_dim = policy_output_dim(output_dim, self._config)
        representation_dim = self._model_config.representation_dim
        self._model = {
            CONST_POLICY: get_model(
                representation_dim, act_dim, self._model_config.policy
            ),
            CONST_REPRESENTATION: get_model(
                input_dim, representation_dim, self._model_config.representation
            ),
        }

        self._optimizer = {
            CONST_POLICY: get_optimizer(self._optimizer_config.policy),
            CONST_REPRESENTATION: get_optimizer(self._optimizer_config.representation),
        }

        model_key = jrandom.PRNGKey(self._config.seeds.model_seed)
        dummy_x = self._generate_dummy_x()
        pi_params = self._model[CONST_POLICY].init(model_key, dummy_x)
        pi_opt_state = self._optimizer[CONST_POLICY].init(pi_params)

        representation_params = self._model[CONST_REPRESENTATION].init(
            model_key, dummy_x
        )
        representation_opt_state = self._optimizer[CONST_REPRESENTATION].init(
            representation_params
        )
        self._model_dict = {
            CONST_MODEL: {
                CONST_POLICY: pi_params,
                CONST_REPRESENTATION: representation_params,
            },
            CONST_OPT_STATE: {
                CONST_POLICY: pi_opt_state,
                CONST_REPRESENTATION: representation_opt_state,
            },
        }

    def make_train_step(self):
        """
        Makes the training step for model update.
        """

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
            updates, opt_state = self._optimizer.update(
                grads,
                model_dict[CONST_OPT_STATE][CONST_POLICY],
                model_dict[CONST_MODEL][CONST_POLICY],
            )
            aux[CONST_GRAD_NORM] = {CONST_POLICY: l2_norm(grads)}
            params = optax.apply_updates(model_dict[CONST_MODEL][CONST_POLICY], updates)
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
            """
            TODO
            1. Sample from multiple buffers and merge them.
            2. Construct a mask so that only the i'th buffer maps to the i'th policy
            """
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
            f"{CONST_PARAM_NORM}/pi": l2_norm(
                self.model_dict[CONST_MODEL][CONST_POLICY]
            ).item(),
            f"{CONST_GRAD_NORM}/pi": auxes[CONST_AUX][CONST_GRAD_NORM][
                CONST_POLICY
            ].item(),
        }
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

    def checkpoint(self, checkpoint_path: str, exist_ok: bool = False):
        """
        Saves the current model state.

        :param checkpoint_path: directory path to store the checkpoint to
        :param exist_ok: whether to overwrite the existing checkpoint path
        :type checkpoint_path: str
        :type exist_ok: bool (Default value = False)

        """
        super().checkpoint(checkpoint_path, exist_ok)
        learner_dict = {
            CONST_OBS_RMS: self.obs_rms,
        }

        with open(os.path.join(checkpoint_path, "learner_dict.pkl"), "wb") as f:
            pickle.dump(learner_dict, f)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Loads a model state from a saved checkpoint.

        :param checkpoint_path: directory path to load the checkpoint from
        :type checkpoint_path: str

        """
        super().load_checkpoint(checkpoint_path)

        with open(os.path.join(checkpoint_path, "learner_dict.pkl"), "rb") as f:
            learner_dict = pickle.load(f)
            self._obs_rms = learner_dict[CONST_OBS_RMS]
