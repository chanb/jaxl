from types import SimpleNamespace
from typing import Any, Dict, Tuple, Union

import _pickle as pickle
import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import math
import numpy as np
import optax
import os
import timeit

from jaxl.buffers import get_buffer
from jaxl.constants import *
from jaxl.learners.learner import OfflineLearner
from jaxl.losses import get_loss_function, make_aggregate_loss
from jaxl.models import (
    get_model,
    get_optimizer,
    policy_output_dim,
)
from jaxl.utils import l2_norm, parse_dict, RunningMeanStd


"""
Naive Multitask Behavioural Cloning.
"""


class MTBC(OfflineLearner):
    """
    Naive Multitask Behavioural Cloning (BC) algorithm. This extends `OfflineLearner`.
    """

    _obs_rms: Union[bool, RunningMeanStd]
    _num_tasks: int

    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        super().__init__(config, model_config, optimizer_config)
        self._initialize_losses()

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

        def loss(
            model_dicts: Dict[str, Any],
            obss: chex.Array,
            h_states: chex.Array,
            acts: chex.Array,
            *args,
            **kwargs,
        ) -> Tuple[chex.Array, Dict]:
            """
            Aggregates the behavioural cloning loss for each task.

            :param model_dict: the actor and critic states and their optimizers state
            :param obss: the training observations
            :param h_states: the training hidden states for memory-based models
            :param acts: the training target actions
            :param *args:
            :param **kwargs:
            :type model_dict: Dict[str, Any]
            :type obss: chex.Array
            :type h_states: chex.Array
            :type acts: chex.Array
            :return: the aggregate loss and auxiliary information
            :rtype: Tuple[chex.Array, Dict[str, Any]]

            """
            reps, _ = jax.vmap(
                self._model[CONST_REPRESENTATION].forward, in_axes=[None, 0, 0]
            )(model_dicts[CONST_REPRESENTATION], obss, h_states)

            bc_loss, bc_aux = jax.vmap(self._pi_loss)(
                model_dicts[CONST_POLICY],
                reps,
                h_states,
                acts,
            )

            agg_loss = jnp.mean(bc_loss)

            return agg_loss, bc_aux

        self._loss = loss
        self.train_step = jax.jit(self.make_train_step())

    @property
    def num_tasks(self):
        """Number of tasks."""
        return self._num_tasks

    def _initialize_losses(self):
        """
        Construct the policy losses.
        We assume the each policy uses the same losses and same coefficients.
        """
        losses = {}
        for loss, loss_setting in zip(self._config.losses, self._config.loss_settings):
            loss_setting_ns = parse_dict(loss_setting)
            losses[loss] = (
                get_loss_function(self._model, loss, loss_setting_ns),
                loss_setting_ns.coefficient,
            )
        self._pi_loss = jax.jit(make_aggregate_loss(losses))

    def _initialize_buffer(self):
        """
        Construct the buffers
        """
        self._buffer = [
            get_buffer(buffer_config, self._config.seeds.buffer_seed)
            for buffer_config in self._config.buffer_configs
        ]
        input_dims = [buffer.input_dim for buffer in self._buffer]
        output_dims = [buffer.output_dim for buffer in self._buffer]
        assert all(
            input_dims[:-1] == input_dims[1:]
        ), "We assume the observation space to be the same for all tasks."

        # XXX: We should be able to relax this in the future.
        assert all(
            output_dims[:-1] == output_dims[1:]
        ), "We assume the action space to be the same for all tasks."
        self._num_tasks = len(self._buffer)

    def _generate_dummy_x(
        self, num_tasks: int, input_dim: chex.Array, representation_dim: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Generates an arbitrary input based on the input space and
        an arbitrary input based on the representation space.
        This is mainly for initializing the model.

        :param num_tasks: number of tasks
        :param input_dim: the input dimension
        :param representation_dim: the representation dimension
        :type num_tasks: int
        :type input_dim: chex.Array
        :type representation_dim: chex.Array
        :return: arbitrary inputs for representation and policy
        :rtype: Tuple[chex.Array, chex.Array]

        """
        return np.zeros((1, 1, *input_dim)), np.zeros(
            (num_tasks, 1, *representation_dim)
        )

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

        model_keys = jrandom.split(
            jrandom.PRNGKey(self._config.seeds.model_seed), num=len(self._buffer) + 1
        )
        dummy_input, dummy_representation = self._generate_dummy_x(
            self.num_tasks, input_dim, representation_dim
        )
        pi_params = jax.vmap(self._model[CONST_POLICY].init)(
            model_keys[:-1], dummy_representation
        )
        pi_opt_state = self._optimizer[CONST_POLICY].init(pi_params)

        representation_params = self._model[CONST_REPRESENTATION].init(
            model_keys[-1], dummy_input
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
                model_dict[CONST_MODEL],
                train_x,
                train_carry,
                train_y,
            )
            aux[CONST_AGG_LOSS] = agg_loss
            aux[CONST_GRAD_NORM] = {
                CONST_POLICY: jax.vmap(l2_norm)(grads[CONST_POLICY]),
                CONST_REPRESENTATION: l2_norm(grads[CONST_REPRESENTATION]),
            }

            updates, pi_opt_state = self._optimizer[CONST_POLICY].update(
                grads[CONST_POLICY],
                model_dict[CONST_OPT_STATE][CONST_POLICY],
                model_dict[CONST_MODEL][CONST_POLICY],
            )
            pi_params = optax.apply_updates(
                model_dict[CONST_MODEL][CONST_POLICY], updates
            )

            updates, representation_opt_state = self._optimizer[
                CONST_REPRESENTATION
            ].update(
                grads[CONST_REPRESENTATION],
                model_dict[CONST_OPT_STATE][CONST_REPRESENTATION],
                model_dict[CONST_MODEL][CONST_REPRESENTATION],
            )
            representation_params = optax.apply_updates(
                model_dict[CONST_MODEL][CONST_REPRESENTATION], updates
            )

            return {
                CONST_MODEL: {
                    CONST_POLICY: pi_params,
                    CONST_REPRESENTATION: representation_params,
                },
                CONST_OPT_STATE: {
                    CONST_POLICY: pi_opt_state,
                    CONST_REPRESENTATION: representation_opt_state,
                },
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
            all_obss = []
            all_h_states = []
            all_acts = []

            for buffer in self.buffer:
                obss, h_states, acts, _, _, _, _, _, _, _ = buffer.sample(
                    self._config.batch_size
                )
                all_obss.append(obss)
                all_h_states.append(h_states)
                all_acts.append(acts)

            all_obss = np.stack(all_obss)
            all_h_states = np.stack(all_h_states)
            all_acts = np.stack(all_acts)

            self.model_dict, aux = self.train_step(
                self._model_dict, all_obss, all_h_states, all_acts
            )
            total_update_time += timeit.default_timer() - tic
            assert np.isfinite(aux[CONST_AGG_LOSS]), f"Loss became NaN\naux: {aux}"

            auxes[-1][CONST_AUX] = aux

        auxes = jax.tree_util.tree_map(lambda *args: np.mean(args), *auxes)
        aux[CONST_LOG] = {
            f"losses/{CONST_AGG_LOSS}": auxes[CONST_AUX][CONST_AGG_LOSS].item(),
            f"time/{CONST_UPDATE_TIME}": total_update_time,
            f"{CONST_PARAM_NORM}/representation": l2_norm(
                self.model_dict[CONST_MODEL][CONST_REPRESENTATION]
            ).item(),
            f"{CONST_GRAD_NORM}/representation": auxes[CONST_AUX][CONST_GRAD_NORM][
                CONST_REPRESENTATION
            ].item(),
        }

        policy_param_norms = jax.vmap(l2_norm)(
            self.model_dict[CONST_MODEL][CONST_POLICY]
        )
        for task_i in range(self.num_tasks):
            aux[CONST_LOG][f"{CONST_PARAM_NORM}/pi_{task_i}"] = policy_param_norms[
                task_i
            ].item()
            aux[CONST_LOG][f"{CONST_PARAM_NORM}/pi_{task_i}"] = auxes[CONST_AUX][
                CONST_GRAD_NORM
            ][CONST_POLICY][task_i].item()

            for loss_key in self._config.losses:
                aux[CONST_LOG][f"losses/{loss_key}_{task_i}"] = auxes[CONST_AUX][
                    loss_key
                ][CONST_LOSS][task_i].item()

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