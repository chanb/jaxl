from types import SimpleNamespace
from typing import Any, Dict, Tuple, Sequence

import chex
import jax
import jax.random as jrandom
import numpy as np
import optax
import timeit

from jaxl.constants import *
from jaxl.learners.reinforcement import OffPolicyLearner
from jaxl.losses.reinforcement import make_sac_qf_loss
from jaxl.models import (
    get_model,
    get_optimizer,
    get_policy,
    get_q_function,
    get_update_function,
    q_function_dims,
    policy_output_dim,
    get_state_action_encoding,
    Temperature,
)
from jaxl.utils import l2_norm, polyak_average_generator


"""
Standard SAC.
"""


class SAC(OffPolicyLearner):
    """
    Soft Actor Critic (SAC) algorithm. This extends `OffPolicyLearner`.
    """

    _target_entropy: float
    _num_qf_updates: int

    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        super().__init__(config, model_config, optimizer_config)
        self.initialize_exploration_strategy()

        self._actor_update_frequency = getattr(config, CONST_ACTOR_UPDATE_FREQUENCY, 1)
        self._target_update_frequency = getattr(
            config, CONST_TARGET_UPDATE_FREQUENCY, 1
        )
        self._buffer_warmup = config.buffer_warmup

        self._num_qf_updates = 0

        self._qf_loss = make_sac_qf_loss(self._agent, self._config.qf_loss_setting)

        self.qf_step = self.make_qf_step()

        self.polyak_average = polyak_average_generator(config.tau)

    @property
    def policy(self):
        """
        Policy.
        """
        return self._pi

    @property
    def policy_params(self):
        """
        Policy parameters.
        """
        return self._model_dict[CONST_MODEL][CONST_POLICY]

    def initialize_exploration_strategy(self):
        """
        Creates a exploration agent for collecting the initial transitions for replay buffer.
        """
        

    def _initialize_model_and_opt(self, input_dim: chex.Array, output_dim: chex.Array):
        """
        Construct the actor and critic, and their corresponding optimizers.

        :param input_dim: input dimension of the data point
        :param output_dim: output dimension of the data point
        :type input_dim: chex.Array
        :type output_dim: chex.Array

        """
        act_dim = policy_output_dim(output_dim, self._config)

        encoding_input_dim = (1, *input_dim)
        encoding_output_dim = output_dim[:-1]
        qf_input_dim, qf_output_dim = q_function_dims(
            encoding_input_dim, encoding_output_dim, self._model_config.qf_encoding
        )
        self._model = {
            CONST_POLICY: get_model(input_dim, act_dim, self._model_config.policy),
            CONST_QF: get_model(qf_input_dim, qf_output_dim, self._model_config.qf),
            CONST_TARGET_QF: get_model(
                qf_input_dim, qf_output_dim, self._model_config.qf
            ),
        }

        # Initialize parameters for policy and critics
        model_keys = jrandom.split(
            jrandom.PRNGKey(self._config.seeds.model_seed), num=4
        )
        dummy_x = self._generate_dummy_x(input_dim)
        pi_params = self._model[CONST_POLICY].init(model_keys[0], dummy_x)

        dummy_x = self._generate_dummy_x(qf_input_dim)
        qf_params = self._model[CONST_QF].init(model_keys[1], dummy_x)
        target_qf_params = self._model[CONST_TARGET_QF].init(model_keys[1], dummy_x)

        self._state_action_encoding = get_state_action_encoding(
            encoding_input_dim, encoding_output_dim, self._model_config.qf_encoding
        )

        enc_params = self._state_action_encoding.init(
            model_keys[2],
            {
                CONST_OBSERVATION: self._generate_dummy_x(encoding_input_dim),
                CONST_ACTION: self._generate_dummy_x(encoding_output_dim),
            },
        )

        pi_opt, pi_opt_state = get_optimizer(
            self._optimizer_config.policy, self._model[CONST_POLICY], pi_params
        )
        qf_opt, qf_opt_state = get_optimizer(
            self._optimizer_config.qf, self._model[CONST_QF], qf_params
        )

        self._optimizer = {
            CONST_POLICY: pi_opt,
            CONST_QF: qf_opt,
        }

        self._model_dict = {
            CONST_MODEL: {
                CONST_POLICY: pi_params,
                CONST_QF: qf_params,
                CONST_TARGET_QF: target_qf_params,
            },
            CONST_OPT_STATE: {
                CONST_POLICY: pi_opt_state,
                CONST_QF: qf_opt_state,
            },
        }

        self._pi = get_policy(self._model[CONST_POLICY], self._config)
        self._qf = get_q_function(
            self._state_action_encoding,
            enc_params,
            self._model[CONST_QF],
            self._model_config.qf_encoding,
        )
        self._target_qf = get_q_function(
            self._state_action_encoding,
            enc_params,
            self._model[CONST_TARGET_QF],
            self._model_config.qf_encoding,
        )

        self._agent = {
            CONST_POLICY: self._pi,
            CONST_QF: self._qf,
            CONST_TARGET_QF: self._target_qf,
        }

        # Temperature
        self._target_entropy = getattr(self._config, CONST_TARGET_ENTROPY, CONST_AUTO)
        if self._target_entropy == CONST_AUTO:
            self._target_entropy = -int(np.product(output_dim))

        if self._target_entropy is not None:
            self._model[CONST_TEMPERATURE] = Temperature(
                self._config.initial_temperature
            )
            temp_params = self._model[CONST_TEMPERATURE].init(model_keys[3])
            temp_opt, temp_opt_state = get_optimizer(
                self._optimizer_config.temp, self._model[CONST_TEMPERATURE], temp_params
            )

            self._optimizer[CONST_TEMPERATURE] = temp_opt
            self._model_dict[CONST_MODEL][CONST_TEMPERATURE] = temp_params
            self._model_dict[CONST_OPT_STATE][CONST_TEMPERATURE] = temp_opt_state
            self._agent[CONST_TEMPERATURE] = self._model[CONST_TEMPERATURE]

    def update_target_model(self):
        self._model_dict[CONST_TARGET_QF] = jax.tree_map(
            self.polyak_average,
            self._model_dict[CONST_MODEL][CONST_QF],
            self._model_dict[CONST_MODEL][CONST_TARGET_QF],
        )

    def make_qf_step(self):
        """
        Makes the training step for the critic update.
        """

        qf_update = get_update_function(self._model[CONST_QF])

        def _qf_step(
            model_dict: Dict[str, Any],
            obss: chex.Array,
            h_states: chex.Array,
            acts: chex.Array,
            rews: chex.Array,
            terminateds: chex.Array,
            next_obss: chex.Array,
            next_h_states: chex.Array,
            keys: Sequence[jrandom.PRNGKey],
            *args,
            **kwargs,
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            """
            The training step that computes the A2C loss and performs actor update and critic update.

            :param model_dict: the model state and optimizer state
            :param obss: the training observations
            :param h_states: the training hidden states for memory-based models
            :param obss: current observations
            :param h_states: current hidden states
            :param acts: actions taken for current observations
            :param rews: received rewards
            :param terminateds: whether or not the episode is terminated
            :param next_obss: next observations
            :param next_h_states: next hidden states
            :param keys: random keys for sampling next actions
            :param *args:
            :param **kwargs:
            :type model_dict: Dict[str, Any]
            :type obss: chex.Array
            :type h_states: chex.Array
            :type acts: chex.Array
            :type rews: chex.Array
            :type terminateds: chex.Array
            :type next_obss: chex.Array
            :type next_h_states: chex.Array
            :type keys: Sequence[jrandom.PRNGKey]
            :return: the updated critic state, the corresponding updated
                     optimizer state, and auxiliary information
            :rtype: Tuple[Dict[str, Any], Dict[str, Any]]

            """
            (agg_loss, aux), grads = jax.value_and_grad(self._qf_loss, has_aux=True)(
                model_dict[CONST_MODEL][CONST_QF],
                model_dict[CONST_MODEL][CONST_TARGET_QF],
                model_dict[CONST_MODEL][CONST_POLICY],
                model_dict[CONST_MODEL][CONST_TEMPERATURE],
                obss,
                h_states,
                acts,
                rews,
                terminateds,
                next_obss,
                next_h_states,
                self._gamma,
                keys,
            )

            aux[CONST_AGG_LOSS] = agg_loss
            aux[CONST_GRAD_NORM] = {
                CONST_QF: l2_norm(grads),
            }

            qf_params, qf_opt_state = qf_update(
                self._optimizer[CONST_QF],
                grads,
                model_dict[CONST_OPT_STATE][CONST_QF],
                model_dict[CONST_MODEL][CONST_QF],
            )

            return {
                CONST_MODEL: qf_params,
                CONST_OPT_STATE: qf_opt_state,
            }, aux

        return _qf_step

    def update(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Updates the actor and the critic.

        :param *args:
        :param **kwargs:
        :return: the update information
        :rtype: Dict[str, Any]

        """

        qf_auxes = []
        pi_auxes = []
        temp_auxes = []
        total_rollout_time = 0
        total_sampling_time = 0
        total_qf_update_time = 0
        total_pi_update_time = 0
        total_temp_update_time = 0
        total_target_qf_update_time = 0

        carried_steps = self._global_step % self._update_frequency
        num_update_steps = (
            self._num_steps_per_epoch + carried_steps
        ) // self._update_frequency

        aux = {
            CONST_LOG: {},
        }

        # Initial exploration to populate the replay buffer
        if self._global_step < self._buffer_warmup:
            step_count = self._buffer_warmup - self._global_step

        for update_i in range(num_update_steps):
            qf_auxes.append({})
            tic = timeit.default_timer()
            if update_i > 0:
                step_count = self._update_frequency
            else:
                step_count = self._update_frequency - carried_steps

            # TODO: Include exploration strategy
            self._rollout.rollout(
                self._model_dict[CONST_MODEL][CONST_POLICY],
                self._pi,
                self._obs_rms,
                self._buffer,
                step_count,
            )
            self._global_step += step_count
            total_rollout_time += timeit.default_timer() - tic

            # TODO: Extend this to higher UTD ratio
            tic = timeit.default_timer()
            (
                obss,
                h_states,
                acts,
                rews,
                _,
                terminateds,
                _,
                next_obss,
                next_h_states,
                _,
                lengths,
                _,
            ) = self._buffer.sample_with_next_obs(batch_size=self._batch_size)
            obss = self.update_obs_rms_and_normalize(obss, lengths)
            total_sampling_time += timeit.default_timer() - tic

            tic = timeit.default_timer()
            qf_keys = jrandom.split(jrandom.PRNGKey(self._global_step), num=1)
            qf_model_dict, qf_aux = self.qf_step(
                self._model_dict,
                obss,
                h_states,
                acts,
                rews,
                terminateds,
                next_obss,
                next_h_states,
                qf_keys,
            )
            self._model_dict[CONST_MODEL][CONST_QF] = qf_model_dict[CONST_MODEL]
            self._model_dict[CONST_OPT_STATE][CONST_QF] = qf_model_dict[CONST_OPT_STATE]
            assert np.isfinite(
                qf_aux[CONST_AGG_LOSS]
            ), f"Loss became NaN\nqf_aux: {qf_aux}"
            self._num_qf_updates += 1
            total_qf_update_time += timeit.default_timer() - tic

            qf_auxes[-1] = qf_aux
            if self._num_qf_updates % self._target_update_frequency == 0:
                tic = timeit.default_timer()
                self.update_target_model()
                total_target_qf_update_time += timeit.default_timer() - tic

            # Update Actor
            if self._num_qf_updates % self._actor_update_frequency == 0:
                pi_auxes.append({})
                tic = timeit.default_timer()
                pi_model_dict, pi_aux = self.pi_step(
                    self._model_dict,
                    obss,
                    h_states,
                    acts,
                )
                assert np.isfinite(
                    pi_aux[CONST_AGG_LOSS]
                ), f"Loss became NaN\npi_aux: {pi_aux}"
                total_pi_update_time += timeit.default_timer() - tic
                pi_auxes[-1] = pi_aux

                # Update temperature
                if self._target_entropy is not None:
                    temp_auxes.append({})
                    tic = timeit.default_timer()
                    temp_model_dict, temp_aux = self.temp_step(
                        self._model_dict,
                        obss,
                        h_states,
                    )
                    assert np.isfinite(
                        temp_aux[CONST_AGG_LOSS]
                    ), f"Loss became NaN\ntemp_aux: {temp_aux}"
                    total_temp_update_time += timeit.default_timer() - tic
                    temp_auxes[-1] = temp_aux

            qf_auxes[-1][CONST_ACTION] = {
                i: {
                    CONST_SATURATION: np.abs(acts[..., i]).max(),
                    CONST_MEAN: np.abs(acts[..., i]).mean(),
                }
                for i in range(acts.shape[-1])
            }

        aux[CONST_LOG][f"time/{CONST_ROLLOUT_TIME}"] = total_rollout_time
        aux[CONST_LOG][f"time/{CONST_SAMPLING_TIME}"] = total_sampling_time
        aux[CONST_LOG][f"time/update_{CONST_QF}"] = total_qf_update_time
        aux[CONST_LOG][f"time/update_{CONST_TARGET_QF}"] = total_target_qf_update_time
        aux[CONST_LOG][f"time/update_{CONST_POLICY}"] = total_pi_update_time
        aux[CONST_LOG][f"time/update_{CONST_TEMPERATURE}"] = total_temp_update_time

        aux[CONST_LOG][
            f"interaction/{CONST_AVERAGE_RETURN}"
        ] = self._rollout.latest_average_return()
        aux[CONST_LOG][
            f"interaction/{CONST_AVERAGE_EPISODE_LENGTH}"
        ] = self._rollout.latest_average_episode_length()

        # TODO: Fix logging for different training
        if qf_auxes:
            qf_auxes = jax.tree_util.tree_map(lambda *args: np.mean(args), *qf_auxes)
            aux[CONST_LOG][f"losses/qf"] = qf_auxes[CONST_AUX][CONST_QF].item()
            aux[CONST_LOG][f"{CONST_GRAD_NORM}/qf"] = qf_auxes[CONST_AUX][CONST_GRAD_NORM][
                CONST_QF
            ].item()
            aux[CONST_LOG][f"{CONST_PARAM_NORM}/qf"] = l2_norm(
                self.model_dict[CONST_MODEL][CONST_QF]
            ).item()

            for act_i in range(acts.shape[-1]):
                aux[CONST_LOG][
                    f"{CONST_ACTION}/{CONST_ACTION}_{act_i}_{CONST_SATURATION}"
                ] = qf_auxes[CONST_ACTION][act_i][CONST_SATURATION]
                aux[CONST_LOG][
                    f"{CONST_ACTION}/{CONST_ACTION}_{act_i}_{CONST_MEAN}"
                ] = qf_auxes[CONST_ACTION][act_i][CONST_MEAN]

        if pi_auxes:
            pi_auxes = jax.tree_util.tree_map(lambda *args: np.mean(args), *pi_auxes)

        if temp_auxes:
            temp_auxes = jax.tree_util.tree_map(
                lambda *args: np.mean(args), *temp_auxes
            )

        self.gather_rms(aux)
        return aux
