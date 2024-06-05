from types import SimpleNamespace
from typing import Any, Dict, Tuple, Sequence

import chex
import jax
import jax.random as jrandom
import numpy as np
import optax
import timeit

from jaxl.buffers import get_buffer
from jaxl.constants import *
from jaxl.learners.sac import SAC
from jaxl.losses.reinforcement import (
    make_cross_q_sac_qf_loss,
    make_sac_qf_loss,
    make_sac_pi_loss,
    make_sac_temp_loss,
)
from jaxl.models import (
    get_model,
    get_policy,
    get_q_function,
    get_update_function,
    q_function_dims,
    policy_output_dim,
    get_state_action_encoding,
    Temperature,
    get_fixed_policy,
)
from jaxl.optimizers import get_optimizer
from jaxl.utils import l2_norm, polyak_average_generator


"""
Standard SAC.
"""

QF_LOG_KEYS = [
    "mean_var_q",
    "min_var_q",
    "max_var_q",
    "max_next_q",
    "min_next_q",
    "mean_next_q",
    "max_curr_q",
    "min_curr_q",
    "mean_curr_q",
    "max_td_error",
    "min_td_error",
    "max_q_log_prob",
    "min_q_log_prob",
    "mean_q_log_prob",
]

PI_LOG_KEYS = [
    "max_policy_log_prob",
    "min_policy_log_prob",
    "mean_policy_log_prob",
    "max_estimated_value",
    "min_estimated_value",
    "mean_estimated_value",
]

TEMP_LOG_KEYS = [
    "max_policy_log_prob",
    "min_policy_log_prob",
    "mean_policy_log_prob",
    "max_temp_penalty",
    "min_temp_penalty",
    "mean_temp_penalty",
    "temperature",
]


class RLPDSAC(SAC):
    """
    Soft Actor Critic (SAC) algorithm with pretrained data. This extends `SAC`.
    """

    def _initialize_buffer(self):
        """
        Construct the buffer
        """
        h_state_dim = (1,)
        if hasattr(self._model_config, "h_state_dim"):
            h_state_dim = self._model_config.h_state_dim
        self._buffer = get_buffer(
            self._config.buffer_config,
            self._config.seeds.buffer_seed,
            self._env,
            h_state_dim,
        )
        self._demo_buffer = get_buffer(
            self._config.demo_buffer_config,
            self._config.seeds.buffer_seed,
            self._env,
            h_state_dim,
        )

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

        carried_steps = (
            self._global_step - self._buffer_warmup
        ) % self._update_frequency
        num_update_steps = (
            self._num_steps_per_epoch + carried_steps
        ) // self._update_frequency

        aux = {
            CONST_LOG: {},
        }

        # Initial exploration to populate the replay buffer
        if self._global_step < self._buffer_warmup:
            tic = timeit.default_timer()
            step_count = self._buffer_warmup - self._global_step
            self._rollout.rollout(
                None,
                self._exploration_pi,
                self._obs_rms,
                self._buffer,
                step_count,
            )
            self._global_step += step_count
            total_rollout_time += timeit.default_timer() - tic

        for update_i in range(num_update_steps):
            tic = timeit.default_timer()
            if update_i > 0:
                step_count = self._update_frequency
            else:
                step_count = self._update_frequency - carried_steps

            self._rollout.rollout(
                self._model_dict[CONST_MODEL][CONST_POLICY],
                self._pi,
                self._obs_rms,
                self._buffer,
                step_count,
            )
            self._global_step += step_count
            total_rollout_time += timeit.default_timer() - tic

            for _ in range(self._config.num_qf_updates):
                qf_auxes.append({})
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
                ) = self._buffer.sample_with_next_obs(batch_size=self._batch_size // 2)

                (
                    demo_obss,
                    demo_h_states,
                    demo_acts,
                    demo_rews,
                    _,
                    demo_terminateds,
                    _,
                    demo_next_obss,
                    demo_next_h_states,
                    _,
                    demo_lengths,
                    _,
                ) = self._demo_buffer.sample_with_next_obs(
                    batch_size=self._batch_size // 2
                )

                if self._config.remove_demo_absorbing_state:
                    demo_obss = demo_obss[..., :-1]
                    demo_next_obss = demo_next_obss[..., :-1]

                obss = np.vstack((obss, demo_obss))
                h_states = np.vstack((h_states, demo_h_states))
                acts = np.vstack((acts, demo_acts))
                rews = np.vstack((rews, demo_rews))
                terminateds = np.vstack((terminateds, demo_terminateds))
                next_obss = np.vstack((next_obss, demo_next_obss))
                next_h_states = np.vstack((next_h_states, demo_next_h_states))
                lengths = np.vstack((lengths, demo_lengths))

                obss = self.update_obs_rms_and_normalize(obss, lengths)
                total_sampling_time += timeit.default_timer() - tic

                tic = timeit.default_timer()
                self._learning_key, qf_keys, pi_keys, temp_keys = jrandom.split(
                    self._learning_key, num=4
                )
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
                self._model_dict[CONST_OPT_STATE][CONST_QF] = qf_model_dict[
                    CONST_OPT_STATE
                ]
                assert np.isfinite(
                    qf_aux[CONST_AGG_LOSS]
                ), f"Loss became NaN\nqf_aux: {qf_aux}"
                self._num_qf_updates += 1
                total_qf_update_time += timeit.default_timer() - tic

                qf_auxes[-1] = qf_aux
                if self._num_qf_updates % self._target_update_frequency == 0:
                    tic = timeit.default_timer()
                    self._model_dict[CONST_MODEL][CONST_TARGET_QF] = (
                        self.update_target_model(self._model_dict)
                    )
                    total_target_qf_update_time += timeit.default_timer() - tic

                # Update Actor
                if self._num_qf_updates % self._actor_update_frequency == 0:
                    pi_auxes.append({})
                    tic = timeit.default_timer()
                    pi_model_dict, pi_aux = self.pi_step(
                        self._model_dict,
                        obss,
                        h_states,
                        pi_keys,
                    )
                    assert np.isfinite(
                        pi_aux[CONST_AGG_LOSS]
                    ), f"Loss became NaN\npi_aux: {pi_aux}"
                    self._model_dict[CONST_MODEL][CONST_POLICY] = pi_model_dict[
                        CONST_MODEL
                    ]
                    self._model_dict[CONST_OPT_STATE][CONST_POLICY] = pi_model_dict[
                        CONST_OPT_STATE
                    ]
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
                            temp_keys,
                        )
                        assert np.isfinite(
                            temp_aux[CONST_AGG_LOSS]
                        ), f"Loss became NaN\ntemp_aux: {temp_aux}"
                        self._model_dict[CONST_MODEL][CONST_TEMPERATURE] = (
                            temp_model_dict[CONST_MODEL]
                        )
                        self._model_dict[CONST_OPT_STATE][CONST_TEMPERATURE] = (
                            temp_model_dict[CONST_OPT_STATE]
                        )
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

        aux[CONST_LOG][f"interaction/{CONST_AVERAGE_RETURN}"] = (
            self._rollout.latest_average_return(num_episodes=10)
        )
        aux[CONST_LOG][f"interaction/{CONST_AVERAGE_EPISODE_LENGTH}"] = (
            self._rollout.latest_average_episode_length(num_episodes=10)
        )

        if qf_auxes:
            qf_auxes = jax.tree_util.tree_map(lambda *args: np.mean(args), *qf_auxes)
            aux[CONST_LOG][f"losses/{CONST_QF}"] = qf_auxes[CONST_AGG_LOSS].item()
            aux[CONST_LOG][f"{CONST_GRAD_NORM}/{CONST_QF}"] = qf_auxes[CONST_GRAD_NORM][
                CONST_QF
            ].item()
            aux[CONST_LOG][f"{CONST_PARAM_NORM}/{CONST_QF}"] = l2_norm(
                self.model_dict[CONST_MODEL][CONST_QF]
            ).item()
            aux[CONST_LOG][f"{CONST_PARAM_NORM}/{CONST_TARGET_QF}"] = l2_norm(
                self.model_dict[CONST_MODEL][CONST_TARGET_QF]
            ).item()

            for act_i in range(acts.shape[-1]):
                aux[CONST_LOG][
                    f"{CONST_ACTION}/{CONST_ACTION}_{act_i}_{CONST_SATURATION}"
                ] = qf_auxes[CONST_ACTION][act_i][CONST_SATURATION]
                aux[CONST_LOG][
                    f"{CONST_ACTION}/{CONST_ACTION}_{act_i}_{CONST_MEAN}"
                ] = qf_auxes[CONST_ACTION][act_i][CONST_MEAN]

            for key in QF_LOG_KEYS:
                aux[CONST_LOG][f"{CONST_QF}_info/{key}"] = qf_auxes[key].item()

        if pi_auxes:
            pi_auxes = jax.tree_util.tree_map(lambda *args: np.mean(args), *pi_auxes)

            aux[CONST_LOG][f"losses/{CONST_POLICY}"] = pi_auxes[CONST_AGG_LOSS].item()
            aux[CONST_LOG][f"{CONST_GRAD_NORM}/{CONST_POLICY}"] = pi_auxes[
                CONST_GRAD_NORM
            ][CONST_POLICY].item()
            aux[CONST_LOG][f"{CONST_PARAM_NORM}/{CONST_POLICY}"] = l2_norm(
                self.model_dict[CONST_MODEL][CONST_POLICY]
            ).item()

            for key in PI_LOG_KEYS:
                aux[CONST_LOG][f"{CONST_POLICY}_info/{key}"] = pi_auxes[key].item()

        if temp_auxes:
            temp_auxes = jax.tree_util.tree_map(
                lambda *args: np.mean(args), *temp_auxes
            )

            aux[CONST_LOG][f"losses/{CONST_TEMPERATURE}"] = temp_auxes[
                CONST_AGG_LOSS
            ].item()
            aux[CONST_LOG][f"{CONST_GRAD_NORM}/{CONST_TEMPERATURE}"] = temp_auxes[
                CONST_GRAD_NORM
            ][CONST_TEMPERATURE].item()

            for key in TEMP_LOG_KEYS:
                aux[CONST_LOG][f"{CONST_TEMPERATURE}_info/{key}"] = temp_auxes[
                    key
                ].item()

        self.gather_rms(aux)
        return aux
