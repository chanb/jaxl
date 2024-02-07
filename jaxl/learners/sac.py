from types import SimpleNamespace
from typing import Any, Dict, Tuple

import chex
import jax
import jax.random as jrandom
import numpy as np
import optax
import timeit

from jaxl.constants import *
from jaxl.learners.reinforcement import OffPolicyLearner
from jaxl.losses.reinforcement import monte_carlo_returns, make_reinforce_loss
from jaxl.losses.supervised import make_squared_loss
from jaxl.models import (
    get_model,
    get_optimizer,
    get_policy,
    get_q_function,
    get_update_function,
    q_function_dims,
    policy_output_dim,
)
from jaxl.utils import l2_norm, polyak_average_generator


"""
Standard SAC.
"""


class SAC(OffPolicyLearner):
    """
    Soft Actor Critic (SAC) algorithm. This extends `OffPolicyLearner`.
    """

    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        super().__init__(config, model_config, optimizer_config)

        self._pi = get_policy(self._model[CONST_POLICY], config)
        self._qf = get_q_function(self._model[CONST_QF], config)
        self._target_qf = get_q_function(self._model[CONST_TARGET_QF], config)

        self._pi_loss = make_reinforce_loss(self._pi, self._config.pi_loss_setting)
        self._vf_loss = make_squared_loss(self._vf, self._config.vf_loss_setting)

        def joint_loss(
            model_dicts: Dict[str, Any],
            obss: chex.Array,
            h_states: chex.Array,
            acts: chex.Array,
            rets: chex.Array,
            *args,
            **kwargs,
        ) -> Tuple[chex.Array, Dict[str, Any]]:
            """
            Aggregates the actor loss and the critic loss.

            :param model_dict: the actor and critic states and their optimizers state
            :param obss: the training observations
            :param h_states: the training hidden states for memory-based models
            :param acts: the training actions
            :param rets: the Monte-Carlo returns
            :param *args:
            :param **kwargs:
            :type model_dict: Dict[str, Any]
            :type obss: chex.Array
            :type h_states: chex.Array
            :type acts: chex.Array
            :type rets: chex.Array
            :return: the aggregate loss and auxiliary information
            :rtype: Tuple[chex.Array, Dict[str, Any]]

            """
            vf_loss, vf_aux = self._vf_loss(
                model_dicts[CONST_VF],
                obss,
                h_states,
                rets,
            )

            baselines = jax.lax.stop_gradient(vf_aux[CONST_PREDICTIONS])
            pi_loss, pi_aux = self._pi_loss(
                model_dicts[CONST_POLICY],
                obss,
                h_states,
                acts,
                rets,
                baselines,
            )

            agg_loss = (
                self._config.pi_loss_setting.coefficient * pi_loss
                + self._config.vf_loss_setting.coefficient * vf_loss
            )

            aux = {
                CONST_POLICY: pi_loss,
                CONST_VF: vf_loss,
                CONST_LOG_PROBS: pi_aux[CONST_LOG_PROBS],
                CONST_ADVANTAGE: (rets - baselines).mean(),
            }
            return agg_loss, aux

        self._joint_loss = joint_loss
        self.joint_step = jax.jit(self.make_joint_step())

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

    def _initialize_model_and_opt(self, input_dim: chex.Array, output_dim: chex.Array):
        """
        Construct the actor and critic, and their corresponding optimizers.

        :param input_dim: input dimension of the data point
        :param output_dim: output dimension of the data point
        :type input_dim: chex.Array
        :type output_dim: chex.Array

        """
        act_dim = policy_output_dim(output_dim, self._config)
        self._model = {
            CONST_POLICY: get_model(input_dim, act_dim, self._model_config.policy),
            CONST_QF: get_model(input_dim, (1,), self._model_config.qf),
            CONST_TARGET_QF: get_model(input_dim, (1,), self._model_config.qf),
        }

        model_keys = jrandom.split(jrandom.PRNGKey(self._config.seeds.model_seed))
        dummy_x = self._generate_dummy_x(input_dim)
        pi_params = self._model[CONST_POLICY].init(model_keys[0], dummy_x)
        vf_params = self._model[CONST_VF].init(model_keys[1], dummy_x)

        pi_opt, pi_opt_state = get_optimizer(
            self._optimizer_config.policy, self._model[CONST_POLICY], pi_params
        )
        vf_opt, vf_opt_state = get_optimizer(
            self._optimizer_config.vf, self._model[CONST_VF], vf_params
        )

        self._optimizer = {
            CONST_POLICY: pi_opt,
            CONST_VF: vf_opt,
        }

        self._model_dict = {
            CONST_MODEL: {
                CONST_POLICY: pi_params,
                CONST_VF: vf_params,
            },
            CONST_OPT_STATE: {
                CONST_POLICY: pi_opt_state,
                CONST_VF: vf_opt_state,
            },
        }

    def make_joint_step(self):
        """
        Makes the training step for both actor update and critic update.
        """

        pi_update = get_update_function(self._model[CONST_POLICY])
        vf_update = get_update_function(self._model[CONST_VF])

        def _joint_step(
            model_dict: Dict[str, Any],
            obss: chex.Array,
            h_states: chex.Array,
            acts: chex.Array,
            rets: chex.Array,
            *args,
            **kwargs,
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            """
            The training step that computes the A2C loss and performs actor update and critic update.

            :param model_dict: the model state and optimizer state
            :param obss: the training observations
            :param h_states: the training hidden states for memory-based models
            :param acts: the training actions
            :param rets: the Monte-Carlo returns
            :param *args:
            :param **kwargs:
            :type model_dict: Dict[str, Any]
            :type obss: chex.Array
            :type h_states: chex.Array
            :type acts: chex.Array
            :type rets: chex.Array
            :return: the updated actor state and critic state, their corresponding updated
                     optimizer state, and auxiliary information
            :rtype: Tuple[Dict[str, Any], Dict[str, Any]]

            """
            (agg_loss, aux), grads = jax.value_and_grad(self._joint_loss, has_aux=True)(
                model_dict[CONST_MODEL],
                obss,
                h_states,
                acts,
                rets,
            )
            aux[CONST_AGG_LOSS] = agg_loss
            aux[CONST_GRAD_NORM] = {
                CONST_POLICY: l2_norm(grads[CONST_POLICY]),
                CONST_VF: l2_norm(grads[CONST_VF]),
            }

            pi_params, pi_opt_state = pi_update(
                self._optimizer[CONST_POLICY],
                grads[CONST_POLICY],
                model_dict[CONST_OPT_STATE][CONST_POLICY],
                model_dict[CONST_MODEL][CONST_POLICY],
            )

            vf_params, vf_opt_state = vf_update(
                self._optimizer[CONST_VF],
                grads[CONST_VF],
                model_dict[CONST_OPT_STATE][CONST_VF],
                model_dict[CONST_MODEL][CONST_VF],
            )

            return {
                CONST_MODEL: {
                    CONST_POLICY: pi_params,
                    CONST_VF: vf_params,
                },
                CONST_OPT_STATE: {
                    CONST_POLICY: pi_opt_state,
                    CONST_VF: vf_opt_state,
                },
            }, aux

        return _joint_step

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

        carried_steps = self._global_step % self._update_frequency
        num_update_steps = (
            self._num_steps_per_epoch + carried_steps
        ) // self._update_frequency

        aux = {
            CONST_LOG: {},
        }
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
            ) = self._buffer.sample_with_next_obs(
                batch_size=self._batch_size
            )
            obss = self.update_obs_rms_and_normalize(obss, lengths)
            total_sampling_time += timeit.default_timer() - tic

            tic = timeit.default_timer()
            self.model_dict, qf_aux = self.qf_step(
                self._model_dict,
                obss,
                h_states,
                acts,
                rews,
                terminateds,
                next_obss,
                next_h_states,
            )
            assert np.isfinite(qf_aux[CONST_AGG_LOSS]), f"Loss became NaN\nqf_aux: {qf_aux}"
            total_qf_update_time += timeit.default_timer() - tic

            qf_auxes[-1][CONST_AUX][CONST_QF] = qf_aux
            if self._step % self._target_update_frequency == 0:
                self.update_target_model()

            # Update Actor
            if self._global_step % self._actor_update_frequency == 0:
                pi_auxes.append({})
                tic = timeit.default_timer()
                self.model_dict, pi_aux = self.pi_step(
                    self._model_dict,
                    obss,
                    h_states,
                    acts,
                )
                assert np.isfinite(pi_aux[CONST_AGG_LOSS]), f"Loss became NaN\npi_aux: {pi_aux}"
                total_pi_update_time += timeit.default_timer() - tic
                pi_auxes[-1][CONST_AUX][CONST_POLICY] = pi_aux

                # Update temperature
                if self._target_entropy is not None:
                    temp_auxes.append({})
                    tic = timeit.default_timer()
                    self.model_dict, temp_aux = self.temp_step(
                        self._model_dict,
                        obss,
                        h_states,
                    )
                    assert np.isfinite(temp_aux[CONST_AGG_LOSS]), f"Loss became NaN\ntemp_aux: {temp_aux}"
                    total_temp_update_time += timeit.default_timer() - tic
                    temp_auxes[-1][CONST_AUX][CONST_TEMPERATURE] = temp_aux

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
        aux[CONST_LOG][f"time/update_{CONST_POLICY}"] = total_pi_update_time
        aux[CONST_LOG][f"time/update_{CONST_TEMPERATURE}"] = total_temp_update_time

        aux[CONST_LOG][f"interaction/{CONST_AVERAGE_RETURN}"] = self._rollout.latest_average_return()
        aux[CONST_LOG][f"interaction/{CONST_AVERAGE_EPISODE_LENGTH}"] = self._rollout.latest_average_episode_length()

        # TODO: Fix logging for different training
        qf_auxes = jax.tree_util.tree_map(lambda *args: np.mean(args), *qf_auxes)
        aux[CONST_LOG][f"losses/vf"] = qf_auxes[CONST_AUX][CONST_VF].item()        
        aux[CONST_LOG][f"{CONST_GRAD_NORM}/vf"] = qf_auxes[CONST_AUX][CONST_GRAD_NORM][CONST_VF].item()
        aux[CONST_LOG][f"{CONST_PARAM_NORM}/vf"] = l2_norm(
            self.model_dict[CONST_MODEL][CONST_VF]
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
            temp_auxes = jax.tree_util.tree_map(lambda *args: np.mean(args), *temp_auxes)

        self.gather_rms(aux)
        return aux
