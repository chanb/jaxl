from types import SimpleNamespace
from typing import Any, Dict, Tuple

import chex
import jax
import jax.random as jrandom
import numpy as np
import optax
import timeit

from jaxl.constants import *
from jaxl.learners.reinforcement import OnPolicyLearner
from jaxl.losses.reinforcement import monte_carlo_returns, make_reinforce_loss
from jaxl.losses.supervised import make_squared_loss
from jaxl.models import (
    get_model,
    get_optimizer,
    get_policy,
    policy_output_dim,
)
from jaxl.utils import l2_norm


"""
Standard A2C.
"""


class A2C(OnPolicyLearner):
    """
    Advantage Actor Critic (A2C) algorithm. This extends `OnPolicyLearner`.
    """

    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        super().__init__(config, model_config, optimizer_config)

        self._pi = get_policy(self._model[CONST_POLICY], config)
        self._vf = self._model[CONST_VF]

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
            pi_loss, _ = self._pi_loss(
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
            CONST_VF: get_model(input_dim, output_dim, self._model_config.vf),
        }

        self._optimizer = {
            CONST_POLICY: get_optimizer(self._optimizer_config.policy),
            CONST_VF: get_optimizer(self._optimizer_config.vf),
        }

        model_keys = jrandom.split(jrandom.PRNGKey(self._config.seeds.model_seed))
        dummy_x = self._generate_dummy_x(input_dim)
        pi_params = self._model[CONST_POLICY].init(model_keys[0], dummy_x)
        pi_opt_state = self._optimizer[CONST_POLICY].init(pi_params)

        vf_params = self._model[CONST_VF].init(model_keys[1], dummy_x)
        vf_opt_state = self._optimizer[CONST_VF].init(vf_params)
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
                {
                    CONST_POLICY: model_dict[CONST_MODEL][CONST_POLICY],
                    CONST_VF: model_dict[CONST_MODEL][CONST_VF],
                },
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

            updates, pi_opt_state = self._optimizer[CONST_POLICY].update(
                grads[CONST_POLICY],
                model_dict[CONST_OPT_STATE][CONST_POLICY],
                model_dict[CONST_MODEL][CONST_POLICY],
            )
            pi_params = optax.apply_updates(
                model_dict[CONST_MODEL][CONST_POLICY], updates
            )

            updates, vf_opt_state = self._optimizer[CONST_VF].update(
                grads[CONST_VF],
                model_dict[CONST_OPT_STATE][CONST_VF],
                model_dict[CONST_MODEL][CONST_VF],
            )
            vf_params = optax.apply_updates(model_dict[CONST_MODEL][CONST_VF], updates)

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

        auxes = []
        total_rollout_time = 0
        total_update_time = 0
        for _ in range(self._num_update_steps):
            auxes.append({})
            tic = timeit.default_timer()
            self._rollout.rollout(
                self._model_dict[CONST_MODEL][CONST_POLICY],
                self._pi,
                self._obs_rms,
                self._buffer,
                self._update_frequency,
            )
            total_rollout_time += timeit.default_timer() - tic

            tic = timeit.default_timer()
            (
                obss,
                h_states,
                acts,
                rews,
                dones,
                _,
                _,
                _,
                lengths,
                _,
            ) = self._buffer.sample(
                batch_size=self._update_frequency, idxes=self._sample_idxes
            )

            obss = self.update_obs_rms_and_normalize(obss, lengths)

            rets = monte_carlo_returns(rews, dones, self._gamma)
            rets = self.update_value_rms_and_normalize(rets)

            self.model_dict, aux = self.joint_step(
                self._model_dict, obss, h_states, acts, rets
            )
            total_update_time += timeit.default_timer() - tic
            assert np.isfinite(aux[CONST_AGG_LOSS]), f"Loss became NaN\naux: {aux}"

            auxes[-1][CONST_AUX] = aux

        auxes = jax.tree_util.tree_map(lambda *args: np.mean(args), *auxes)
        aux[CONST_LOG] = {
            f"losses/{CONST_AGG_LOSS}": auxes[CONST_AUX][CONST_AGG_LOSS].item(),
            f"losses/pi": auxes[CONST_AUX][CONST_POLICY].item(),
            f"losses/vf": auxes[CONST_AUX][CONST_VF].item(),
            f"losses_info/{CONST_ADVANTAGE}": auxes[CONST_AUX][CONST_ADVANTAGE].item(),
            f"{CONST_GRAD_NORM}/pi": auxes[CONST_AUX][CONST_GRAD_NORM][
                CONST_POLICY
            ].item(),
            f"{CONST_PARAM_NORM}/pi": l2_norm(
                self.model_dict[CONST_MODEL][CONST_POLICY]
            ).item(),
            f"{CONST_GRAD_NORM}/vf": auxes[CONST_AUX][CONST_GRAD_NORM][CONST_VF].item(),
            f"{CONST_PARAM_NORM}/vf": l2_norm(
                self.model_dict[CONST_MODEL][CONST_VF]
            ).item(),
            f"interaction/{CONST_AVERAGE_RETURN}": self._rollout.latest_average_return(),
            f"interaction/{CONST_AVERAGE_EPISODE_LENGTH}": self._rollout.latest_average_episode_length(),
            f"time/{CONST_ROLLOUT_TIME}": total_rollout_time,
            f"time/{CONST_UPDATE_TIME}": total_update_time,
        }

        self.gather_rms(aux)
        return aux
