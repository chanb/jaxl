from types import SimpleNamespace
from typing import Any, Dict, Tuple

import chex
import dill
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import os
import timeit

from jaxl.constants import *
from jaxl.learners.reinforce import REINFORCE
from jaxl.learners.reinforcement import OnPolicyLearner
from jaxl.learners.ppo import PPO
from jaxl.losses.reinforcement import monte_carlo_returns
from jaxl.losses.supervised import make_squared_loss
from jaxl.models import (
    get_model,
    get_wsrl_model,
    get_policy,
    get_update_function,
    load_params,
    policy_output_dim,
)
from jaxl.optimizers import (
    get_scheduler,
    get_optimizer,
)
from jaxl.utils import l2_norm


"""
Standard Warm-start RL.
"""


class WSRLPPO(PPO):
    """
    WSRL with Proximal Policy Optimization (PPO) algorithm.
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
        self._model = {
            CONST_POLICY: get_wsrl_model(input_dim, act_dim, self._model_config.policy),
            CONST_VF: get_model(input_dim, (1,), self._model_config.vf),
        }

        model_keys = jrandom.split(jrandom.PRNGKey(self._config.seeds.model_seed))
        dummy_x = self._generate_dummy_x(input_dim)
        pi_params = self._model[CONST_POLICY].init(model_keys[0], dummy_x)
        vf_params = self._model[CONST_VF].init(model_keys[1], dummy_x)

        if getattr(self._config, "load_pretrain", False):
            all_params = load_params(self._config.load_pretrain.checkpoint_path)
            if CONST_POLICY in self._config.load_pretrain.load_components:
                pi_params[CONST_MEAN] = all_params[CONST_MODEL_DICT][CONST_MODEL][
                    CONST_POLICY
                ]
            if CONST_VF in self._config.load_pretrain.load_components:
                vf_params = all_params[CONST_MODEL_DICT][CONST_MODEL][CONST_VF]
            if getattr(self._config, CONST_OBS_RMS, False):
                self._obs_rms_state = all_params[CONST_OBS_RMS]

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


class WSRLREINFORCE(REINFORCE):
    """
    WSRL with REINFORCE algorithm.
    """

    def _initialize_model_and_opt(self, input_dim: chex.Array, output_dim: chex.Array):
        """
        Construct the policy and the optimizer.

        :param input_dim: input dimension of the data point
        :param output_dim: output dimension of the data point
        :type input_dim: chex.Array
        :type output_dim: chex.Array

        """
        output_dim = policy_output_dim(output_dim, self._config)
        self._model = get_wsrl_model(input_dim, output_dim, self._model_config.policy)

        model_key = jrandom.PRNGKey(self._config.seeds.model_seed)
        dummy_x = self._generate_dummy_x(input_dim)
        params = self._model.init(model_key, dummy_x)
        if getattr(self._config, "load_pretrain", False):
            all_params = load_params(self._config.load_pretrain.checkpoint_path)
            if CONST_POLICY in self._config.load_pretrain.load_components:
                params[CONST_MEAN] = all_params[CONST_MODEL_DICT][CONST_MODEL][
                    CONST_POLICY
                ]

        self._optimizer, opt_state = get_optimizer(
            self._optimizer_config.policy, self._model, params
        )

        self._model_dict = {
            CONST_MODEL: {CONST_POLICY: params},
            CONST_OPT_STATE: {CONST_POLICY: opt_state},
        }


class WSRLPolicyEvaluation(OnPolicyLearner):
    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        super().__init__(config, model_config, optimizer_config)

        self._pi = get_policy(self._model[CONST_POLICY], config)
        self._vf = self._model[CONST_VF]
        self._vf_loss = make_squared_loss(self._vf, self._config.vf_loss_setting)

        def critic_loss(
            model_dicts: Dict[str, Any],
            obss: chex.Array,
            h_states: chex.Array,
            rets: chex.Array,
            *args,
            **kwargs,
        ) -> Tuple[chex.Array, Dict[str, Any]]:
            vf_loss, vf_aux = self._vf_loss(
                model_dicts[CONST_VF],
                obss,
                h_states,
                rets,
            )

            baselines = jax.lax.stop_gradient(vf_aux[CONST_PREDICTIONS])
            aux = {
                CONST_VF: vf_loss,
                CONST_ADVANTAGE: (rets - baselines).mean(),
                CONST_UPDATES: {
                    CONST_VF: vf_aux[CONST_UPDATES],
                },
            }
            return vf_loss, aux

        self._critic_loss = critic_loss
        self.critic_step = jax.jit(self.make_critic_step())

    @property
    def policy(self):
        return self._pi

    @property
    def policy_params(self):
        return self._model_dict[CONST_MODEL][CONST_POLICY]

    def _initialize_model_and_opt(self, input_dim: chex.Array, output_dim: chex.Array):
        # XXX: The policy should be deterministic policy
        self._model = {
            CONST_POLICY: get_model(input_dim, output_dim, self._model_config.policy),
            CONST_VF: get_model(input_dim, (1,), self._model_config.vf),
        }

        pretrained_params = load_params(self._config.load_pretrain.checkpoint_path)
        pi_params = pretrained_params[CONST_MODEL_DICT][CONST_MODEL][CONST_POLICY]

        model_keys = jrandom.split(jrandom.PRNGKey(self._config.seeds.model_seed))
        dummy_x = self._generate_dummy_x(input_dim)
        vf_params = self._model[CONST_VF].init(model_keys[1], dummy_x)

        vf_opt, vf_opt_state = get_optimizer(
            self._optimizer_config.vf, self._model[CONST_VF], vf_params
        )

        self._optimizer = {
            CONST_VF: vf_opt,
        }

        self._model_dict = {
            CONST_MODEL: {
                CONST_POLICY: pi_params,
                CONST_VF: vf_params,
            },
            CONST_OPT_STATE: {
                CONST_VF: vf_opt_state,
            },
        }

    def make_critic_step(self):
        vf_update = get_update_function(self._model[CONST_VF])

        def _critic_step(
            model_dict: Dict[str, Any],
            obss: chex.Array,
            h_states: chex.Array,
            rets: chex.Array,
            *args,
            **kwargs,
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            (agg_loss, aux), grads = jax.value_and_grad(
                self._critic_loss, has_aux=True
            )(
                model_dict[CONST_MODEL],
                obss,
                h_states,
                rets,
            )
            aux[CONST_AGG_LOSS] = agg_loss
            aux[CONST_GRAD_NORM] = {
                CONST_VF: l2_norm(grads[CONST_VF]),
            }

            vf_params, vf_opt_state = vf_update(
                self._optimizer[CONST_VF],
                grads[CONST_VF],
                model_dict[CONST_OPT_STATE][CONST_VF],
                model_dict[CONST_MODEL][CONST_VF],
                aux[CONST_UPDATES][CONST_VF],
            )

            return {
                CONST_MODEL: {
                    CONST_POLICY: model_dict[CONST_MODEL][CONST_POLICY],
                    CONST_VF: vf_params,
                },
                CONST_OPT_STATE: {
                    CONST_VF: vf_opt_state,
                },
            }, aux

        return _critic_step

    def update(self, *args, **kwargs) -> Dict[str, Any]:
        auxes = []
        total_rollout_time = 0
        total_update_time = 0

        carried_steps = self._global_step % self._update_frequency
        num_update_steps = (
            self._num_steps_per_epoch + carried_steps
        ) // self._update_frequency

        for update_i in range(num_update_steps):
            auxes.append({})
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

            self.model_dict, aux = self.critic_step(
                self._model_dict, obss, h_states, rets
            )
            total_update_time += timeit.default_timer() - tic
            assert np.isfinite(aux[CONST_AGG_LOSS]), f"Loss became NaN\naux: {aux}"

            auxes[-1][CONST_AUX] = aux
            auxes[-1][CONST_ACTION] = {
                i: {
                    CONST_SATURATION: np.abs(acts[..., i]).max(),
                    CONST_MEAN: np.abs(acts[..., i]).mean(),
                }
                for i in range(acts.shape[-1])
            }

        auxes = jax.tree_util.tree_map(lambda *args: np.mean(args), *auxes)
        aux[CONST_LOG] = {
            f"losses/{CONST_AGG_LOSS}": auxes[CONST_AUX][CONST_AGG_LOSS].item(),
            f"losses/vf": auxes[CONST_AUX][CONST_VF].item(),
            f"losses_info/{CONST_ADVANTAGE}": auxes[CONST_AUX][CONST_ADVANTAGE].item(),
            f"{CONST_GRAD_NORM}/vf": auxes[CONST_AUX][CONST_GRAD_NORM][CONST_VF].item(),
            f"{CONST_PARAM_NORM}/vf": l2_norm(
                self.model_dict[CONST_MODEL][CONST_VF]
            ).item(),
            f"interaction/{CONST_AVERAGE_RETURN}": self._rollout.latest_average_return(),
            f"interaction/{CONST_AVERAGE_EPISODE_LENGTH}": self._rollout.latest_average_episode_length(),
            f"time/{CONST_ROLLOUT_TIME}": total_rollout_time,
            f"time/{CONST_UPDATE_TIME}": total_update_time,
        }

        for act_i in range(acts.shape[-1]):
            aux[CONST_LOG][
                f"{CONST_ACTION}/{CONST_ACTION}_{act_i}_{CONST_SATURATION}"
            ] = auxes[CONST_ACTION][act_i][CONST_SATURATION]
            aux[CONST_LOG][f"{CONST_ACTION}/{CONST_ACTION}_{act_i}_{CONST_MEAN}"] = (
                auxes[CONST_ACTION][act_i][CONST_MEAN]
            )

        self.gather_rms(aux)
        return aux
