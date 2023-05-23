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
        ) -> Tuple[chex.Array, Dict]:
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

    def _initialize_model_and_opt(self, input_dim: chex.Array, output_dim: chex.Array):
        act_dim = policy_output_dim(output_dim, self._config)
        self._model = {
            CONST_POLICY: get_model(input_dim, act_dim, self._model_config.policy),
            CONST_VF: get_model(input_dim, output_dim, self._model_config.vf),
        }

        self._optimizer = {
            CONST_POLICY: get_optimizer(self._optimizer_config.policy),
            CONST_VF: get_optimizer(self._optimizer_config.vf),
        }

        model_key = jrandom.PRNGKey(self._config.seeds.model_seed)
        dummy_x = self._generate_dummy_x()
        pi_params = self._model[CONST_POLICY].init(model_key, dummy_x)
        pi_opt_state = self._optimizer[CONST_POLICY].init(pi_params)

        vf_params = self._model[CONST_VF].init(model_key, dummy_x)
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
        def _joint_step(
            model_dict: Dict[str, Any],
            obss: chex.Array,
            h_states: chex.Array,
            acts: chex.Array,
            rets: chex.Array,
            *args,
            **kwargs,
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
        tic = timeit.default_timer()
        self._rollout.rollout(
            self._model_dict[CONST_MODEL][CONST_POLICY],
            self._pi,
            self._obs_rms,
            self._buffer,
            self._update_frequency,
        )
        rollout_time = timeit.default_timer() - tic

        tic = timeit.default_timer()
        obss, h_states, acts, rews, dones, _, _, _, lengths, _ = self._buffer.sample(
            batch_size=self._update_frequency, idxes=self._sample_idxes
        )

        if self.obs_rms:
            idxes = lengths.reshape((-1, *[1 for _ in range(obss.ndim - 1)])) - 1
            update_obss = np.take_along_axis(obss, idxes, 1)
            self.model.obs_rms.update(update_obss)
            obss = self.obs_rms.normalize(obss)

        rets = monte_carlo_returns(rews, dones, self._gamma)
        if self.val_rms:
            self.val_rms.update(rets)
            rets = self.val_rms.normalize(rets)

        self.model_dict, aux = self.joint_step(
            self._model_dict, obss, h_states, acts, rets
        )
        update_time = timeit.default_timer() - tic
        assert np.isfinite(aux[CONST_AGG_LOSS]), f"Loss became NaN\naux: {aux}"

        aux[CONST_LOG] = {
            f"losses/{CONST_AGG_LOSS}": aux[CONST_AGG_LOSS].item(),
            f"losses/pi": aux[CONST_POLICY].item(),
            f"losses/vf": aux[CONST_VF].item(),
            f"losses/advantage_mean": aux[CONST_ADVANTAGE].item(),
            f"{CONST_GRAD_NORM}/pi": aux[CONST_GRAD_NORM][CONST_POLICY].item(),
            f"{CONST_PARAM_NORM}/pi": l2_norm(
                self.model_dict[CONST_MODEL][CONST_POLICY]
            ).item(),
            f"{CONST_GRAD_NORM}/vf": aux[CONST_GRAD_NORM][CONST_VF].item(),
            f"{CONST_PARAM_NORM}/vf": l2_norm(
                self.model_dict[CONST_MODEL][CONST_VF]
            ).item(),
            f"interaction/{CONST_LATEST_RETURN}": self._rollout.latest_return,
            f"interaction/{CONST_LATEST_EPISODE_LENGTH}": self._rollout.latest_episode_length,
            f"time/{CONST_ROLLOUT_TIME}": rollout_time,
            f"time/{CONST_UPDATE_TIME}": update_time,
        }

        self.gather_rms(aux)
        return aux
