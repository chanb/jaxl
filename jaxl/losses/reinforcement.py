from flax.core.scope import FrozenVariableDict
from types import SimpleNamespace
from typing import Any, Callable, Dict, Union, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from jaxl.constants import *
from jaxl.models.common import Model
from jaxl.models.policies import StochasticPolicy
from jaxl.utils import get_reduction


def monte_carlo_returns(
    rews: chex.Array, dones: chex.Array, gamma: float
) -> chex.Array:
    rets = np.zeros((rews.shape[0] + 1, *rews.shape[1:]))
    for step in reversed(range(len(rews))):
        rets[step] = rets[step + 1] * gamma * (1 - dones[step]) + rews[step]
    return rets[:-1]

def scan_monte_carlo_returns(
    rews: chex.Array, dones: chex.Array, gamma: float
):
    def _returns(next_val, transition):
        rew, done = transition
        val = next_val * gamma * (1 - done) + rew
        return val, val

    return jax.lax.scan(
        _returns,
        0,
        np.concatenate((rews, dones), axis=-1),
        len(rews),
        reverse=True
    )[1]

def gae_lambda_returns(
    rews: chex.Array,
    vals: chex.Array,
    dones: chex.Array,
    gamma: float,
    gae_lambda: float,
) -> chex.Array:
    rets = np.zeros((rews.shape[0] + 1, *rews.shape[1:]))
    gae = 0
    for step in reversed(range(len(rews))):
        delta = rews[step] + (1 - dones[step]) * gamma * vals[step + 1] - vals[step]
        gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
        rets[step] = gae + vals[step]
    return rets[:-1]

def scan_gae_lambda_returns(
    rews: chex.Array,
    vals: chex.Array,
    dones: chex.Array,
    gamma: float,
    gae_lambda: float,
):
    def _returns(next_res, transition):
        rew, val, done = transition
        next_gae, next_val = next_res
        delta = rew + (1 - done) * gamma * next_val - val
        gae = delta + gamma * gae_lambda * (1 - done) * next_gae
        return (gae, val), gae + val

    return jax.lax.scan(
        _returns,
        (0, vals[-1, 0]),
        np.concatenate((rews, vals[:-1], dones), axis=-1),
        len(rews),
        reverse=True
    )[1]

def make_reinforce_loss(
    policy: StochasticPolicy,
    loss_setting: SimpleNamespace,
) -> Callable[
    [
        Union[FrozenVariableDict, Dict[str, Any]],
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
    ],
    Tuple[chex.Array, Dict],
]:
    reduction = get_reduction(loss_setting.reduction)

    def reinforce_loss(
        params: Union[FrozenVariableDict, Dict[str, Any]],
        obss: chex.Array,
        h_states: chex.Array,
        acts: chex.Array,
        rets: chex.Array,
        baselines: chex.Array,
    ) -> Tuple[chex.Array, Dict]:
        lprobs = policy.lprob(params, obss, h_states, acts)

        # TODO: Logging of action lprobs
        return reduction(-lprobs * (rets - baselines)), {}

    return reinforce_loss


def make_ppo_pi_loss(
    policy: StochasticPolicy,
    loss_setting: SimpleNamespace,
) -> Callable[
    [
        Union[FrozenVariableDict, Dict[str, Any]],
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
    ],
    Tuple[chex.Array, Dict],
]:
    reduction = get_reduction(loss_setting.reduction)

    def pi_loss(
        params: Union[FrozenVariableDict, Dict[str, Any]],
        obss: chex.Array,
        h_states: chex.Array,
        acts: chex.Array,
        advs: chex.Array,
        old_lprobs: chex.Array,
    ) -> Tuple[chex.Array, Dict]:
        lprobs = policy.lprob(params, obss, h_states, acts)
        is_ratio = jnp.exp(lprobs - old_lprobs)

        surrogate_1 = is_ratio * advs
        surrogate_2 = (
            jnp.clip(
                is_ratio,
                a_min=1 - loss_setting.clip_param,
                a_max=1 + loss_setting.clip_param,
            )
            * advs
        )
        pi_surrogate = jnp.minimum(surrogate_1, surrogate_2)

        # TODO: Logging of action lprobs
        return reduction(-pi_surrogate), {
            CONST_NUM_CLIPPED: (surrogate_1 == surrogate_2).sum(),
            CONST_IS_RATIO: is_ratio,
        }

    return pi_loss


def make_ppo_vf_loss(
    model: Model,
    loss_setting: SimpleNamespace,
) -> Callable[
    [
        Union[FrozenVariableDict, Dict[str, Any]],
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
    ],
    Tuple[chex.Array, Dict],
]:
    reduction = get_reduction(loss_setting.reduction)

    def vf_loss(
        params: Union[FrozenVariableDict, Dict[str, Any]],
        obss: chex.Array,
        h_states: chex.Array,
        rets: chex.Array,
        vals: chex.Array,
    ) -> Tuple[chex.Array, Dict]:
        preds, _ = model.forward(params, obss, h_states)

        clipped_preds = vals + jnp.clip(
            preds - vals, a_min=-loss_setting.clip_param, a_max=loss_setting.clip_param
        )
        surrogate_1 = (rets - preds) ** 2
        surrogate_2 = (rets - clipped_preds) ** 2
        vf_surrogate = jnp.maximum(surrogate_1, surrogate_2)

        return reduction(vf_surrogate), {
            CONST_NUM_CLIPPED: (surrogate_1 == surrogate_2).sum()
        }

    return vf_loss
