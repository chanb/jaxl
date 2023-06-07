from types import SimpleNamespace
from typing import Any, Callable, Dict, Union, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxl.constants import *
from jaxl.models.common import Model
from jaxl.models.policies import StochasticPolicy
from jaxl.utils import get_reduction


def monte_carlo_returns(
    rews: chex.Array, dones: chex.Array, gamma: float
) -> chex.Array:
    """
    Computes Monte-Carlo return.

    :param rews: rewards
    :param dones: terminated
    :param gamma: discount factor
    :type rews: chex.Array
    :type dones: chex.Array
    :type gamma: float
    :return: Monte-Carlo return
    :rtype: chex.Array

    """
    rets = np.zeros((rews.shape[0] + 1, *rews.shape[1:]))
    for step in reversed(range(len(rews))):
        rets[step] = rets[step + 1] * gamma * (1 - dones[step]) + rews[step]
    return rets[:-1]


def scan_monte_carlo_returns(rews: chex.Array, dones: chex.Array, gamma: float):
    """
    Computes Monte-Carlo return using `jax.lax.scan`.

    :param rews: rewards
    :param dones: terminated
    :param gamma: discount factor
    :type rews: chex.Array
    :type dones: chex.Array
    :type gamma: float
    :return: Monte-Carlo return
    :rtype: chex.Array

    """

    def _returns(
        next_val: chex.Array, transition: Tuple[chex.Array, chex.Array]
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Compute per-step return

        :param next_val: next value estimate
        :param transition: current transition consisting of reward and termination
        :type next_val: chex.Array
        :type transition: Tuple[chex.Array, chex.Array]
        :return: the current return
        :rtype: Tuple[chex.Array, chex.Array]

        """
        rew, done = transition
        val = next_val * gamma * (1 - done) + rew
        return val, val

    return jax.lax.scan(
        _returns, 0, np.concatenate((rews, dones), axis=-1), len(rews), reverse=True
    )[1]


def gae_lambda_returns(
    rews: chex.Array,
    vals: chex.Array,
    dones: chex.Array,
    gamma: float,
    gae_lambda: float,
) -> chex.Array:
    """
    Computes Generalized Advantage Estimation (GAE).

    :param rews: rewards
    :param vals: predicted values
    :param dones: terminated
    :param gamma: discount factor
    :param gae_lambda: GAE lambda
    :type rews: chex.Array
    :type vals: chex.Array
    :type dones: chex.Array
    :type gamma: float
    :type gae_lambda: float
    :return: GAE
    :rtype: chex.Array

    """
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
    """
    Computes Generalized Advantage Estimation (GAE) using `jax.lax.scan`.

    :param rews: rewards
    :param vals: predicted values
    :param dones: terminated
    :param gamma: discount factor
    :param gae_lambda: GAE lambda
    :type rews: chex.Array
    :type vals: chex.Array
    :type dones: chex.Array
    :type gamma: float
    :type gae_lambda: float
    :return: GAE
    :rtype: chex.Array

    """

    def _returns(
        next_res: Tuple[chex.Array, chex.Array],
        transition: Tuple[chex.Array, chex.Array, chex.Array],
    ) -> Tuple[Tuple[chex.Array, chex.Array], chex.Array]:
        """
        Compute per-step return

        :param next_res: GAE and value from the next timestep
        :param transition: current transition consisting of reward, value, and termination
        :type next_res: Tuple[chex.Array, chex.Array]
        :type transition: Tuple[chex.Array, chex.Array, chex.Array]
        :return: the current GAE and return, and current GAE estimate
        :rtype: Tuple[Tuple[chex.Array, chex.Array], chex.Array]

        """
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
        reverse=True,
    )[1]


def make_reinforce_loss(
    policy: StochasticPolicy,
    loss_setting: SimpleNamespace,
) -> Callable[
    [
        Union[optax.Params, Dict[str, Any]],
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
    ],
    Tuple[chex.Array, Dict],
]:
    """
    Gets REINFORCE loss function.

    :param policy: the policy
    :param loss_setting: the loss configuration
    :type policy: StochasticPolicy
    :type loss_setting: SimpleNamespace

    """
    reduction = get_reduction(loss_setting.reduction)

    def reinforce_loss(
        params: Union[optax.Params, Dict[str, Any]],
        obss: chex.Array,
        h_states: chex.Array,
        acts: chex.Array,
        rets: chex.Array,
        baselines: chex.Array,
    ) -> Tuple[chex.Array, Dict]:
        """
        REINFORCE with baseline.

        :param params: the model parameters
        :param obss: the observations
        :param h_states: the hidden states
        :param acts: the actions taken
        :param rets: the returns
        :param baselines: the baselines
        :type params: Union[optax.Params, Dict[str, Any]]
        :type obss: chex.Array
        :type h_states: chex.Array
        :type acts: chex.Array
        :type rets: chex.Array
        :type baselines: chex.Array
        :return: the loss and auxiliary information
        :rtype: Tuple[chex.Array, Dict]

        """
        lprobs = policy.lprob(params, obss, h_states, acts)

        # TODO: Logging of action lprobs
        return reduction(-lprobs * (rets - baselines)), {}

    return reinforce_loss


def make_ppo_pi_loss(
    policy: StochasticPolicy,
    loss_setting: SimpleNamespace,
) -> Callable[
    [
        Union[optax.Params, Dict[str, Any]],
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
    ],
    Tuple[chex.Array, Dict],
]:
    """
    Gets PPO policy loss function.

    :param policy: the policy
    :param loss_setting: the loss configuration
    :type policy: StochasticPolicy
    :type loss_setting: SimpleNamespace

    """
    reduction = get_reduction(loss_setting.reduction)

    def pi_loss(
        params: Union[optax.Params, Dict[str, Any]],
        obss: chex.Array,
        h_states: chex.Array,
        acts: chex.Array,
        advs: chex.Array,
        old_lprobs: chex.Array,
    ) -> Tuple[chex.Array, Dict]:
        """
        PPO policy loss.

        :param params: the model parameters
        :param obss: the observations
        :param h_states: the hidden states
        :param acts: the actions taken
        :param advs: the advantages
        :param old_lprobs: the action log probabilities for importance sampling
        :type params: Union[optax.Params, Dict[str, Any]]
        :type obss: chex.Array
        :type h_states: chex.Array
        :type acts: chex.Array
        :type advs: chex.Array
        :type old_lprobs: chex.Array
        :return: the loss and auxiliary information
        :rtype: Tuple[chex.Array, Dict]

        """
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

        return reduction(-pi_surrogate), {
            CONST_NUM_CLIPPED: (pi_surrogate == surrogate_2).sum(),
            CONST_IS_RATIO: is_ratio,
            CONST_LOG_PROB: lprobs,
        }

    return pi_loss


def make_ppo_vf_loss(
    model: Model,
    loss_setting: SimpleNamespace,
) -> Callable[
    [
        Union[optax.Params, Dict[str, Any]],
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
    ],
    Tuple[chex.Array, Dict],
]:
    """
    Gets PPO value loss function.

    :param model: the value function
    :param loss_setting: the loss configuration
    :type model: Model
    :type loss_setting: SimpleNamespace

    """
    reduction = get_reduction(loss_setting.reduction)

    def vf_loss(
        params: Union[optax.Params, Dict[str, Any]],
        obss: chex.Array,
        h_states: chex.Array,
        rets: chex.Array,
        vals: chex.Array,
    ) -> Tuple[chex.Array, Dict]:
        """
        PPO value loss.

        :param params: the model parameters
        :param obss: the observations
        :param h_states: the hidden states
        :param rets: the estimated returns
        :param vals: the old value estimate to clip from
        :type params: Union[optax.Params, Dict[str, Any]]
        :type obss: chex.Array
        :type h_states: chex.Array
        :type rets: chex.Array
        :type vals: chex.Array
        :return: the loss and auxiliary information
        :rtype: Tuple[chex.Array, Dict]

        """
        preds, _ = model.forward(params, obss, h_states)

        clipped_preds = vals + jnp.clip(
            preds - vals, a_min=-loss_setting.clip_param, a_max=loss_setting.clip_param
        )
        surrogate_1 = (rets - preds) ** 2
        surrogate_2 = (rets - clipped_preds) ** 2
        vf_surrogate = jnp.maximum(surrogate_1, surrogate_2)

        return reduction(vf_surrogate), {
            CONST_NUM_CLIPPED: (vf_surrogate == surrogate_2).sum()
        }

    return vf_loss
