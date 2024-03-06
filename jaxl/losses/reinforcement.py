from types import SimpleNamespace
from typing import Any, Callable, Dict, Union, Tuple, Sequence

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

from jaxl.constants import *
from jaxl.models.common import Model
from jaxl.models import (
    StochasticPolicy,
)
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
        lprobs, aux = policy.lprob(params, obss, h_states, acts)

        return reduction(-lprobs * (rets - baselines)), {
            CONST_LOG_PROBS: lprobs,
            CONST_AUX: aux,
        }

    return reinforce_loss


def make_pi_is_loss(
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
    Gets policy with importance sampling loss function.

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
        Policy with importance sampling loss.

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
        lprobs, aux = policy.lprob(params, obss, h_states, acts)
        is_ratio = jnp.exp(lprobs - old_lprobs)
        # XXX: Deal with inf values
        is_ratio = jax.lax.select(
            jnp.isfinite(is_ratio), is_ratio, jnp.zeros_like(is_ratio)
        )
        pi_surrogate = is_ratio * advs

        return reduction(-pi_surrogate), {
            CONST_NUM_CLIPPED: 0,
            CONST_IS_RATIO: is_ratio,
            CONST_LOG_PROBS: lprobs,
            CONST_AUX: aux,
        }

    return pi_loss


def make_ppo_clip_loss(
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
        lprobs, aux = policy.lprob(params, obss, h_states, acts)
        is_ratio = jnp.exp(lprobs - old_lprobs)
        # XXX: Deal with inf values
        is_ratio = jax.lax.select(
            jnp.isfinite(is_ratio), is_ratio, jnp.zeros_like(is_ratio)
        )

        clipped_is_ratio = jnp.clip(
            is_ratio,
            a_min=1 - loss_setting.clip_param,
            a_max=1 + loss_setting.clip_param,
        )

        surrogate_1 = is_ratio * advs
        surrogate_2 = clipped_is_ratio * advs
        pi_surrogate = jnp.minimum(surrogate_1, surrogate_2)

        return reduction(-pi_surrogate), {
            CONST_NUM_CLIPPED: (clipped_is_ratio != is_ratio).sum(),
            CONST_IS_RATIO: is_ratio,
            CONST_LOG_PROBS: lprobs,
            CONST_AUX: aux,
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
        preds, _, updates = model.forward(params, obss, h_states)
        # XXX: Deal with inf values
        preds = jnp.nan_to_num(preds, posinf=0.0, neginf=0.0)

        clipped_preds = vals + jnp.clip(
            preds - vals, a_min=-loss_setting.clip_param, a_max=loss_setting.clip_param
        )

        surrogate_1 = (rets - preds) ** 2
        surrogate_2 = (rets - clipped_preds) ** 2
        vf_surrogate = jnp.maximum(surrogate_1, surrogate_2)

        return reduction(vf_surrogate), {
            CONST_NUM_CLIPPED: (preds != clipped_preds).sum(),
            CONST_UPDATES: updates,
        }

    return vf_loss


def make_sac_qf_loss(
    models: Dict[str, Any],
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
    Gets SAC critic loss function.

    :param models: the models involved for SAC critic update
    :param loss_setting: the loss configuration
    :type model: Model
    :type loss_setting: SimpleNamespace

    """
    reduction = get_reduction(loss_setting.reduction)

    # XXX: It's designed this way so that we don't keep track of gradient of other models.
    def qf_loss(
        qf_params: Union[optax.Params, Dict[str, Any]],
        target_qf_params: Union[optax.Params, Dict[str, Any]],
        pi_params: Union[optax.Params, Dict[str, Any]],
        temp_params: Union[optax.Params, Dict[str, Any]],
        obss: chex.Array,
        h_states: chex.Array,
        acts: chex.Array,
        rews: chex.Array,
        terminateds: chex.Array,
        next_obss: chex.Array,
        next_h_states: chex.Array,
        gamma: chex.Array,
        keys: Sequence[jrandom.PRNGKey],
    ) -> Tuple[chex.Array, Dict]:
        """
        SAC critic loss.

        :param qf_params: the Q parameters
        :param target_qf_params: the target Q parameters
        :param pi_params: the policy parameters
        :param temp_params: the temperature
        :param obss: current observations
        :param h_states: current hidden states
        :param acts: actions taken for current observations
        :param rews: received rewards
        :param terminateds: whether or not the episode is terminated
        :param next_obss: next observations
        :param next_h_states: next hidden states
        :param gamma: discount factor
        :param keys: random keys for sampling next actions
        :type qf_params: Union[optax.Params, Dict[str, Any]]
        :type target_qf_params: Union[optax.Params, Dict[str, Any]]
        :type pi_params: Union[optax.Params, Dict[str, Any]]
        :type temp_params: Union[optax.Params, Dict[str, Any]]
        :type obss: chex.Array
        :type h_states: chex.Array
        :type acts: chex.Array
        :type rews: chex.Array
        :type terminateds: chex.Array
        :type next_obss: chex.Array
        :type next_h_states: chex.Array
        :type gamma: chex.Array
        :type keys: Sequence[jrandom.PRNGKey]
        :return: the loss and auxiliary information
        :rtype: Tuple[chex.Array, Dict]

        """
        # Action for next timestep
        next_acts, next_lprobs, _, _ = models[CONST_POLICY].act_lprob(
            pi_params, next_obss, next_h_states, keys
        )
        next_lprobs = jnp.sum(next_lprobs, axis=-1, keepdims=True)

        # Q-value for current timestep
        curr_q_preds, _, updates = models[CONST_QF].q_values(
            qf_params, obss, h_states, acts
        )
        curr_q_preds_min = jnp.min(curr_q_preds, axis=0)

        # Q-value for next timestep
        next_q_preds, _, _ = models[CONST_TARGET_QF].q_values(
            target_qf_params, next_obss, next_h_states, next_acts
        )
        next_q_preds_min = jnp.min(next_q_preds, axis=0)

        # Compute temperature
        temp = models[CONST_TEMPERATURE].apply(temp_params)

        # Compute min. clipped TD error
        next_vs = next_q_preds_min - temp * next_lprobs
        curr_q_targets = rews + gamma * (1 - terminateds) * next_vs
        td_errors = (curr_q_preds - curr_q_targets[None]) ** 2
        loss = reduction(td_errors)

        return loss, {
            "mean_var_q": jnp.mean(jnp.var(curr_q_preds, axis=0)),
            "min_var_q": jnp.min(jnp.var(curr_q_preds, axis=0)),
            "max_var_q": jnp.max(jnp.var(curr_q_preds, axis=0)),
            "max_next_q": jnp.max(next_q_preds_min),
            "min_next_q": jnp.min(next_q_preds_min),
            "mean_next_q": jnp.mean(next_q_preds_min),
            "max_curr_q": jnp.max(curr_q_preds_min),
            "min_curr_q": jnp.min(curr_q_preds_min),
            "mean_curr_q": jnp.mean(curr_q_preds_min),
            "max_td_error": jnp.max(td_errors),
            "min_td_error": jnp.min(td_errors),
            "max_q_log_prob": jnp.max(next_lprobs),
            "min_q_log_prob": jnp.min(next_lprobs),
            "mean_q_log_prob": jnp.mean(next_lprobs),
            "curr_q_targets": curr_q_targets,
            CONST_UPDATES: updates,
        }

    return qf_loss


def make_cross_q_sac_qf_loss(
    models: Dict[str, Any],
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
    Gets CrossQ SAC critic loss function.

    :param models: the models involved for CrossQ SAC critic update
    :param loss_setting: the loss configuration
    :type model: Model
    :type loss_setting: SimpleNamespace

    """
    reduction = get_reduction(loss_setting.reduction)

    # XXX: It's designed this way so that we don't keep track of gradient of other models.
    def qf_loss(
        qf_params: Union[optax.Params, Dict[str, Any]],
        pi_params: Union[optax.Params, Dict[str, Any]],
        temp_params: Union[optax.Params, Dict[str, Any]],
        obss: chex.Array,
        h_states: chex.Array,
        acts: chex.Array,
        rews: chex.Array,
        terminateds: chex.Array,
        next_obss: chex.Array,
        next_h_states: chex.Array,
        gamma: chex.Array,
        keys: Sequence[jrandom.PRNGKey],
    ) -> Tuple[chex.Array, Dict]:
        """
        CrossQ SAC critic loss.

        :param qf_params: the Q parameters
        :param pi_params: the policy parameters
        :param temp_params: the temperature
        :param obss: current observations
        :param h_states: current hidden states
        :param acts: actions taken for current observations
        :param rews: received rewards
        :param terminateds: whether or not the episode is terminated
        :param next_obss: next observations
        :param next_h_states: next hidden states
        :param gamma: discount factor
        :param keys: random keys for sampling next actions
        :type qf_params: Union[optax.Params, Dict[str, Any]]
        :type pi_params: Union[optax.Params, Dict[str, Any]]
        :type temp_params: Union[optax.Params, Dict[str, Any]]
        :type obss: chex.Array
        :type h_states: chex.Array
        :type acts: chex.Array
        :type rews: chex.Array
        :type terminateds: chex.Array
        :type next_obss: chex.Array
        :type next_h_states: chex.Array
        :type gamma: chex.Array
        :type keys: Sequence[jrandom.PRNGKey]
        :return: the loss and auxiliary information
        :rtype: Tuple[chex.Array, Dict]

        """
        # Action for next timestep
        next_acts, next_lprobs, _, _ = models[CONST_POLICY].act_lprob(
            pi_params, next_obss, next_h_states, keys
        )
        next_lprobs = jnp.sum(next_lprobs, axis=-1, keepdims=True)

        all_q_preds, _, updates = models[CONST_QF].q_values(
            qf_params,
            jnp.concatenate(
                (obss, next_obss),
                axis=0,
            ),
            jnp.concatenate(
                (h_states, next_h_states),
                axis=0,
            ),
            jnp.concatenate(
                (acts, next_acts),
                axis=0,
            ),
            eval=False,
        )
        all_q_preds_min = jax.lax.stop_gradient(jnp.min(all_q_preds, axis=0))
        curr_q_preds_min, next_q_preds_min = jnp.split(all_q_preds_min, 2)
        curr_q_preds, _ = jnp.split(all_q_preds, 2, axis=1)

        # Compute temperature
        temp = models[CONST_TEMPERATURE].apply(temp_params)

        # Compute min. clipped TD error
        next_vs = next_q_preds_min - temp * next_lprobs
        curr_q_targets = rews + gamma * (1 - terminateds) * next_vs
        td_errors = (curr_q_preds - curr_q_targets[None]) ** 2
        loss = reduction(td_errors)

        return loss, {
            "mean_var_q": jnp.mean(jnp.var(curr_q_preds, axis=0)),
            "min_var_q": jnp.min(jnp.var(curr_q_preds, axis=0)),
            "max_var_q": jnp.max(jnp.var(curr_q_preds, axis=0)),
            "max_next_q": jnp.max(next_q_preds_min),
            "min_next_q": jnp.min(next_q_preds_min),
            "mean_next_q": jnp.mean(next_q_preds_min),
            "max_curr_q": jnp.max(curr_q_preds_min),
            "min_curr_q": jnp.min(curr_q_preds_min),
            "mean_curr_q": jnp.mean(curr_q_preds_min),
            "max_td_error": jnp.max(td_errors),
            "min_td_error": jnp.min(td_errors),
            "max_q_log_prob": jnp.max(next_lprobs),
            "min_q_log_prob": jnp.min(next_lprobs),
            "mean_q_log_prob": jnp.mean(next_lprobs),
            "curr_q_targets": curr_q_targets,
            CONST_UPDATES: updates,
        }

    return qf_loss


def make_sac_pi_loss(
    models: Dict[str, Any],
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
    Gets SAC actor loss function.

    :param models: the models involved for SAC actor update
    :param loss_setting: the loss configuration
    :type model: Model
    :type loss_setting: SimpleNamespace

    """
    reduction = get_reduction(loss_setting.reduction)

    # XXX: It's designed this way so that we don't keep track of gradient of other models.
    def pi_loss(
        pi_params: Union[optax.Params, Dict[str, Any]],
        qf_params: Union[optax.Params, Dict[str, Any]],
        temp_params: Union[optax.Params, Dict[str, Any]],
        obss: chex.Array,
        h_states: chex.Array,
        keys: Sequence[jrandom.PRNGKey],
    ) -> Tuple[chex.Array, Dict]:
        """
        SAC actor loss.

        :param pi_params: the policy parameters
        :param qf_params: the Q parameters
        :param temp_params: the temperature
        :param obss: current observations
        :param h_states: current hidden states
        :param keys: random keys for sampling next actions
        :type pi_params: Union[optax.Params, Dict[str, Any]]
        :type qf_params: Union[optax.Params, Dict[str, Any]]
        :type temp_params: Union[optax.Params, Dict[str, Any]]
        :type obss: chex.Array
        :type h_states: chex.Array
        :type keys: Sequence[jrandom.PRNGKey]
        :return: the loss and auxiliary information
        :rtype: Tuple[chex.Array, Dict]

        """

        acts, lprobs, _, updates = models[CONST_POLICY].act_lprob(
            pi_params, obss, h_states, keys
        )
        lprobs = jnp.sum(lprobs, axis=-1, keepdims=True)
        curr_q_preds, _, _ = models[CONST_QF].q_values(
            qf_params,
            obss,
            h_states,
            acts,
            eval=True,
        )
        curr_q_preds_min = jnp.min(curr_q_preds, axis=0)

        temp = models[CONST_TEMPERATURE].apply(temp_params)

        vals = -(curr_q_preds_min - temp * lprobs)
        loss = reduction(vals)

        return loss, {
            "max_policy_log_prob": jnp.max(lprobs),
            "min_policy_log_prob": jnp.min(lprobs),
            "mean_policy_log_prob": jnp.mean(lprobs),
            "max_estimated_value": jnp.max(vals),
            "min_estimated_value": jnp.min(vals),
            "mean_estimated_value": jnp.mean(vals),
            CONST_UPDATES: updates,
        }

    return pi_loss


def make_sac_temp_loss(
    models: Dict[str, Any],
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
    Gets SAC temperature loss function.

    :param models: the models involved for SAC temperature update
    :param loss_setting: the loss configuration
    :type model: Model
    :type loss_setting: SimpleNamespace

    """
    reduction = get_reduction(loss_setting.reduction)

    # XXX: It's designed this way so that we don't keep track of gradient of other models.
    def temp_loss(
        temp_params: Union[optax.Params, Dict[str, Any]],
        pi_params: Union[optax.Params, Dict[str, Any]],
        obss: chex.Array,
        h_states: chex.Array,
        target_entropy: float,
        keys: Sequence[jrandom.PRNGKey],
    ) -> Tuple[chex.Array, Dict]:
        """
        SAC temperature loss.

        :param temp_params: the temperature
        :param pi_params: the policy parameters
        :param obss: current observations
        :param h_states: current hidden states
        :param target_entropy: the target entropy
        :param keys: random keys for sampling next actions
        :type temp_params: Union[optax.Params, Dict[str, Any]]
        :type pi_params: Union[optax.Params, Dict[str, Any]]
        :type obss: chex.Array
        :type h_states: chex.Array
        :type target_entropy: float
        :type keys: Sequence[jrandom.PRNGKey]
        :return: the loss and auxiliary information
        :rtype: Tuple[chex.Array, Dict]

        """

        _, lprobs, _, _ = models[CONST_POLICY].act_lprob(
            pi_params, obss, h_states, keys, eval=True
        )
        lprobs = jnp.sum(lprobs, axis=-1, keepdims=True)

        temp = models[CONST_TEMPERATURE].apply(temp_params)

        temp_penalty = temp * -(lprobs + target_entropy)
        loss = reduction(temp_penalty)

        return loss, {
            "max_policy_log_prob": jnp.max(lprobs),
            "min_policy_log_prob": jnp.min(lprobs),
            "mean_policy_log_prob": jnp.mean(lprobs),
            "max_temp_penalty": jnp.max(temp_penalty),
            "min_temp_penalty": jnp.min(temp_penalty),
            "mean_temp_penalty": jnp.mean(temp_penalty),
            "temperature": temp,
        }

    return temp_loss
