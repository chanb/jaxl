from flax.core.scope import FrozenVariableDict
from typing import Any, Callable, Dict, Union, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom

from jaxl.distributions import Normal
from jaxl.models.common import Model, StochasticPolicy


class GaussianPolicy(StochasticPolicy):
    def __init__(
        self,
        policy: Model,
        min_std: float = 1e-7,
    ):
        self._min_std = min_std
        self.deterministic_action = jax.jit(self.make_deterministic_action(policy))
        self.random_action = jax.jit(self.make_random_action(policy))
        self.compute_action = jax.jit(self.make_compute_action(policy))
        self.act_lprob = jax.jit(self.make_act_lprob(policy))
        self.lprob = jax.jit(self.make_lprob(policy))

    def make_compute_action(
        self, policy: Model
    ) -> Callable[
        [
            Union[FrozenVariableDict, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        def compute_action(
            params: Union[FrozenVariableDict, Dict[str, Any]],
            obs: chex.Array,
            h_state: chex.Array,
            key: jrandom.PRNGKey,
        ) -> chex.Array:
            act_params, h_state = policy.forward(params, obs, h_state)
            act_mean, act_raw_std = jnp.split(act_params, 2, axis=-1)
            act_std = jax.nn.softplus(act_raw_std) + self._min_std
            act = Normal.sample(act_mean, act_std, key)
            return act[0], h_state[0]

        return compute_action

    def make_deterministic_action(
        self, policy: Model
    ) -> Callable[
        [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array],
        Tuple[chex.Array, chex.Array],
    ]:
        def deterministic_action(
            params: Union[FrozenVariableDict, Dict[str, Any]],
            obs: chex.Array,
            h_state: chex.Array,
        ) -> chex.Array:
            act_params, h_state = policy.forward(params, obs, h_state)
            act_mean, _ = jnp.split(act_params, 2, axis=-1)
            return act_mean, h_state

        return deterministic_action

    def make_random_action(
        self, policy: Model
    ) -> Callable[
        [
            Union[FrozenVariableDict, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        def random_action(
            params: Union[FrozenVariableDict, Dict[str, Any]],
            obs: chex.Array,
            h_state: chex.Array,
            key: jrandom.PRNGKey,
        ) -> chex.Array:
            act_params, h_state = policy.forward(params, obs, h_state)
            act_mean, act_raw_std = jnp.split(act_params, 2, axis=-1)
            act_std = jax.nn.softplus(act_raw_std) + self._min_std
            act = Normal.sample(act_mean, act_std, key)
            return act, h_state

        return random_action

    def make_act_lprob(
        self, policy: Model
    ) -> Callable[
        [
            Union[FrozenVariableDict, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array, chex.Array],
    ]:
        def act_lprob(
            params: Union[FrozenVariableDict, Dict[str, Any]],
            obs: chex.Array,
            h_state: chex.Array,
            key: jrandom.PRNGKey,
        ) -> Tuple[chex.Array, chex.Array]:
            act_params, h_state = policy.forward(params, obs, h_state)
            act_mean, act_raw_std = jnp.split(act_params, 2, axis=-1)
            act_std = jax.nn.softplus(act_raw_std) + self._min_std
            act = Normal.sample(act_mean, act_std, key)
            lprob = Normal.lprob(act_mean, act_std, act).sum(-1, keepdims=True)
            return act, lprob, h_state

        return act_lprob

    def make_lprob(
        self, policy: Model
    ) -> Callable[
        [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
        chex.Array,
    ]:
        def lprob(
            params: Union[FrozenVariableDict, Dict[str, Any]],
            obs: chex.Array,
            h_state: chex.Array,
            act: chex.Array,
        ) -> chex.Array:
            act_params, _ = policy.forward(params, obs, h_state)
            act_mean, act_raw_std = jnp.split(act_params, 2, axis=-1)
            act_std = jax.nn.softplus(act_raw_std) + self._min_std
            lprob = Normal.lprob(act_mean, act_std, act).sum(-1, keepdims=True)
            return lprob

        return lprob
