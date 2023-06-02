from typing import Any, Callable, Dict, Union, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from jaxl.constants import DEFAULT_MIN_STD
from jaxl.distributions import Normal
from jaxl.models.common import Model, Policy, StochasticPolicy


class MultitaskPolicy(Policy):
    """Multitask Policy."""

    # . The policy to use for interactions
    policy_head: int

    def __init__(self, policy: Policy, model: Model, num_tasks: int):
        super().__init__(model)
        self.policy = policy
        self.policy_head = 0
        self.num_tasks = num_tasks

    def set_policy_head(self, task_idx: int):
        """
        Sets the policy to use.

        :param task_int: the policy head index
        :type task_int: int
        """
        assert (
            0 <= task_idx < self.num_tasks
        ), f"task_idx {task_idx} needs to be from 0 to {self.num_tasks}"
        self.policy_head = task_idx

    def compute_action(
        self,
        params: Union[optax.Params, Dict[str, Any]],
        obs: chex.Array,
        h_state: chex.Array,
        key: jrandom.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Compute action to take in environment.

        :param params: the model parameters
        :param obs: the observation
        :param h_state: the hidden state
        :param key: the random number generator key for sampling
        :type params: Union[optax.Params, Dict[str, Any]]
        :type obs: chex.Array
        :type h_state: chex.Array
        :type key: jrandom.PRNGKey
        :return: an action and the next hidden state
        :rtype: Tuple[chex.Array, chex.Array]

        """
        acts, h_states = self.policy.compute_action(params, obs, h_state, key)
        return acts[self.policy_head], h_states[self.policy_head]

    def deterministic_action(
        self,
        params: Union[optax.Params, Dict[str, Any]],
        obs: chex.Array,
        h_state: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Compute deterministic action.

        :param params: the model parameters
        :param obs: the observation
        :param h_state: the hidden state
        :type params: Union[optax.Params, Dict[str, Any]]
        :type obs: chex.Array
        :type h_state: chex.Array
        :return: an action and the next hidden state
        :rtype: Tuple[chex.Array, chex.Array]

        """
        acts, h_states = self.policy.deterministic_action(params, obs, h_state)
        return acts[self.policy_head], h_states[self.policy_head]

    def random_action(
        self,
        params: Union[optax.Params, Dict[str, Any]],
        obs: chex.Array,
        h_state: chex.Array,
        key: jrandom.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Compute random action.

        :param params: the model parameters
        :param obs: the observation
        :param h_state: the hidden state
        :param key: the random number generator key for sampling
        :type params: Union[optax.Params, Dict[str, Any]]
        :type obs: chex.Array
        :type h_state: chex.Array
        :type key: jrandom.PRNGKey
        :return: an action and the next hidden state
        :rtype: Tuple[chex.Array, chex.Array]

        """
        acts, h_states = self.policy.random_action(params, obs, h_state, key)
        return acts[self.policy_head], h_states[self.policy_head]


class DeterministicPolicy(Policy):
    """Deterministic Policy."""

    def __init__(self, model: Model):
        super().__init__(model)
        self.compute_action = jax.jit(self.make_compute_action(model))
        self.deterministic_action = jax.jit(self.make_deterministic_action(model))

    def make_compute_action(
        self, model: Model
    ) -> Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the function for taking action during interaction

        :param model: the model
        :type model: Model
        :return: a function for taking action during interaction
        :rtype: Callable[
            [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array, jrandom.PRNGKey],
            Tuple[chex.Array, chex.Array],
        ]

        """

        def compute_action(
            params: Union[optax.Params, Dict[str, Any]],
            obs: chex.Array,
            h_state: chex.Array,
            key: jrandom.PRNGKey,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Compute action to take in environment.

            :param params: the model parameters
            :param obs: the observation
            :param h_state: the hidden state
            :param key: the random number generator key for sampling
            :type params: Union[optax.Params, Dict[str, Any]]
            :type obs: chex.Array
            :type h_state: chex.Array
            :type key: jrandom.PRNGKey
            :return: an action and the next hidden state
            :rtype: Tuple[chex.Array, chex.Array]

            """
            act, h_state = model.forward(params, obs, h_state)
            return act, h_state

        return compute_action

    def make_deterministic_action(
        self, model: Model
    ) -> Callable[
        [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the function for taking deterministic action.

        :param model: the model
        :type model: Model
        :return: a function for taking deterministic action
        :rtype: Callable[
            [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array],
            Tuple[chex.Array, chex.Array],
        ]

        """

        def deterministic_action(
            params: Union[optax.Params, Dict[str, Any]],
            obs: chex.Array,
            h_state: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Compute deterministic action.

            :param params: the model parameters
            :param obs: the observation
            :param h_state: the hidden state
            :type params: Union[optax.Params, Dict[str, Any]]
            :type obs: chex.Array
            :type h_state: chex.Array
            :return: an action and the next hidden state
            :rtype: Tuple[chex.Array, chex.Array]

            """
            act, h_state = model.forward(params, obs, h_state)
            return act, h_state

        return deterministic_action


class GaussianPolicy(StochasticPolicy):
    """Gaussian Policy."""

    def __init__(
        self,
        model: Model,
        min_std: float = DEFAULT_MIN_STD,
    ):
        super().__init__(model)
        self._min_std = min_std
        self.deterministic_action = jax.jit(self.make_deterministic_action(model))
        self.random_action = jax.jit(self.make_random_action(model))
        self.compute_action = jax.jit(self.make_compute_action(model))
        self.act_lprob = jax.jit(self.make_act_lprob(model))
        self.lprob = jax.jit(self.make_lprob(model))

    def make_compute_action(
        self, model: Model
    ) -> Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the function for taking action during interaction

        :param model: the model
        :type model: Model
        :return: a function for taking action during interaction
        :rtype: Callable[
            [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array, jrandom.PRNGKey],
            Tuple[chex.Array, chex.Array],
        ]

        """

        def compute_action(
            params: Union[optax.Params, Dict[str, Any]],
            obs: chex.Array,
            h_state: chex.Array,
            key: jrandom.PRNGKey,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Compute action to take in environment.

            :param params: the model parameters
            :param obs: the observation
            :param h_state: the hidden state
            :param key: the random number generator key for sampling
            :type params: Union[optax.Params, Dict[str, Any]]
            :type obs: chex.Array
            :type h_state: chex.Array
            :type key: jrandom.PRNGKey
            :return: an action and the next hidden state
            :rtype: Tuple[chex.Array, chex.Array]

            """
            act_params, h_state = model.forward(params, obs, h_state)
            act_mean, act_raw_std = jnp.split(act_params, 2, axis=-1)
            act_std = jax.nn.softplus(act_raw_std) + self._min_std
            act = Normal.sample(act_mean, act_std, key)
            return act, h_state

        return compute_action

    def make_deterministic_action(
        self, model: Model
    ) -> Callable[
        [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the function for taking deterministic action.

        :param model: the model
        :type model: Model
        :return: a function for taking deterministic action
        :rtype: Callable[
            [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array],
            Tuple[chex.Array, chex.Array],
        ]

        """

        def deterministic_action(
            params: Union[optax.Params, Dict[str, Any]],
            obs: chex.Array,
            h_state: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Compute deterministic action.

            :param params: the model parameters
            :param obs: the observation
            :param h_state: the hidden state
            :type params: Union[optax.Params, Dict[str, Any]]
            :type obs: chex.Array
            :type h_state: chex.Array
            :return: an action and the next hidden state
            :rtype: Tuple[chex.Array, chex.Array]

            """
            act_params, h_state = model.forward(params, obs, h_state)
            act_mean, _ = jnp.split(act_params, 2, axis=-1)
            return act_mean, h_state

        return deterministic_action

    def make_random_action(
        self, model: Model
    ) -> Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the function for taking random action.

        :param model: the model
        :type model: Model
        :return: a function for taking random action
        :rtype: Callable[
            [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array, jrandom.PRNGKey],
            Tuple[chex.Array, chex.Array],
        ]

        """

        def random_action(
            params: Union[optax.Params, Dict[str, Any]],
            obs: chex.Array,
            h_state: chex.Array,
            key: jrandom.PRNGKey,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Compute random action.

            :param params: the model parameters
            :param obs: the observation
            :param h_state: the hidden state
            :param key: the random number generator key for sampling
            :type params: Union[optax.Params, Dict[str, Any]]
            :type obs: chex.Array
            :type h_state: chex.Array
            :type key: jrandom.PRNGKey
            :return: an action and the next hidden state
            :rtype: Tuple[chex.Array, chex.Array]

            """
            act_params, h_state = model.forward(params, obs, h_state)
            act_mean, act_raw_std = jnp.split(act_params, 2, axis=-1)
            act_std = jax.nn.softplus(act_raw_std) + self._min_std
            act = Normal.sample(act_mean, act_std, key)
            return act, h_state

        return random_action

    def make_act_lprob(
        self, model: Model
    ) -> Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array, chex.Array],
    ]:
        """
        Makes the function for taking random action and computing its log probability.

        :param model: the model
        :type model: Model
        :return: a function for taking random action and computing its log probability
        :rtype: Callable[
            [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array, jrandom.PRNGKey],
            Tuple[chex.Array, chex.Array, chex.Array],
        ]

        """

        def act_lprob(
            params: Union[optax.Params, Dict[str, Any]],
            obs: chex.Array,
            h_state: chex.Array,
            key: jrandom.PRNGKey,
        ) -> Tuple[chex.Array, chex.Array, chex.Array]:
            """
            Compute action and its log probability.

            :param params: the model parameters
            :param obs: the observation
            :param h_state: the hidden state
            :param key: the random number generator key for sampling
            :type params: Union[optax.Params, Dict[str, Any]]
            :type obs: chex.Array
            :type h_state: chex.Array
            :type key: jrandom.PRNGKey
            :return: an action, its log probability, and the next hidden state
            :rtype: Tuple[chex.Array, chex.Array, chex.Array]

            """
            act_params, h_state = model.forward(params, obs, h_state)
            act_mean, act_raw_std = jnp.split(act_params, 2, axis=-1)
            act_std = jax.nn.softplus(act_raw_std) + self._min_std
            act = Normal.sample(act_mean, act_std, key)
            lprob = Normal.lprob(act_mean, act_std, act).sum(-1, keepdims=True)
            return act, lprob, h_state

        return act_lprob

    def make_lprob(
        self, model: Model
    ) -> Callable[
        [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
        chex.Array,
    ]:
        """
        Makes the function for computing action log probability.

        :param model: the model
        :type model: Model
        :return: a function for computing action log probability
        :rtype: Callable[
            [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
            chex.Array,
        ]

        """

        def lprob(
            params: Union[optax.Params, Dict[str, Any]],
            obs: chex.Array,
            h_state: chex.Array,
            act: chex.Array,
        ) -> chex.Array:
            """
            Compute action log probability.

            :param params: the model parameters
            :param obs: the observation
            :param h_state: the hidden state
            :param act: the action
            :type params: Union[optax.Params, Dict[str, Any]]
            :type obs: chex.Array
            :type h_state: chex.Array
            :type act: chex.Array
            :return: an action log probability
            :rtype: chex.Array

            """
            act_params, _ = model.forward(params, obs, h_state)
            act_mean, act_raw_std = jnp.split(act_params, 2, axis=-1)
            act_std = jax.nn.softplus(act_raw_std) + self._min_std
            lprob = Normal.lprob(act_mean, act_std, act).sum(-1, keepdims=True)
            return lprob

        return lprob
