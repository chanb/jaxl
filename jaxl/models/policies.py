from abc import ABC
from types import SimpleNamespace
from typing import Any, Callable, Dict, Union, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from jaxl.constants import *
from jaxl.distributions import Bernoulli, Normal, Softmax, get_transform
from jaxl.distributions.transforms import TanhTransform
from jaxl.models.common import Model


class Policy(ABC):
    """Abstract policy class."""

    #: Compute action for interacting with the environment.
    compute_action: Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]

    #: Compute deterministic action.
    deterministic_action: Callable[
        [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array],
        Tuple[chex.Array, chex.Array],
    ]

    def __init__(self, model: Model) -> None:
        self.reset = jax.jit(self.make_reset(model))

    def make_reset(self, model: Model) -> Callable[..., chex.Array]:
        """
        Makes the function that resets the policy.
        This is often used for resetting the hidden state.

        :param model: the model
        :type model: Model
        :return: a function for initializing the hidden state
        :rtype: chex.Array
        """

        def _reset() -> chex.Array:
            """
            Resets hidden state.

            :return: a hidden state
            :rtype: chex.Array
            """
            return model.reset_h_state()

        return _reset


class StochasticPolicy(Policy):
    """Abstract stochastic policy class that extends ``Policy``."""

    #: Compute random action.
    random_action: Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]

    # . Compute action and its log probability.
    act_lprob: Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array, chex.Array],
    ]

    # . Compute action log probability.
    lprob: Callable[
        [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
        chex.Array,
    ]


class MultitaskPolicy(Policy):
    """Multitask Policy."""

    # : The policy to use for interactions
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
        std_transform: Callable[chex.Array, chex.Array] = jax.nn.squareplus,
    ):
        super().__init__(model)
        self._min_std = min_std
        self._std_transform = std_transform
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
            act_std = self._std_transform(act_raw_std) + self._min_std
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
            act_std = self._std_transform(act_raw_std) + self._min_std
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
            act_std = self._std_transform(act_raw_std) + self._min_std
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
        ) -> Tuple[chex.Array, Dict[str, Any]]:
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
            :return: an action log probability and distribution parameters
            :rtype: Tuple[chex.Array, Dict[str, Any]]

            """
            act_params, _ = model.forward(params, obs, h_state)
            act_mean, act_raw_std = jnp.split(act_params, 2, axis=-1)
            act_std = self._std_transform(act_raw_std) + self._min_std
            lprob = Normal.lprob(act_mean, act_std, act).sum(-1, keepdims=True)
            return lprob, {
                CONST_MEAN: act_mean,
                CONST_STD: act_std,
                CONST_ENTROPY: Normal.entropy(act_mean, act_std),
            }

        return lprob


class SquashedGaussianPolicy(StochasticPolicy):
    """Squashed Gaussian Policy."""

    def __init__(
        self,
        model: Model,
        min_std: float = DEFAULT_MIN_STD,
        std_transform: Callable[chex.Array, chex.Array] = jax.nn.squareplus,
    ):
        super().__init__(model)
        self._min_std = min_std
        self._std_transform = std_transform
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
            act_std = self._std_transform(act_raw_std) + self._min_std
            act = Normal.sample(act_mean, act_std, key)
            act = TanhTransform.transform(act)
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
            act_mean = TanhTransform.transform(act_mean)
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
            act_std = self._std_transform(act_raw_std) + self._min_std
            act = Normal.sample(act_mean, act_std, key)
            act = TanhTransform.transform(act)
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
            act_std = self._std_transform(act_raw_std) + self._min_std
            act = Normal.sample(act_mean, act_std, key)
            act_t = TanhTransform.transform(act)
            lprob = Normal.lprob(act_mean, act_std, act).sum(-1, keepdims=True)
            lprob = lprob - TanhTransform.log_abs_det_jacobian(act, act_t)
            return act_t, lprob, h_state

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
        ) -> Tuple[chex.Array, Dict[str, Any]]:
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
            :return: an action log probability and distribution parameters
            :rtype: Tuple[chex.Array, Dict[str, Any]]

            """
            act_params, _ = model.forward(params, obs, h_state)
            act_mean, act_raw_std = jnp.split(act_params, 2, axis=-1)
            act_std = self._std_transform(act_raw_std) + self._min_std

            act_inv = TanhTransform.inv(act)

            lprob = Normal.lprob(act_inv, act_std, act).sum(-1, keepdims=True)
            lprob = lprob - TanhTransform.log_abs_det_jacobian(act_inv, act)
            return lprob, {
                CONST_MEAN: act_mean,
                CONST_STD: act_std,
                CONST_ENTROPY: Normal.entropy(act_inv, act_std),
            }

        return lprob


class SoftmaxPolicy(StochasticPolicy):
    """Softmax Policy."""

    def __init__(
        self,
        model: Model,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        super().__init__(model)
        assert temperature > 0.0, "temperature needs to be positive, got {}".format(
            temperature
        )
        self._temperature = temperature
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
            act = Softmax.sample(act_params / self._temperature, key)
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
            act_max = jnp.argmax(act_params, axis=-1)
            return act_max, h_state

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
            act = Softmax.sample(act_params / self._temperature, key)
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
            act_params = act_params / self._temperature
            act = Softmax.sample(act_params, key)
            lprob = Softmax.lprob(act_params, act)
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
        ) -> Tuple[chex.Array, Dict[str, Any]]:
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
            :return: an action log probability and distribution parameters
            :rtype: Tuple[chex.Array, Dict[str, Any]]

            """
            act_params, _ = model.forward(params, obs, h_state)
            act_params = act_params / self._temperature
            lprob = Softmax.lprob(act_params, act)
            return lprob, {
                CONST_LOGITS: act_params,
                CONST_ENTROPY: Softmax.entropy(act_params),
            }

        return lprob


class BangBangPolicy(StochasticPolicy):
    """BangBang Policy."""

    def __init__(
        self,
        model: Model,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        super().__init__(model)
        assert temperature > 0.0, "temperature needs to be positive, got {}".format(
            temperature
        )
        self._temperature = temperature
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
            act = Bernoulli.sample(act_params / self._temperature, key)
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
            act_max, _ = jnp.argmax(act_params / self._temperature, axis=-1)
            return act_max, h_state

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
            act = Bernoulli.sample(act_params / self._temperature, key)
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
            logits = act_params / self._temperature
            act = Bernoulli.sample(logits, key)
            lprob = Bernoulli.lprob(logits, act).sum(-1, keepdims=True)
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
        ) -> Tuple[chex.Array, Dict[str, Any]]:
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
            :return: an action log probability and distribution parameters
            :rtype: Tuple[chex.Array, Dict[str, Any]]

            """
            act_params, _ = model.forward(params, obs, h_state)
            logits = act_params / self._temperature
            lprob = Bernoulli.lprob(logits, act).sum(-1, keepdims=True)
            return lprob, {
                CONST_ENTROPY: Bernoulli.entropy(logits),
            }

        return lprob


def get_policy(model: Model, config: SimpleNamespace) -> Policy:
    """
    Gets a policy

    :param model: a model
    :param config: the policy configuration
    :type model: Model
    :type config: SimpleNamespace
    :return: a policy
    :rtype: Policy

    """
    assert (
        config.policy_distribution in VALID_POLICY_DISTRIBUTION
    ), f"{config.policy_distribution} is not supported (one of {VALID_POLICY_DISTRIBUTION})"

    if config.policy_distribution == CONST_GAUSSIAN:
        return GaussianPolicy(
            model,
            getattr(config, CONST_MIN_STD, DEFAULT_MIN_STD),
            get_transform(getattr(config, CONST_STD_TRANSFORM, CONST_SQUAREPLUS)),
        )
    elif config.policy_distribution == CONST_SQUASHED_GAUSSIAN:
        return SquashedGaussianPolicy(
            model,
            getattr(config, CONST_MIN_STD, DEFAULT_MIN_STD),
            get_transform(getattr(config, CONST_STD_TRANSFORM, CONST_SQUAREPLUS)),
        )
    elif config.policy_distribution == CONST_DETERMINISTIC:
        return DeterministicPolicy(model)
    elif config.policy_distribution == CONST_SOFTMAX:
        return SoftmaxPolicy(
            model, getattr(config, CONST_TEMPERATURE, DEFAULT_TEMPERATURE)
        )
    elif config.policy_distribution == CONST_BANG_BANG:
        return BangBangPolicy(
            model, getattr(config, CONST_TEMPERATURE, DEFAULT_TEMPERATURE)
        )
    else:
        raise NotImplementedError


def policy_output_dim(output_dim: chex.Array, config: SimpleNamespace) -> chex.Array:
    """
    Gets the policy output dimension based on its distribution.

    :param output_dim: the original output dimension
    :param config: the policy configuration
    :type output_dim: chex.Array
    :type config: SimpleNamespace
    :return: the output dimension of the policy
    :rtype: chex.Array

    """
    assert (
        config.policy_distribution in VALID_POLICY_DISTRIBUTION
    ), f"{config.policy_distribution} is not supported (one of {VALID_POLICY_DISTRIBUTION})"
    if config.policy_distribution == CONST_GAUSSIAN:
        return [*output_dim[:-1], output_dim[-1] * 2]
    return output_dim
