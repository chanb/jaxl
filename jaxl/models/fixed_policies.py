from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple

import chex
import jax
import jax.random as jrandom
import numpy as np

from jaxl.constants import *
from jaxl.distributions import Normal
from jaxl.distributions.transforms import TanhTransform
from jaxl.models.common import Model
from jaxl.models.policies import Policy


class FixedModel(Model):
    pass


class FixedSquashedGaussianPolicy(Policy):
    def __init__(
        self,
        means: chex.Array,
        stds: chex.Array,
    ):
        super().__init__(FixedModel())
        assert np.all(stds > 0), "stds {} should be positive".format(stds)
        self._means = means
        self._stds = stds
        self.deterministic_action = jax.jit(self.make_deterministic_action())
        self.random_action = jax.jit(self.make_random_action())
        self.compute_action = jax.jit(self.make_compute_action())
        self.act_lprob = jax.jit(self.make_act_lprob())
        self.lprob = jax.jit(self.make_lprob())

    def make_compute_action(
        self,
    ) -> Callable[
        [
            Any,
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the function for taking action during interaction

        :return: a function for taking action during interaction
        :rtype: Callable[
            [Any, chex.Array, chex.Array, jrandom.PRNGKey],
            Tuple[chex.Array, chex.Array],
        ]

        """

        def compute_action(
            params: Any,
            obs: chex.Array,
            h_state: chex.Array,
            key: jrandom.PRNGKey,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Compute action to take in environment.

            :param params: the parameters of the policy
            :param obs: the observation
            :param h_state: the hidden state
            :param key: the random number generator key for sampling
            :type params: Any
            :type obs: chex.Array
            :type h_state: chex.Array
            :type key: jrandom.PRNGKey
            :return: an action and the next hidden state
            :rtype: Tuple[chex.Array, chex.Array]

            """
            act = Normal.sample(self._means, self._stds, key)
            act = TanhTransform.transform(act)
            return act, h_state

        return compute_action

    def make_deterministic_action(
        self,
    ) -> Callable[
        [Any, chex.Array, chex.Array],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the function for taking deterministic action.

        :return: a function for taking deterministic action
        :rtype: Callable[
            [Any, chex.Array, chex.Array],
            Tuple[chex.Array, chex.Array],
        ]

        """

        def deterministic_action(
            params: Any,
            obs: chex.Array,
            h_state: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Compute deterministic action.

            :param params: the parameters of the policy
            :param obs: the observation
            :param h_state: the hidden state
            :type params: Any
            :type obs: chex.Array
            :type h_state: chex.Array
            :return: an action and the next hidden state
            :rtype: Tuple[chex.Array, chex.Array]

            """
            act_mean = TanhTransform.transform(self._means)
            return act_mean, h_state

        return deterministic_action

    def make_random_action(
        self,
    ) -> Callable[
        [
            Any,
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the function for taking random action.

        :return: a function for taking random action
        :rtype: Callable[
            [Any, chex.Array, chex.Array, jrandom.PRNGKey],
            Tuple[chex.Array, chex.Array],
        ]

        """

        def random_action(
            params: Any,
            obs: chex.Array,
            h_state: chex.Array,
            key: jrandom.PRNGKey,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Compute random action.

            :param params: the parameters of the policy
            :param obs: the observation
            :param h_state: the hidden state
            :param key: the random number generator key for sampling
            :type params: Any
            :type obs: chex.Array
            :type h_state: chex.Array
            :type key: jrandom.PRNGKey
            :return: an action and the next hidden state
            :rtype: Tuple[chex.Array, chex.Array]

            """
            act = Normal.sample(self._means, self._stds, key)
            act = TanhTransform.transform(act)
            return act, h_state

        return random_action

    def make_act_lprob(
        self,
    ) -> Callable[
        [
            Any,
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array, chex.Array],
    ]:
        """
        Makes the function for taking random action and computing its log probability.

        :return: a function for taking random action and computing its log probability
        :rtype: Callable[
            [Any, chex.Array, chex.Array, jrandom.PRNGKey],
            Tuple[chex.Array, chex.Array, chex.Array],
        ]

        """

        def act_lprob(
            params: Any,
            obs: chex.Array,
            h_state: chex.Array,
            key: jrandom.PRNGKey,
        ) -> Tuple[chex.Array, chex.Array, chex.Array]:
            """
            Compute action and its log probability.

            :param params: the parameters of the policy
            :param obs: the observation
            :param h_state: the hidden state
            :param key: the random number generator key for sampling
            :type params: Any
            :type obs: chex.Array
            :type h_state: chex.Array
            :type key: jrandom.PRNGKey
            :return: an action, its log probability, and the next hidden state
            :rtype: Tuple[chex.Array, chex.Array, chex.Array]

            """
            act = Normal.sample(self._means, self._stds, key)
            act_t = TanhTransform.transform(act)
            lprob = Normal.lprob(self._means, self._stds, act).sum(-1, keepdims=True)
            lprob = lprob - TanhTransform.log_abs_det_jacobian(act, act_t)
            return act_t, lprob, h_state

        return act_lprob

    def make_lprob(
        self,
    ) -> Callable[
        [Any, chex.Array, chex.Array, chex.Array],
        chex.Array,
    ]:
        """
        Makes the function for computing action log probability.

        :return: a function for computing action log probability
        :rtype: Callable[
            [Any, chex.Array, chex.Array, chex.Array],
            chex.Array,
        ]

        """

        def lprob(
            params: Any,
            obs: chex.Array,
            h_state: chex.Array,
            act: chex.Array,
        ) -> Tuple[chex.Array, Dict[str, Any]]:
            """
            Compute action log probability.

            :param params: the parameters of the policy
            :param obs: the observation
            :param h_state: the hidden state
            :param act: the action
            :type params: Any
            :type obs: chex.Array
            :type h_state: chex.Array
            :type act: chex.Array
            :return: an action log probability and distribution parameters
            :rtype: Tuple[chex.Array, Dict[str, Any]]

            """
            act_inv = TanhTransform.inv(act)

            lprob = Normal.lprob(self._means, self._stds, act_inv).sum(-1, keepdims=True)
            lprob = lprob - TanhTransform.log_abs_det_jacobian(act_inv, act)
            return lprob, {
                CONST_MEAN: self._means,
                CONST_STD: self._stds,
                CONST_ENTROPY: Normal.entropy(act_inv, self._stds),
            }

        return lprob


def get_fixed_policy(act_dim: chex.Array, config: SimpleNamespace) -> Policy:
    """
    Gets a fixed policy

    :param act_dim: the action dimensionality
    :param config: the policy configuration
    :type act_dim: chex.Array
    :type config: SimpleNamespace
    :return: a policy
    :rtype: Policy

    """
    assert (
        config.policy_distribution in VALID_EXPLORATION_POLICY
    ), f"{config.policy_distribution} is not supported (one of {VALID_EXPLORATION_POLICY})"

    if config.policy_distribution == CONST_SQUASHED_GAUSSIAN:
        config_kwargs = config.kwargs
        means = None
        stds = None

        if isinstance(config_kwargs.means, float):
            means = np.ones(act_dim) * config_kwargs.means
        else:
            raise NotImplementedError

        if isinstance(config_kwargs.stds, float):
            stds = np.ones(act_dim) * config_kwargs.stds
        else:
            raise NotImplementedError

        return FixedSquashedGaussianPolicy(
            means=means,
            stds=stds,
        )
    else:
        raise NotImplementedError