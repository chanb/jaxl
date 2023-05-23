from abc import ABC, abstractmethod
from flax import linen as nn
from flax.core.scope import FrozenVariableDict
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import chex
import jax
import jax.random as jrandom
import numpy as np

from jaxl.constants import *
from jaxl.models.modules import MLPModule


class Model(ABC):
    """Abstract model class."""

    forward: Callable[
        [
            Union[FrozenVariableDict, Dict[str, Any]],
            chex.Array,
            chex.Array,
        ],
        Tuple[chex.Array, chex.Array],
    ]

    init: Callable[
        [
            jrandom.PRNGKey,
            chex.Array,
        ],
        Union[FrozenVariableDict, Dict[str, Any]],
    ]


class MLP(Model):
    """A multilayer perceptron."""

    def __init__(self, layers: Sequence[int]) -> None:
        self.model = MLPModule(layers)
        self.forward = jax.jit(self.make_forward())

    def init(
        self, model_key: jrandom.PRNGKey, dummy_x: chex.Array
    ) -> Union[FrozenVariableDict, Dict[str, Any]]:
        """
        Initialize model parameters.

        :param model_key: the random number generation key for initializing parameters
        :param dummy_x: the input data
        :type model_key: jrandom.PRNGKey
        :type dummy_x: chex.Array
        :return: the initialized parameters
        :rtype: Union[FrozenVariableDict, Dict[str, Any]]

        """
        return self.model.init(model_key, dummy_x)

    def make_forward(
        self,
    ) -> Callable[
        [
            Union[FrozenVariableDict, Dict[str, Any]],
            chex.Array,
            chex.Array,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the forward call of the MLP model.

        :return: the forward call.
        :rtype: Callable[
            [
                Union[FrozenVariableDict, Dict[str, Any]],
                chex.Array,
                chex.Array,
            ],
            Tuple[chex.Array, chex.Array],
        ]
        """

        def forward(
            params: Union[FrozenVariableDict, Dict[str, Any]],
            input: chex.Array,
            carry: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Forward call of the MLP.

            :param params: the model parameters
            :param input: the input
            :param carry: the hidden state (not used)
            :type params: Union[FrozenVariableDict
            :type input: chex.Array
            :type carry: chex.Array
            :return: the output and a pass-through carry
            :rtype: Tuple[chex.Array, chex.Array]

            """
            # NOTE: Assume batch size is first dim
            input = input.reshape((input.shape[0], -1))
            return self.model.apply(params, input), carry

        return forward


class Policy(ABC):
    """Abstract policy class."""

    compute_action: Callable[
        [
            Union[FrozenVariableDict, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]

    @abstractmethod
    def make_deterministic_action(
        self, policy: Model
    ) -> Callable[
        [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the function for taking deterministic action.

        :param policy: the policy
        :type policy: Model
        :return: a function for taking deterministic action
        :rtype: Callable[
            [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array],
            Tuple[chex.Array, chex.Array],
        ]

        """
        raise NotImplementedError

    def reset(self) -> chex.Array:
        """
        Resets hidden state.

        :return: a hidden state
        :rtype: chex.Array
        """
        return np.array([0.0], dtype=np.float32)


class StochasticPolicy(Policy):
    """Abstract stochastic policy class that extends ``Policy``."""

    @abstractmethod
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
        """
        Makes the function for taking random action.

        :param policy: the policy
        :type policy: Model
        :return: a function for taking random action
        :rtype: Callable[
            [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array, jrandom.PRNGKey],
            Tuple[chex.Array, chex.Array],
        ]

        """
        raise NotImplementedError

    @abstractmethod
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
        """
        Makes the function for taking random action and computing its log probability.

        :param policy: the policy
        :type policy: Model
        :return: a function for taking random action and computing its log probability
        :rtype: Callable[
            [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array, jrandom.PRNGKey],
            Tuple[chex.Array, chex.Array, chex.Array],
        ]

        """
        raise NotImplementedError

    @abstractmethod
    def make_lprob(
        self, policy: Model
    ) -> Callable[
        [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
        chex.Array,
    ]:
        """
        Makes the function for computing action log probability.

        :param policy: the policy
        :type policy: Model
        :return: a function for computing action log probability
        :rtype: Callable[
            [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
            chex.Array,
        ]

        """
        raise NotImplementedError
