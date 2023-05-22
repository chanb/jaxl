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
    def __init__(self, layers: Sequence[int]) -> None:
        self.model = MLPModule(layers)
        self.forward = jax.jit(self.make_forward())

    def init(
        self, model_key: jrandom.PRNGKey, dummy_x: chex.Array
    ) -> Union[FrozenVariableDict, Dict[str, Any]]:
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
        def forward(
            params: Union[FrozenVariableDict, Dict[str, Any]],
            input: chex.Array,
            carry: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            # NOTE: Assume batch size is first dim
            input = input.reshape((input.shape[0], -1))
            return self.model.apply(params, input), carry

        return forward


class Policy(ABC):
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
        self, policy: nn.Module
    ) -> Callable[
        [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array],
        Tuple[chex.Array, chex.Array],
    ]:
        raise NotImplementedError

    def reset(self):
        return np.array([0.0], dtype=np.float32)


class StochasticPolicy(Policy):
    @abstractmethod
    def make_random_action(
        self, policy: nn.Module
    ) -> Callable[
        [
            Union[FrozenVariableDict, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        raise NotImplementedError

    @abstractmethod
    def make_act_lprob(
        self, policy: nn.Module
    ) -> Callable[
        [
            Union[FrozenVariableDict, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array, chex.Array],
    ]:
        raise NotImplementedError

    @abstractmethod
    def make_lprob(
        self, policy: nn.Module
    ) -> Callable[
        [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
        chex.Array,
    ]:
        raise NotImplementedError
