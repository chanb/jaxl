from abc import ABC
from collections.abc import Iterable
from functools import partial
from flax import linen as nn
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import jax.numpy as jnp
import jax.random as jrandom
import optax

from jaxl.constants import *
from jaxl.models.common import get_activation, Model
from jaxl.models.modules import *


class MLPWithStatelessStd(Model):
    """A multilayer perceptron that also outputs."""

    def __init__(
        self,
        layers: Sequence[int],
        activation: str = CONST_RELU,
        output_activation: str = CONST_IDENTITY,
        use_batch_norm: bool = False,
        use_bias: bool = True,
        flatten: bool = False,
        init_log_std: float = 1.0,
    ) -> None:
        self.use_batch_norm = use_batch_norm
        self.model = MLPModule(
            layers,
            get_activation(activation),
            get_activation(output_activation),
            use_batch_norm,
            use_bias,
            flatten,
        )
        self.log_std = ParameterVector(layers[-1], init_log_std)
        self.forward = jax.jit(self.make_forward(), static_argnames=[CONST_EVAL])

    def init(
        self, model_key: jrandom.PRNGKey, dummy_x: chex.Array
    ) -> Union[optax.Params, Dict[str, Any]]:
        return {
            CONST_MEAN: self.model.init(model_key, dummy_x, eval=True),
            CONST_STD: self.log_std.init(model_key),
        }

    def make_forward(
        self,
    ) -> Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
            bool,
        ],
        Tuple[chex.Array, chex.Array, Any],
    ]:
        def forward(
            params: Union[optax.Params, Dict[str, Any]],
            input: chex.Array,
            carry: chex.Array,
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array, Any]:
            # NOTE: Assume batch size is first dim
            (mean, updates) = self.model.apply(
                params[CONST_MEAN],
                input,
                eval,
                mutable=[CONST_BATCH_STATS],
            )
            log_std = self.log_std.apply(params[CONST_STD])
            std = jnp.tile(jnp.exp(log_std), reps=(len(mean), 1))
            return jnp.concatenate((mean, std), axis=-1), carry, {CONST_MEAN: updates}

        return forward

    def update_batch_stats(
        self, params: Dict[str, Any], batch_stats: Any
    ) -> Dict[str, Any]:
        if self.use_batch_norm:
            params[CONST_MEAN][CONST_BATCH_STATS] = batch_stats[CONST_MEAN][
                CONST_BATCH_STATS
            ]
        return params
