from flax.core.scope import FrozenVariableDict
from flax import linen as nn
from types import SimpleNamespace
from typing import Any, Callable, Dict, Union, Tuple

import chex
import jax
import jax.numpy as jnp


def make_weight_decay(
    model: nn.Module,
    loss_setting: SimpleNamespace,
) -> Callable[
    [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array],
    Tuple[chex.Array, Dict],
]:
    def weight_decay(
        params: Union[FrozenVariableDict, Dict[str, Any]], *args, **kwargs
    ) -> Tuple[chex.Array, Dict]:
        return (
            sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)),
            {},
        )

    return weight_decay
