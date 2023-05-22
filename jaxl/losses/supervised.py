from flax.core.scope import FrozenVariableDict
from types import SimpleNamespace
from typing import Any, Callable, Dict, Union, Tuple

import chex
import jax
import jax.numpy as jnp
import optax

from jaxl.constants import *
from jaxl.models import Model
from jaxl.utils import get_reduction


def make_squared_loss(
    model: Model,
    loss_setting: SimpleNamespace,
) -> Callable[
    [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
    Tuple[chex.Array, Dict],
]:
    reduction = get_reduction(loss_setting.reduction)

    def squared_loss(
        params: Union[FrozenVariableDict, Dict[str, Any]],
        x: chex.Array,
        carry: chex.Array,
        y: chex.Array,
    ) -> Tuple[chex.Array, Dict]:
        y_pred, _ = model.forward(params, x, carry)
        return reduction((y_pred - y) ** 2), {CONST_PREDICTIONS: y_pred}

    return squared_loss


def make_cross_entropy_loss(
    model: Model,
    loss_setting: SimpleNamespace,
) -> Callable[
    [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
    Tuple[chex.Array, Dict],
]:
    reduction = get_reduction(loss_setting.reduction)

    def cross_entropy_loss(
        params: Union[FrozenVariableDict, Dict[str, Any]],
        x: chex.Array,
        carry: chex.Array,
        y: chex.Array,
    ) -> Tuple[chex.Array, Dict]:
        logits, _ = model.forward(params, x, carry)
        y_one_hot = jax.nn.one_hot(jnp.squeeze(y), num_classes=loss_setting.num_classes)

        return reduction(optax.softmax_cross_entropy(logits, y_one_hot)), {}

    return cross_entropy_loss
