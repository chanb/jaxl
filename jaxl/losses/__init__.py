from flax.core.scope import FrozenVariableDict
from flax import linen as nn
from types import SimpleNamespace
from typing import Any, Callable, Dict, Union, Tuple

import chex
import numpy as np

from jaxl.constants import *
from jaxl.losses.regularization import *
from jaxl.losses.reinforcement import *
from jaxl.losses.supervised import *
from jaxl.models.common import Model


def get_worst_value_for_loss(loss: str) -> chex.Array:
    assert loss in VALID_LOSS, f"{loss} is not supported (one of {VALID_LOSS})"
    if loss == CONST_GAUSSIAN:
        return np.inf
    elif loss == CONST_L2:
        return np.inf

    raise NotImplementedError


def get_loss_function(
    model: Model, loss: str, loss_setting: SimpleNamespace
) -> Callable[..., chex.Array]:
    assert loss in VALID_LOSS, f"{loss} is not supported (one of {VALID_LOSS})"

    if loss == CONST_GAUSSIAN:
        make_loss_function = make_squared_loss
    elif loss == CONST_CATEGORICAL:
        make_loss_function = make_cross_entropy_loss
    elif loss == CONST_L2:
        make_loss_function = make_weight_decay
    else:
        raise NotImplementedError
    return make_loss_function(model, loss_setting)


def make_aggregate_loss(
    losses: Dict[str, Tuple[Callable[..., chex.Array], float]]
) -> Callable[
    [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
    Tuple[chex.Array, Dict],
]:
    def compute_aggregate_loss(
        params: Union[FrozenVariableDict, Dict[str, Any]],
        x: chex.Array,
        carry: chex.Array,
        y: chex.Array,
    ):
        total_loss = 0.0
        aux = {}
        for loss_key, (loss_function, coefficient) in losses.items():
            loss, loss_aux = loss_function(params, x, carry, y)
            aux[loss_key] = {
                CONST_LOSS: loss,
                CONST_AUX: loss_aux,
            }
            total_loss += coefficient * loss
        return total_loss, aux

    return compute_aggregate_loss
