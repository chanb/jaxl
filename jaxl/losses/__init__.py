from types import SimpleNamespace
from typing import Any, Callable, Dict, Union, Tuple

import chex
import optax

from jaxl.constants import *
from jaxl.losses.regularization import *
from jaxl.losses.reinforcement import *
from jaxl.losses.supervised import *
from jaxl.models.common import Model


def get_loss_function(
    model: Model,
    loss: str,
    loss_setting: SimpleNamespace,
    *args,
    **kwargs,
) -> Callable[..., chex.Array]:
    """
    Gets a loss function.

    :param model: the model
    :param loss: the loss name
    :param loss_setting: the loss configuration
    :type model: Model
    :type loss: str
    :type loss_setting: SimpleNamespace
    :return: the loss function
    :rtype: Callable[..., chex.Array]

    """
    assert loss in VALID_LOSS, f"{loss} is not supported (one of {VALID_LOSS})"

    if loss == CONST_GAUSSIAN:
        make_loss_function = make_squared_loss
    elif loss == CONST_CATEGORICAL:
        make_loss_function = make_cross_entropy_loss
    elif loss == CONST_L2:
        make_loss_function = make_weight_decay
    else:
        raise NotImplementedError
    return make_loss_function(model, loss_setting, *args, **kwargs)


def make_aggregate_loss(
    losses: Dict[str, Tuple[Callable[..., chex.Array], float]]
) -> Callable[
    [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
    Tuple[chex.Array, Dict],
]:
    """
    Aggregates losses by a weighted sum.

    :param losses: the loss functions to aggregate
    :type losses: Dict[str, Tuple[Callable[..., chex.Array], float]]
    :return: the aggregated loss function
    :type: Callable[
        [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
        Tuple[chex.Array, Dict],
    ]

    """

    def compute_aggregate_loss(
        params: Union[optax.Params, Dict[str, Any]],
        x: chex.Array,
        carry: chex.Array,
        y: chex.Array,
    ):
        """

        :param params: Union[optax.Params:
        :param Dict[str:
        :param Any]]:
        :param x: chex.Array:
        :param carry: chex.Array:
        :param y: chex.Array:

        """
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
