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
    *args,
    **kwargs,
) -> Callable[
    [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
    Tuple[chex.Array, Dict],
]:
    """
    Gets squared loss function.

    :param model: the model
    :param loss_setting: the loss configuration
    :type model: Model
    :type loss_setting: SimpleNamespace
    :return: the loss function
    :rtype: Callable[..., chex.Array]

    """
    reduction = get_reduction(loss_setting.reduction)

    def squared_loss(
        params: Union[optax.Params, Dict[str, Any]],
        x: chex.Array,
        carry: chex.Array,
        y: chex.Array,
    ) -> Tuple[chex.Array, Dict]:
        """
        Squared Loss.

        :param params: the model parameters
        :param x: the input
        :param carry: the hidden state
        :param y: the output
        :param *args:
        :param **kwargs:
        :type params: Union[optax.Params, Dict[str, Any]]
        :return: the loss and auxiliary information
        :rtype: Tuple[chex.Array, Dict]

        """
        y_pred, _ = model.forward(params, x, carry)
        return reduction((y_pred - y) ** 2), {CONST_PREDICTIONS: y_pred}

    return squared_loss


def make_cross_entropy_loss(
    model: Model,
    loss_setting: SimpleNamespace,
    num_classes: int,
    *args,
    **kwargs,
) -> Callable[
    [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
    Tuple[chex.Array, Dict],
]:
    """
    Gets cross-entropy loss function.

    :param model: the model
    :param loss_setting: the loss configuration
    :param num_classes: the number of classes
    :type model: Model
    :type loss_setting: SimpleNamespace
    :type num_classes: int
    :return: the loss function
    :rtype: Callable[..., chex.Array]

    """
    reduction = get_reduction(loss_setting.reduction)

    def cross_entropy_loss(
        params: Union[optax.Params, Dict[str, Any]],
        x: chex.Array,
        carry: chex.Array,
        y: chex.Array,
    ) -> Tuple[chex.Array, Dict]:
        """
        Cross-entropy Loss.

        :param params: the model parameters
        :param x: the input
        :param carry: the hidden state
        :param y: the output
        :param *args:
        :param **kwargs:
        :type params: Union[optax.Params, Dict[str, Any]]
        :return: the loss and auxiliary information
        :rtype: Tuple[chex.Array, Dict]

        """
        logits, _ = model.forward(params, x, carry)
        y_one_hot = jax.nn.one_hot(jnp.squeeze(y), num_classes=num_classes)

        return reduction(optax.softmax_cross_entropy(logits, y_one_hot)), {}

    return cross_entropy_loss
