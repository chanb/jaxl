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
        y_pred, _, updates = model.forward(params, x, carry)
        return reduction((y_pred - y) ** 2), {
            CONST_PREDICTIONS: y_pred,
            CONST_UPDATES: updates,
        }

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

    if getattr(loss_setting, "is_one_hot", False):

        def convert_to_one_hot(y):
            return y

    else:

        def convert_to_one_hot(y):
            return jax.nn.one_hot(jnp.squeeze(y), num_classes=num_classes)

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
        logits, _, updates = model.forward(params, x, carry)
        y_one_hot = convert_to_one_hot(y)

        return reduction(optax.softmax_cross_entropy(logits, y_one_hot)), {
            "logits": logits,
            "y_one_hot": y_one_hot,
            CONST_UPDATES: updates,
        }

    return cross_entropy_loss


def make_hinge_loss(
    model: Model,
    loss_setting: SimpleNamespace,
    num_classes: int,
    margin: float = 1.0,
    *args,
    **kwargs,
) -> Callable[
    [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
    Tuple[chex.Array, Dict],
]:
    """
    Gets hinge loss function.

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

    if getattr(loss_setting, "is_one_hot", False):

        def convert_to_one_hot(y):
            return y

    else:

        def convert_to_one_hot(y):
            return jax.nn.one_hot(jnp.squeeze(y), num_classes=num_classes)

    def hinge_loss(
        params: Union[optax.Params, Dict[str, Any]],
        x: chex.Array,
        carry: chex.Array,
        y: chex.Array,
    ) -> Tuple[chex.Array, Dict]:
        """
        Hinge Loss.

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
        logits, _, updates = model.forward(params, x, carry)
        y_one_hot = convert_to_one_hot(y)

        return (
            reduction(
                jnp.clip(
                    margin
                    - jnp.sum(logits * y_one_hot, axis=-1)
                    + jnp.sum(logits * (1 - y_one_hot), axis=-1),
                    a_min=0.0,
                    a_max=jnp.inf,
                )
            ),
            {
                CONST_UPDATES: updates,
            },
        )

    return hinge_loss


def make_sigmoid_bce_loss(
    model: Model,
    loss_setting: SimpleNamespace,
    *args,
    **kwargs,
) -> Callable[
    [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
    Tuple[chex.Array, Dict],
]:
    """
    Gets sigmoid binary cross-entropy loss function.

    :param model: the model
    :param loss_setting: the loss configuration
    :type model: Model
    :type loss_setting: SimpleNamespace
    :return: the loss function
    :rtype: Callable[..., chex.Array]

    """
    reduction = get_reduction(loss_setting.reduction)

    if getattr(loss_setting, "is_one_hot", False):

        def convert_to_label(y):
            return jnp.argmax(y, axis=-1, keepdims=True)

    else:

        def convert_to_label(y):
            return y

    def sigmoid_bce_loss(
        params: Union[optax.Params, Dict[str, Any]],
        x: chex.Array,
        carry: chex.Array,
        y: chex.Array,
    ) -> Tuple[chex.Array, Dict]:
        """
        Sigmoid binary cross-entropy Loss.

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
        logits, _, updates = model.forward(params, x, carry)
        y_label = convert_to_label(y)

        return reduction(optax.sigmoid_binary_cross_entropy(logits, y_label)), {
            CONST_UPDATES: updates,
        }

    return sigmoid_bce_loss
