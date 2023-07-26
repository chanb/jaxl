from types import SimpleNamespace
from typing import Any, Callable, Dict, Union, Tuple

import chex
import jax
import jax.numpy as jnp
import optax

from jaxl.models.common import Model


def make_weight_decay(
    model: Model,
    loss_setting: SimpleNamespace,
    *args,
    **kwargs,
) -> Callable[
    [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array],
    Tuple[chex.Array, Dict],
]:
    """
    Gets weight decay function.

    :param model: the model
    :param loss_setting: the loss configuration
    :type model: Model
    :type loss_setting: SimpleNamespace
    :return: the loss function
    :rtype: Callable[..., chex.Array]

    """

    def weight_decay(
        params: Union[optax.Params, Dict[str, Any]], *args, **kwargs
    ) -> Tuple[chex.Array, Dict]:
        """
        Weight decay.

        :param params: the model parameters
        :param *args:
        :param **kwargs:
        :type params: Union[optax.Params, Dict[str, Any]]
        :return: the loss and auxiliary information
        :rtype: Tuple[chex.Array, Dict]

        """
        return (
            sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)),
            {},
        )

    return weight_decay
