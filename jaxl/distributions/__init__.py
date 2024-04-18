import jaxl.constants.distributions as c

from jaxl.distributions.distributions import *

import chex
import jax

from typing import Callable


def get_transform(transform_name: str) -> Callable[[chex.Array], chex.Array]:
    """
    Gets a transform

    :param transform_name: the transform name
    :type transform_name: str
    :return: a transform
    :rtype: Callable[chex.Array, chex.Array]

    """
    assert (
        transform_name in c.VALID_TRANSFORM
    ), f"{transform_name} is not supported (one of {c.VALID_TRANSFORM})"

    if transform_name == c.CONST_SOFTPLUS:
        return jax.nn.softplus
    elif transform_name == c.CONST_SQUAREPLUS:
        return jax.nn.squareplus
    elif transform_name == c.CONST_SIGMOID:
        return jax.nn.sigmoid
    else:
        raise NotImplementedError
