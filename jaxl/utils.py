from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Iterator, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import random

from jaxl.constants import *


def set_seed(seed: int = 0):
    """
    Sets the random number generators' seed.

    :param seed: the seed
    :type seed: int:  (Default value = 0)

    """
    random.seed(seed)
    np.random.seed(seed)


def to_jnp(*args: Iterable) -> Iterable[chex.Array]:
    """
    Convert iterables to JAX NumPy arrays.

    :param *args: an iterable of items
    :type *args: Iterable
    :return: JAX NumPy array variant of the items
    :rtype: Iterable[chex.Array]

    """
    return [jax.device_put(arg) for arg in args]


def batch_flatten(*args: Iterable) -> Iterable[chex.Array]:
    """
    Flatten the batch of items in the iterable.

    :param *args: an iterable of items
    :type *args: Iterable
    :return: Flattened items
    :rtype: Iterable[chex.Array]

    """
    return [arg.reshape((len(arg), -1)) for arg in args]


def parse_dict(d: Dict) -> SimpleNamespace:
    """
    Parse dictionary into a namespace.
    Reference: https://stackoverflow.com/questions/66208077/how-to-convert-a-nested-python-dictionary-into-a-simple-namespace

    :param d: the dictionary
    :type d: Dict
    :return: the namespace version of the dictionary's content
    :rtype: SimpleNamespace

    """
    x = SimpleNamespace()
    _ = [
        setattr(x, k, parse_dict(v)) if isinstance(v, dict) else setattr(x, k, v)
        for k, v in d.items()
    ]
    return x


def flatten_dict(d: Dict, label: str = None) -> Iterator:
    """
    Flattens a dictionary.

    :param d: the dictionary
    :param label: the parent's key name
    :type d: Dict
    :type label: str:  (Default value = None)
    :return: an iterator that yields the key-value pairs
    :rtype: Iterator

    """
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flatten_dict(v, k if label is None else f"{label}.{k}")
    else:
        yield (label, d)


def set_dict_value(d: Dict, key: str, val: Any) -> None:
    """
    Set dictionary value by key (mutation).

    :param d: the dictionary
    :param key: the key to look for
    :param value: the value to set
    :type d: Dict
    :type key: str
    :type value: Any

    """
    if key in d:
        d[key] = val

    for k in d:
        if isinstance(d[k], dict):
            set_dict_value(d[k], key, val)


def get_dict_value(d: Dict, key: str) -> Tuple[bool, Any]:
    """
    Get first occurence dictionary value by key.

    :param d: the dictionary
    :param key: the key to look for
    :type d: Dict
    :type key: str
    :return: whether a value is retrieved, if so return the value based on the dictionary key as well
    :rtype: Tuple[bool, Any]

    """
    if key in d:
        return (d[key], True)

    for k in d:
        if isinstance(d[k], dict):
            val = get_dict_value(d[k], key)
            if val[1]:
                return val

    return (None, False)


def get_reduction(reduction: str) -> Callable[..., chex.Array]:
    """
    Gets a reduction function.

    :param reduction: the reduction function name
    :type reduction: str

    """
    assert (
        reduction in VALID_REDUCTION
    ), f"{reduction} is not supported (one of {VALID_REDUCTION})"
    reduction_func = lambda x: x
    if reduction == CONST_MEAN:
        reduction_func = jnp.mean
    elif reduction == CONST_SUM:
        reduction_func = jnp.sum
    else:
        raise NotImplementedError
    return reduction_func


def l2_norm(params: chex.PyTreeDef) -> chex.Array:
    """
    Computes the L2 norm of a complete PyTree.

    :param params: the pytree object with scalars
    :type params: PyTreeDef
    :return: L2 norm of the complete PyTree
    :rtype: chex.Array

    """
    return sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))


def per_leaf_l2_norm(params: chex.PyTreeDef) -> chex.PyTreeDef:
    """
    Computes the L2 norm of each leaf in a PyTree.

    :param params: the pytree object with scalars
    :type params: PyTreeDef
    :return: L2 norm of each leaf in the PyTree
    :rtype: PyTreeDef

    """
    return jax.tree_util.tree_map(lambda p: jnp.sum(p**2), params)


class DummySummaryWriter:
    """
    A fake SummaryWriter class for Tensorboard.
    """

    def add_scalar(self, *args, **kwargs):
        """

        :param *args:
        :param **kwargs:

        """
        pass

    def add_scalars(self, *args, **kwargs):
        """

        :param *args:
        :param **kwargs:

        """
        pass


class RunningMeanStd:
    """
    This keeps track of the running mean and standard deviation.
    Modified from Baseline.
    Assumes shape to be (number of inputs, input_shape).
    """

    #: The running mean
    mean: chex.Array

    #: The running variance
    var: chex.Array

    def __init__(
        self,
        epsilon: float = 1e-4,
        shape: Sequence[int] = (),
        a_min: float = -5.0,
        a_max: float = 5.0,
    ):
        assert epsilon > 0.0
        self.shape = shape
        self.mean = np.zeros(shape, dtype=float)
        self.var = np.ones(shape, dtype=float)
        self.epsilon = epsilon
        self.count = 0
        self.a_min = a_min
        self.a_max = a_max

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the state of the running statistics.

        :return: the running statistics state
        :rtype: Dict[str, Any]

        """
        return {
            CONST_SHAPE: self.shape,
            CONST_MEAN: self.mean,
            CONST_VAR: self.var,
            CONST_EPSILON: self.epsilon,
            CONST_COUNT: self.count,
            CONST_A_MIN: self.a_min,
            CONST_A_MAX: self.a_max,
        }

    def set_state(self, state: Dict[str, Any]):
        """
        Sets the state of the running statistics.

        :param state: the running statistics state
        :type state: Dict[str, Any]

        """
        for k, v in state.items():
            setattr(self, k, v)

    def update(self, x: chex.Array):
        """
        Updates the running statistics.

        :param x: the data
        :type x: chex.Array

        """
        x = x.reshape(-1, *self.shape)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = len(x)

        if batch_count == 0:
            return
        elif batch_count == 1:
            batch_var.fill(0.0)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: chex.Array, batch_var: chex.Array, batch_count: int
    ):
        """
        Updates from the moments

        :param batch_mean: the mean of the current batch
        :param batch_var: the variance of the current batch
        :param batch_count: the amount of data in the current batch
        :type batch_mean: chex.Array
        :type batch_var: chex.Array
        :type batch_count: chex.Array

        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta**2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, x: chex.Array) -> chex.Array:
        """
        Normalizes the data using running statistics.

        :param x: the data
        :type x: chex.Array
        :return: the normalized data
        :rtype: chex.Array

        """
        x_shape = x.shape
        x = x.reshape(-1, *self.shape)
        normalized_x = np.clip(
            (x - self.mean) / np.sqrt(self.var + self.epsilon),
            a_min=self.a_min,
            a_max=self.a_max,
        )
        normalized_x[normalized_x != normalized_x] = 0.0
        normalized_x = normalized_x.reshape(x_shape)
        return normalized_x

    def unnormalize(self, x: chex.Array) -> chex.Array:
        """
        Unnormalizes the data using running statistics.

        :param x: the data
        :type x: chex.Array
        :return: the normalized data
        :rtype: chex.Array

        """
        x_shape = x.shape
        x = x.reshape(-1, *self.shape)
        return (x * np.sqrt(self.var + self.epsilon) + self.mean).reshape(x_shape)
