from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Iterator, Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np
import random

from jaxl.constants import *


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)


def to_jnp(*args: Iterable) -> Iterable[chex.Array]:
    return [jax.device_put(arg) for arg in args]


def batch_flatten(*args: Iterable) -> Iterable[chex.Array]:
    return [arg.reshape((len(arg), -1)) for arg in args]


def parse_dict(d: Dict) -> SimpleNamespace:
    # Reference: https://stackoverflow.com/questions/66208077/how-to-convert-a-nested-python-dictionary-into-a-simple-namespace
    x = SimpleNamespace()
    _ = [
        setattr(x, k, parse_dict(v)) if isinstance(v, dict) else setattr(x, k, v)
        for k, v in d.items()
    ]
    return x


def flatten_dict(p: Dict, label: str = None) -> Iterator:
    if isinstance(p, dict):
        for k, v in p.items():
            yield from flatten_dict(v, k if label is None else f"{label}.{k}")
    else:
        yield (label, p)


def get_reduction(reduction: str) -> Callable[..., chex.Array]:
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


def l2_norm(params: Any):
    return sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))


class DummySummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass


class RunningMeanStd:
    """Modified from Baseline
    Assumes shape to be (number of inputs, input_shape)
    """

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

    def update(self, x: np.ndarray):
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
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ):
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

    def normalize(self, x: np.ndarray) -> np.ndarray:
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

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        x_shape = x.shape
        x = x.reshape(-1, *self.shape)
        return (x * np.sqrt(self.var + self.epsilon) + self.mean).reshape(x_shape)
