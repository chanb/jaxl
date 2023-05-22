from abc import abstractstaticmethod
from typing import Optional

import chex
import jax.numpy as jnp
import jax.random as jrandom
import math


class Distribution:
    @abstractstaticmethod
    def sample(
        *, key: jrandom.PRNGKey, num_samples: Optional[int] = None
    ) -> chex.Array:
        raise NotImplementedError

    @abstractstaticmethod
    def lprob(*, x: chex.Array) -> chex.Array:
        raise NotImplementedError


class Normal(Distribution):
    @staticmethod
    def sample(
        mean: chex.Array,
        std: chex.Array,
        key: jrandom.PRNGKey,
        num_samples: Optional[int] = None,
    ) -> chex.Array:
        if num_samples:
            shape = (num_samples, *mean.shape)
        else:
            shape = mean.shape
        return mean + jrandom.normal(key=key, shape=shape) * std

    def lprob(mean: chex.Array, std: chex.Array, x: chex.Array) -> chex.Array:
        var = std**2
        log_std = jnp.log(std)
        return (
            -((x - mean) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))
        )
