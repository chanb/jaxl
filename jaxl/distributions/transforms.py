from abc import abstractstaticmethod, ABC

import chex
import jax
import math


class Transform(ABC):
    @abstractstaticmethod
    def transform(x: chex.Array) -> chex.Array:
        pass

    @abstractstaticmethod
    def log_abs_det_jacobian(self, x: chex.Array, x_t: chex.Array) -> chex.Array:
        pass


class TanhTransform(Transform):
    def transform(x: chex.Array) -> chex.Array:
        return jax.nn.tanh(x)

    def log_abs_det_jacobian(x: chex.Array, x_t: chex.Array) -> chex.Array:
        return 2.0 * (math.log(2.0) - x - jax.nn.softplus(-2.0 * x))
