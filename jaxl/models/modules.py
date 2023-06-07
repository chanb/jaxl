from flax import linen as nn
from typing import Callable, Sequence

import chex


class MLPModule(nn.Module):
    """Multilayer Perceptron."""

    # The number of hidden units in each hidden layer.
    layers: Sequence[int]
    activation: Callable

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        for layer in self.layers[:-1]:
            x = self.activation(nn.Dense(layer)(x))
        x = nn.Dense(self.layers[-1])(x)
        return x
