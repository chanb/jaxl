from flax import linen as nn
from typing import Sequence

import chex


class MLPModule(nn.Module):
    """Multilayer Perceptron."""

    # The number of hidden units in each hidden layer.
    layers: Sequence[int]

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        for layer in self.layers[:-1]:
            x = nn.relu(nn.Dense(layer)(x))
        x = nn.Dense(self.layers[-1])(x)
        return x
