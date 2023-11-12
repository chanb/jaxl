from flax import linen as nn
from typing import Callable, Sequence

import chex


class MLPModule(nn.Module):
    """Multilayer Perceptron."""

    # The number of hidden units in each hidden layer.
    layers: Sequence[int]
    activation: Callable
    output_activation: Callable

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        for layer in self.layers[:-1]:
            x = self.activation(nn.Dense(layer)(x))
        x = self.output_activation(nn.Dense(self.layers[-1])(x))
        return x


class CNNModule(nn.Module):
    """Convolutional layer."""

    # The number of hidden units in each hidden layer.
    features: Sequence[int]
    kernel_sizes: Sequence[Sequence[int]]
    activation: Callable
    output_activation: Callable

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        for feature, kernel_size in zip(self.features[:-1], self.kernel_sizes[:-1]):
            x = self.activation(nn.Conv(feature, kernel_size)(x))
        x = self.output_activation(
            nn.Dense(self.features[-1], self.kernel_sizes[-1])(x)
        )
        return x
