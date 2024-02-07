from flax import linen as nn
from typing import Dict, Any, Sequence

import chex
import math
import numpy as np


class NoEncoding(nn.Module):
    """No encoding."""

    @nn.compact
    def __call__(self, x: Any):
        return x


class PositionalEncoding(nn.Module):
    """
    Default positional encoding used in Transformers.
    Reference: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html
    """

    embed_dim: int
    max_len: int

    def setup(self):
        pe = np.zeros((self.max_len, self.embed_dim))
        position = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, self.embed_dim, 2) * (-math.log(10000.0) / self.embed_dim)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe[None]

    def __call__(self, x: chex.Array):
        x = x + self.pe[:, : x.shape[1]]
        return x


class ConcatenateInputsEncoding(nn.Module):
    """
    Concatenates all the values in the PyTree.
    """

    input_dims: Dict[str, Sequence[int]]

    @nn.compact
    def __call__(self, x: Dict[str, chex.Array]):
        return np.concatenate([
            x[key].reshape((*x[key].shape[:-len(input_dim)], -1))for key, input_dim in self.input_dims.items()
        ])
