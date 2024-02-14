from flax import linen as nn
from types import SimpleNamespace
from typing import Dict, Any, Sequence, Tuple

import chex
import jax.numpy as jnp
import math
import numpy as np

from jaxl.constants import *


class NoEncoding(nn.Module):
    """No encoding."""

    @nn.compact
    def __call__(self, x: Any, **kwargs):
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

    def __call__(self, x: chex.Array, **kwargs):
        x = x + self.pe[:, : x.shape[1]]
        return x


def get_positional_encoding(encoding: SimpleNamespace) -> nn.Module:
    """
    Gets a positional encoding

    :param encoding: the positional encoding configuration
    :type encoding: SimpleNamespace
    :return: a positional encoding
    :rtype: nn.Module

    """
    assert (
        encoding.type in VALID_POSITIONAL_ENCODING
    ), f"{encoding.type} is not supported (one of {VALID_POSITIONAL_ENCODING})"
    if encoding.type == CONST_NO_ENCODING:
        return NoEncoding()
    elif encoding.type == CONST_DEFAULT_ENCODING:
        return PositionalEncoding(
            encoding.kwargs.embed_dim,
            encoding.kwargs.max_len,
        )
    else:
        raise NotImplementedError


class ConcatenateInputsEncoding(nn.Module):
    """
    Concatenates all the values in the PyTree.
    """

    input_dims: Dict[str, Sequence[int]]

    @nn.compact
    def __call__(self, x: Dict[str, chex.Array], **kwargs):
        return jnp.concatenate(
            [
                x[key].reshape((*x[key].shape[: -len(input_dim)], -1))
                for key, input_dim in self.input_dims.items()
            ],
            axis=-1,
        )


def get_state_action_encoding(
    obs_dim: chex.Array, act_dim: chex.Array, encoding: SimpleNamespace
) -> nn.Module:
    """
    Gets a state-action pair encoding

    :param obs_dim: the observation dimensionality
    :param act_dim: the action dimensionality
    :param encoding: the state-action pair configuration
    :type obs_dim: chex.Array
    :type act_dim: chex.Array
    :type encoding: SimpleNamespace
    :return: a state-action pair encoding
    :rtype: nn.Module

    """
    assert (
        encoding.type in VALID_STATE_ACTION_ENCODING
    ), f"{encoding.type} is not supported (one of {VALID_STATE_ACTION_ENCODING})"
    if encoding.type == CONST_CONCATENATE_INPUTS_ENCODING:
        return ConcatenateInputsEncoding(
            {
                CONST_OBSERVATION: obs_dim,
                CONST_ACTION: act_dim,
            }
        )
    else:
        raise NotImplementedError
