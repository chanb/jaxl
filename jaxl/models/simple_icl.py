from flax import linen as nn
from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

from jaxl.constants import *
from jaxl.models.common import (
    Model,
    CNN,
    MLP,
    ResNetV1,
)
from jaxl.models.encodings import get_positional_encoding
from jaxl.models.modules import GPTModule


def make_h(similarity: str):
    if similarity == "gaussian":

        def h_fn(context_inputs, queries):
            return jnp.exp(
                -jnp.sum((context_inputs - queries) ** 2, axis=-1, keepdims=True)
            )

        return h_fn
    else:
        raise NotImplementedError


def make_g(ground_truth_prob: float):
    def g_fn(queries, outputs):
        return jnp.clip(
            jnp.full_like(
                outputs, fill_value=((1 - ground_truth_prob) / (outputs.shape[-1] - 1))
            )
            + outputs,
            a_min=0.0,
            a_max=ground_truth_prob,
        )

    return g_fn


class SimpleICLModel(Model):
    def __init__(
        self,
        ground_truth_prob: float,
        similarity: str,
    ):
        self.alpha = nn.Dense(2)
        self.h_fn = make_h(similarity)
        self.g_fn = make_g(ground_truth_prob)
        self.forward = jax.jit(self.make_forward())

    def init(self, model_key, dummy_input, dummy_output):
        return {"alpha": self.alpha.init(model_key, dummy_input)}

    def make_forward(self):
        def forward(
            params,
            queries,
            contexts,
            eval=False,
            **kwargs,
        ):
            alphas = self.alpha.apply(params["alpha"], queries)
            p_iwl = jax.nn.softmax(alphas, axis=2)[:, 0]
            similarity = self.h_fn(contexts[CONST_CONTEXT_INPUT], queries)
            icl_pred = jnp.sum(
                jax.nn.softmax(similarity, axis=1) * contexts[CONST_CONTEXT_OUTPUT],
                axis=1,
            )
            iwl_pred = self.g_fn(queries, contexts["output"])

            return (1 - p_iwl) * icl_pred + p_iwl * iwl_pred, None, {}

        return forward
