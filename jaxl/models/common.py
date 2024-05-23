from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import partial
from flax import linen as nn
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

from jaxl.constants import *
from jaxl.models.modules import MLPModule, CNNModule, ResNetV1Module


def get_activation(activation: str) -> Callable:
    """
    Gets an activation function

    :param activation: the activation function name
    :type activation: str
    :return: an activation function
    :rtype: Callable

    """
    assert (
        activation in VALID_ACTIVATION
    ), f"{activation} is not supported (one of {VALID_ACTIVATION})"

    if activation == CONST_IDENTITY:

        def identity(x: Any) -> Any:
            return x

        return identity
    elif activation == CONST_RELU:
        return nn.relu
    elif activation == CONST_TANH:
        return nn.tanh
    else:
        raise NotImplementedError


class Model(ABC):
    """Abstract model class."""

    #: Model forward call.
    forward: Callable[
        [
            Dict[str, Any],
            chex.Array,
            chex.Array,
            bool,
        ],
        Tuple[chex.Array, chex.Array, Any],
    ]

    #: Initialize model train states.
    init: Callable[
        [
            jrandom.PRNGKey,
            chex.Array,
        ],
        Dict[str, Any],
    ]

    @abstractmethod
    def make_forward(self) -> Callable[
        [
            Dict[str, Any],
            chex.Array,
            chex.Array,
            bool,
        ],
        Tuple[chex.Array, chex.Array, Any],
    ]:
        """
        Creates the forward call of the model
        """
        pass

    def reset_h_state(self) -> chex.Array:
        """
        Resets hidden state.

        :return: a hidden state
        :rtype: chex.Array
        """
        return np.zeros((1,), dtype=np.float32)

    def update_batch_stats(
        self, params: Dict[str, Any], batch_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        if CONST_BATCH_STATS in batch_stats:
            params[CONST_BATCH_STATS] = batch_stats[CONST_BATCH_STATS]
        return params

    def update_random_keys(self, random_keys: Dict[str, Any]) -> Dict[str, Any]:
        return random_keys


class EncoderPredictorModel(Model):
    """
    Model with two components: encoder and prediction function.
    The encoder can be seen as a model that embeds observation into a representation space.
    The prediction function makes prediction after taking an input from the representation space.
    """

    #: Encode input to representation space.
    encode: Callable[
        [
            Dict[str, Any],
            chex.Array,
            chex.Array,
            bool,
        ],
        Tuple[chex.Array, chex.Array, Any],
    ]

    def __init__(
        self,
        encoder: Model,
        predictor: Model,
    ) -> None:
        self.encoder = encoder
        self.predictor = predictor
        self.forward = jax.jit(self.make_forward(), static_argnames=[CONST_EVAL])
        self.encode = jax.jit(self.make_encode(), static_argnames=[CONST_EVAL])

    def init(self, model_key: jrandom.PRNGKey, dummy_x: chex.Array) -> Dict[str, Any]:
        encoder_key, predictor_key = jrandom.split(model_key)
        encoder_train_state = self.encoder.init(encoder_key, dummy_x)
        dummy_carry = self.encoder.reset_h_state()
        dummy_repr, _, _ = self.encoder.forward(
            encoder_train_state[CONST_PARAMS], dummy_x, dummy_carry
        )
        predictor_train_state = self.predictor.init(predictor_key, dummy_repr)
        return {
            CONST_ENCODER: encoder_train_state,
            CONST_PREDICTOR: predictor_train_state,
        }

    def reset_h_state(self) -> chex.Array:
        """
        Resets hidden state.

        :return: a hidden state
        :rtype: chex.Array
        """
        return self.encoder.reset_h_state()

    def make_forward(
        self,
    ) -> Callable[
        [
            Dict[str, Any],
            chex.Array,
            chex.Array,
            bool,
        ],
        Tuple[chex.Array, chex.Array, Any],
    ]:
        def forward(
            train_states: Dict[str, Any],
            input: chex.Array,
            carry: chex.Array,
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array, Any]:
            repr, _, enc_updates = self.encoder.forward(
                train_states[CONST_ENCODER],
                input,
                carry,
                eval,
                **kwargs,
            )
            pred, pred_carry, pred_updates = self.predictor.forward(
                train_states[CONST_PREDICTOR],
                repr,
                carry,
                eval,
                **kwargs,
            )
            carry = pred_carry

            return (
                pred,
                carry,
                {
                    CONST_ENCODER: enc_updates,
                    CONST_PREDICTOR: pred_updates,
                },
            )

        return forward

    def make_encode(
        self,
    ) -> Callable[
        [
            Dict[str, Any],
            chex.Array,
            chex.Array,
            bool,
        ],
        Tuple[chex.Array, chex.Array, Any],
    ]:
        def encode(
            train_states: Dict[str, Any],
            input: chex.Array,
            carry: chex.Array,
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array, Any]:
            repr, repr_carry, repr_updates = self.encoder.forward(
                train_states[CONST_ENCODER],
                input,
                carry,
                eval,
                **kwargs,
            )
            return repr, repr_carry, repr_updates

        return encode


class EnsembleModel(Model):
    """
    Ensemble Model.
    We assume all models are identical.
    """

    #: Number of models in the ensemble.
    num_models: int

    def __init__(self, model: Model, num_models: int, vmap_all: bool = True) -> None:
        self.model = model
        self.num_models = num_models
        self.forward = jax.jit(
            self.make_forward(vmap_all), static_argnames=[CONST_EVAL]
        )

    def init(self, model_key: jrandom.PRNGKey, dummy_x: chex.Array) -> Dict[str, Any]:
        model_keys = jrandom.split(model_key, num=self.num_models)
        train_states = jax.vmap(self.model.init)(
            model_keys,
            np.tile(
                dummy_x, (self.num_models, *([1 for _ in range(dummy_x.ndim - 1)]))
            ),
        )
        return train_states

    def make_forward(
        self,
        vmap_all: bool,
    ) -> Callable[
        [
            Dict[str, Any],
            chex.Array,
            chex.Array,
            bool,
        ],
        Tuple[chex.Array, chex.Array, Any],
    ]:
        in_axes = [0, 0, 0, None] if vmap_all else [0, None, None, None]

        def forward(
            train_states: Dict[str, Any],
            input: chex.Array,
            carry: chex.Array,
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array, Any]:
            pred, carry, updates = jax.vmap(
                partial(self.model.forward, **kwargs),
                in_axes=in_axes,
            )(
                train_states,
                input,
                carry,
                eval,
            )
            return pred, carry, updates

        return forward

    def reset_h_state(self) -> chex.Array:
        return np.zeros((self.num_models, 1), dtype=np.float32)

    def update_batch_stats(
        self, params: Dict[str, Any], batch_stats: Any
    ) -> Dict[str, Any]:
        return self.model.update_batch_stats(params, batch_stats)


class MLP(Model):
    """A multilayer perceptron."""

    def __init__(
        self,
        layers: Sequence[int],
        activation: str = CONST_RELU,
        output_activation: str = CONST_IDENTITY,
        use_batch_norm: bool = False,
        use_bias: bool = True,
        flatten: bool = False,
    ) -> None:
        self.model = MLPModule(
            layers,
            get_activation(activation),
            get_activation(output_activation),
            use_batch_norm,
            use_bias,
            flatten,
        )
        self.forward = jax.jit(self.make_forward(), static_argnames=[CONST_EVAL])

    def init(self, model_key: jrandom.PRNGKey, dummy_x: chex.Array) -> Dict[str, Any]:
        params = self.model.init(model_key, dummy_x, eval=True)
        return {
            CONST_PARAMS: params,
            CONST_RANDOM_KEYS: {},
        }

    def make_forward(
        self,
    ) -> Callable[
        [
            Dict[str, Any],
            chex.Array,
            chex.Array,
            bool,
        ],
        Tuple[chex.Array, chex.Array, Any],
    ]:
        def forward(
            train_state: Dict[str, Any],
            input: chex.Array,
            carry: chex.Array,
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array, Any]:
            # NOTE: Assume batch size is first dim
            (out, updates) = self.model.apply(
                train_state[CONST_PARAMS],
                input,
                eval,
                mutable=[CONST_BATCH_STATS],
                rngs=train_state[CONST_RANDOM_KEYS],
            )
            return out, carry, updates

        return forward


class CNN(Model):
    """A convolutional network."""

    def __init__(
        self,
        features: Sequence[int],
        kernel_sizes: Sequence[int],
        layers: Sequence[int],
        activation: str = CONST_RELU,
        output_activation: str = CONST_IDENTITY,
        use_batch_norm: bool = False,
    ) -> None:
        if isinstance(features[0], Iterable):
            self.spatial_dim = -(len(features[0]) + 1)
        else:
            self.spatial_dim = 1
        self.conv = CNNModule(
            features, kernel_sizes, get_activation(activation), use_batch_norm
        )
        self.mlp = MLPModule(
            layers,
            get_activation(activation),
            get_activation(output_activation),
            False,
            True,
        )
        self.forward = jax.jit(self.make_forward(), static_argnames=[CONST_EVAL])

    def init(self, model_key: jrandom.PRNGKey, dummy_x: chex.Array) -> Dict[str, Any]:
        conv_params = self.conv.init(model_key, dummy_x, eval=True)
        dummy_latent, conv_batch_stats = self.conv.apply(
            conv_params, dummy_x, eval=True, mutable=[CONST_BATCH_STATS]
        )
        mlp_params = self.mlp.init(
            model_key,
            dummy_latent.reshape((*dummy_latent.shape[: self.spatial_dim], -1)),
            eval=True,
        )
        return {
            CONST_PARAMS: {
                CONST_CNN: conv_params,
                CONST_MLP: mlp_params,
            },
            CONST_RANDOM_KEYS: {
                CONST_CNN: {},
                CONST_MLP: {},
            },
        }

    def make_forward(
        self,
    ) -> Callable[
        [
            Dict[str, Any],
            chex.Array,
            chex.Array,
            bool,
        ],
        Tuple[chex.Array, chex.Array, Any],
    ]:
        def forward(
            train_states: Dict[str, Any],
            input: chex.Array,
            carry: chex.Array,
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array, Any]:
            # NOTE: Assume batch size is first dim
            (conv_latent, conv_updates) = self.conv.apply(
                train_states[CONST_PARAMS][CONST_CNN],
                input,
                eval,
                mutable=[CONST_BATCH_STATS],
            )
            conv_latent = conv_latent.reshape(
                (*conv_latent.shape[: self.spatial_dim], -1)
            )
            (out, mlp_updates) = self.mlp.apply(
                train_states[CONST_PARAMS][CONST_MLP],
                conv_latent,
                eval,
                mutable=[CONST_BATCH_STATS],
            )
            return (
                out,
                carry,
                {
                    CONST_CNN: conv_updates,
                    CONST_MLP: mlp_updates,
                },
            )

        return forward

    def update_batch_stats(
        self, params: Dict[str, Any], batch_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        if CONST_BATCH_STATS in batch_stats[CONST_CNN]:
            params[CONST_CNN][CONST_BATCH_STATS] = batch_stats[CONST_CNN][
                CONST_BATCH_STATS
            ]
        return params


class ResNetV1(Model):
    """A residual network."""

    def __init__(
        self,
        blocks_per_group: Sequence[int],
        features: Sequence[int],
        stride: Sequence[Sequence[int]],
        use_projection: Sequence[bool],
        use_bottleneck: bool,
        use_batch_norm: bool = True,
    ) -> None:
        self.resnet = ResNetV1Module(
            blocks_per_group=blocks_per_group,
            features=features,
            stride=stride,
            use_projection=use_projection,
            use_bottleneck=use_bottleneck,
            use_batch_norm=use_batch_norm,
        )
        self.forward = jax.jit(self.make_forward(), static_argnames=[CONST_EVAL])

    def init(self, model_key: jrandom.PRNGKey, dummy_x: chex.Array) -> Dict[str, Any]:
        resnet_params = self.resnet.init(model_key, dummy_x, eval=True)
        return {
            CONST_PARAMS: resnet_params,
            CONST_RANDOM_KEYS: {},
        }

    def make_forward(
        self,
    ) -> Callable[
        [
            Dict[str, Any],
            chex.Array,
            chex.Array,
            bool,
        ],
        Tuple[chex.Array, chex.Array, Any],
    ]:
        def forward(
            train_states: Dict[str, Any],
            input: chex.Array,
            carry: chex.Array,
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array, Any]:
            # NOTE: Assume batch size is first dim
            (out, updates) = self.resnet.apply(
                train_states[CONST_PARAMS],
                input,
                eval,
                mutable=[CONST_BATCH_STATS],
            )
            return out, carry, updates

        return forward
