from abc import ABC
from flax import linen as nn
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import chex
import jax
import jax.random as jrandom
import numpy as np
import optax

from jaxl.constants import *
from jaxl.models.modules import MLPModule, CNNModule


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
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
        ],
        Tuple[chex.Array, chex.Array],
    ]

    #: Initialize model parameters.
    init: Callable[
        [
            jrandom.PRNGKey,
            chex.Array,
        ],
        Union[optax.Params, Dict[str, Any]],
    ]

    def reset_h_state(self) -> chex.Array:
        """
        Resets hidden state.

        :return: a hidden state
        :rtype: chex.Array
        """
        return np.zeros((1,), dtype=np.float32)


class EncoderPredictorModel(Model):
    """
    Model with two components: encoder and prediction function.
    The encoder can be seen as a model that embeds observation into a representation space.
    The prediction function makes prediction after taking an input from the representation space.
    """

    #: Encode input to representation space.
    encode: Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
        ],
        Tuple[chex.Array, chex.Array],
    ]

    def __init__(
        self,
        encoder: Model,
        predictor: Model,
    ) -> None:
        self.encoder = encoder
        self.predictor = predictor
        self.forward = jax.jit(self.make_forward())
        self.encode = jax.jit(self.make_encode())

    def init(self, model_key: jrandom.PRNGKey, dummy_x: chex.Array) -> Dict[str, Any]:
        """
        Initialize model parameters.

        :param model_key: the random number generation key for initializing parameters
        :param dummy_x: the input data
        :type model_key: jrandom.PRNGKey
        :type dummy_x: chex.Array
        :return: the initialized parameters for both the encoder and the predictor
        :rtype: Dict[str, Any]

        """
        encoder_key, predictor_key = jrandom.split(model_key)
        encoder_params = self.encoder.init(encoder_key, dummy_x)
        dummy_carry = self.encoder.reset_h_state()
        dummy_repr, _ = self.encoder.forward(encoder_params, dummy_x, dummy_carry)
        predictor_params = self.predictor.init(predictor_key, dummy_repr)
        return {
            CONST_ENCODER: encoder_params,
            CONST_PREDICTOR: predictor_params,
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
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the forward call of the encoder-predictor model.

        :return: the forward call.
        :rtype: Callable[
            [
                Union[optax.Params, Dict[str, Any]],
                chex.Array,
                chex.Array,
            ],
            Tuple[chex.Array, chex.Array],
        ]
        """

        def forward(
            params: Union[optax.Params, Dict[str, Any]],
            input: chex.Array,
            carry: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Forward call of the encoder-predictor.

            :param params: the model parameters
            :param input: the input
            :param carry: the hidden state (not used)
            :type params: Union[optax.Params
            :type input: chex.Array
            :type carry: chex.Array
            :return: the output and a pass-through carry
            :rtype: Tuple[chex.Array, chex.Array]

            """
            repr, _ = self.encoder.forward(
                params[CONST_ENCODER],
                input,
                carry,
            )
            pred, pred_carry = self.predictor.forward(
                params[CONST_PREDICTOR],
                repr,
                carry,
            )
            carry = pred_carry

            return pred, carry

        return forward

    def make_encode(
        self,
    ) -> Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the forward call of the encoder model.

        :return: the forward call.
        :rtype: Callable[
            [
                Union[optax.Params, Dict[str, Any]],
                chex.Array,
                chex.Array,
            ],
            Tuple[chex.Array, chex.Array],
        ]
        """

        def encode(
            params: Union[optax.Params, Dict[str, Any]],
            input: chex.Array,
            carry: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Forward call of the encoder.

            :param params: the model parameters
            :param input: the input
            :param carry: the hidden state (not used)
            :type params: Union[optax.Params
            :type input: chex.Array
            :type carry: chex.Array
            :return: the encoded input and next carry
            :rtype: Tuple[chex.Array, chex.Array]

            """
            repr, repr_carry = self.encoder.forward(
                params,
                input,
                carry,
            )
            return repr, repr_carry

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
        self.forward = jax.jit(self.make_forward(vmap_all))

    def init(
        self, model_key: jrandom.PRNGKey, dummy_x: chex.Array
    ) -> Union[optax.Params, Dict[str, Any]]:
        """
        Initialize model parameters.

        :param model_key: the random number generation key for initializing parameters
        :param dummy_x: the input data
        :type model_key: jrandom.PRNGKey
        :type dummy_x: chex.Array
        :return: the initialized parameters for all of the models
        :rtype: Union[optax.Params, Dict[str, Any]]

        """
        model_keys = jrandom.split(model_key, num=self.num_models)
        model_params = jax.vmap(self.model.init)(
            model_keys,
            np.tile(
                dummy_x, (self.num_models, *([1 for _ in range(dummy_x.ndim - 1)]))
            ),
        )
        return model_params

    def make_forward(
        self,
        vmap_all: bool,
    ) -> Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the forward call of the ensemble model.

        :param vmap_all: whether or not to vmap all inputs
        :type vmap_all: bool
        :return: the forward call
        :rtype: Callable[
            [
                Union[optax.Params, Dict[str, Any]],
                chex.Array,
                chex.Array,
            ],
            Tuple[chex.Array, chex.Array],
        ]
        """

        in_axes = [0, 0, 0] if vmap_all else [0, None, None]

        def forward(
            params: Union[optax.Params, Dict[str, Any]],
            input: chex.Array,
            carry: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Forward call of the ensemble.

            :param params: the model parameters
            :param input: the input
            :param carry: the hidden state (not used)
            :type params: Union[optax.Params
            :type input: chex.Array
            :type carry: chex.Array
            :return: the output and a pass-through carry
            :rtype: Tuple[chex.Array, chex.Array]

            """
            pred, carry = jax.vmap(self.model.forward, in_axes=in_axes)(
                params, input, carry
            )
            return pred, carry

        return forward

    def reset_h_state(self) -> chex.Array:
        """
        Resets hidden state.

        :return: a hidden state
        :rtype: chex.Array
        """
        return np.zeros((self.num_models, 1), dtype=np.float32)


class MLP(Model):
    """A multilayer perceptron."""

    def __init__(
        self,
        layers: Sequence[int],
        activation: str = CONST_RELU,
        output_activation: str = CONST_IDENTITY,
    ) -> None:
        self.model = MLPModule(
            layers, get_activation(activation), get_activation(output_activation)
        )
        self.forward = jax.jit(self.make_forward())

    def init(
        self, model_key: jrandom.PRNGKey, dummy_x: chex.Array
    ) -> Union[optax.Params, Dict[str, Any]]:
        """
        Initialize model parameters.

        :param model_key: the random number generation key for initializing parameters
        :param dummy_x: the input data
        :type model_key: jrandom.PRNGKey
        :type dummy_x: chex.Array
        :return: the initialized parameters
        :rtype: Union[optax.Params, Dict[str, Any]]

        """
        return self.model.init(model_key, dummy_x)

    def make_forward(
        self,
    ) -> Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the forward call of the MLP model.

        :return: the forward call.
        :rtype: Callable[
            [
                Union[optax.Params, Dict[str, Any]],
                chex.Array,
                chex.Array,
            ],
            Tuple[chex.Array, chex.Array],
        ]
        """

        def forward(
            params: Union[optax.Params, Dict[str, Any]],
            input: chex.Array,
            carry: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Forward call of the MLP.

            :param params: the model parameters
            :param input: the input
            :param carry: the hidden state (not used)
            :type params: Union[optax.Params
            :type input: chex.Array
            :type carry: chex.Array
            :return: the output and a pass-through carry
            :rtype: Tuple[chex.Array, chex.Array]

            """
            # NOTE: Assume batch size is first dim
            input = input.reshape((input.shape[0], -1))
            return self.model.apply(params, input), carry

        return forward
    

class CNN(Model):
    """A convolutional net."""

    def __init__(
        self,
        features: Sequence[int],
        kernel_sizes: Sequence[int],
        layers: Sequence[int],
        activation: str = CONST_RELU,
        output_activation: str = CONST_IDENTITY,
    ) -> None:
        self.conv = CNNModule(
            features,
            kernel_sizes,
            get_activation(activation),
        )
        self.mlp = MLPModule(
            layers,
            get_activation(activation),
            get_activation(output_activation),
        )
        self.forward = jax.jit(self.make_forward())

    def init(
        self, model_key: jrandom.PRNGKey, dummy_x: chex.Array
    ) -> Union[optax.Params, Dict[str, Any]]:
        """
        Initialize model parameters.

        :param model_key: the random number generation key for initializing parameters
        :param dummy_x: the input data
        :type model_key: jrandom.PRNGKey
        :type dummy_x: chex.Array
        :return: the initialized parameters
        :rtype: Union[optax.Params, Dict[str, Any]]

        """
        conv_params = self.conv.init(model_key, dummy_x)
        dummy_latent = self.conv.apply(conv_params, dummy_x)
        mlp_params = self.mlp.init(
            model_key,
            dummy_latent.reshape((dummy_latent.shape[0], -1))
        )
        return {
            CONST_CNN: conv_params,
            CONST_MLP: mlp_params,
        }

    def make_forward(
        self,
    ) -> Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the forward call of the CNN model.

        :return: the forward call.
        :rtype: Callable[
            [
                Union[optax.Params, Dict[str, Any]],
                chex.Array,
                chex.Array,
            ],
            Tuple[chex.Array, chex.Array],
        ]
        """

        def forward(
            params: Union[optax.Params, Dict[str, Any]],
            input: chex.Array,
            carry: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Forward call of the CNN.

            :param params: the model parameters
            :param input: the input
            :param carry: the hidden state (not used)
            :type params: Union[optax.Params
            :type input: chex.Array
            :type carry: chex.Array
            :return: the output and a pass-through carry
            :rtype: Tuple[chex.Array, chex.Array]

            """
            # NOTE: Assume batch size is first dim
            conv_latent = self.conv.apply(params[CONST_CNN], input)
            conv_latent = conv_latent.reshape((conv_latent.shape[0], -1))
            out = self.mlp.apply(params[CONST_MLP], conv_latent)
            return out, carry

        return forward


class Policy(ABC):
    """Abstract policy class."""

    #: Compute action for interacting with the environment.
    compute_action: Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]

    #: Compute deterministic action.
    deterministic_action: Callable[
        [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array],
        Tuple[chex.Array, chex.Array],
    ]

    def __init__(self, model: Model) -> None:
        self.reset = jax.jit(self.make_reset(model))

    def make_reset(self, model: Model) -> Callable[..., chex.Array]:
        """
        Makes the function that resets the policy.
        This is often used for resetting the hidden state.

        :param model: the model
        :type model: Model
        :return: a function for initializing the hidden state
        :rtype: chex.Array
        """

        def _reset() -> chex.Array:
            """
            Resets hidden state.

            :return: a hidden state
            :rtype: chex.Array
            """
            return model.reset_h_state()

        return _reset


class StochasticPolicy(Policy):
    """Abstract stochastic policy class that extends ``Policy``."""

    #: Compute random action.
    random_action: Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]

    # . Compute action and its log probability.
    act_lprob: Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array, chex.Array],
    ]

    # . Compute action log probability.
    lprob: Callable[
        [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
        chex.Array,
    ]
