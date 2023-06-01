from abc import ABC, abstractmethod
from flax import linen as nn
from flax.core.scope import FrozenVariableDict
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import chex
import jax
import jax.random as jrandom
import numpy as np

from jaxl.constants import *
from jaxl.models.modules import MLPModule


class Model(ABC):
    """Abstract model class."""

    #: Model forward call.
    forward: Callable[
        [
            Union[FrozenVariableDict, Dict[str, Any]],
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
        Union[FrozenVariableDict, Dict[str, Any]],
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
            Union[FrozenVariableDict, Dict[str, Any]],
            chex.Array,
            chex.Array,
        ],
        Tuple[chex.Array, chex.Array],
    ]

    def __init__(
        self,
        encoder: Model,
        predictor: Model,
        encoder_name: str = CONST_ENCODER,
        predictor_name: str = CONST_PREDICTOR,
    ) -> None:
        self.encoder_name = encoder_name
        self.predictor_name = predictor_name
        self.encoder = encoder
        self.predictor = predictor
        self._h_state_shapes = (
            self.encoder.reset_h_state().shape,
            self.predictor.reset_h_state().shape,
        )
        self._flattened_h_states_shape = (
            self.encoder.reset_h_state().size,
            self.predictor.reset_h_state().size,
        )
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
            self.encoder_name: encoder_params,
            self.predictor_name: predictor_params,
        }

    def reset_h_state(self) -> chex.Array:
        """
        Resets hidden state.

        :return: a hidden state
        :rtype: chex.Array
        """
        return np.concatenate(
            [
                self.encoder.reset_h_state().reshape(-1),
                self.predictor.reset_h_state().reshape(-1),
            ],
            axis=-1,
        )

    def make_forward(
        self,
    ) -> Callable[
        [
            Union[FrozenVariableDict, Dict[str, Any]],
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
                Union[FrozenVariableDict, Dict[str, Any]],
                chex.Array,
                chex.Array,
            ],
            Tuple[chex.Array, chex.Array],
        ]
        """

        def forward(
            params: Union[FrozenVariableDict, Dict[str, Any]],
            input: chex.Array,
            carry: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Forward call of the encoder-predictor.

            :param params: the model parameters
            :param input: the input
            :param carry: the hidden state (not used)
            :type params: Union[FrozenVariableDict
            :type input: chex.Array
            :type carry: chex.Array
            :return: the output and a pass-through carry
            :rtype: Tuple[chex.Array, chex.Array]

            """
            carry_shape = carry.shape[:-1]
            repr, repr_carry = self.encoder.forward(
                params[self.encoder_name],
                input,
                carry[..., : self._flattened_h_states_shape[0]].reshape(
                    (*carry_shape, *self._h_state_shapes[0])
                ),
            )
            pred, pred_carry = self.predictor.forward(
                params[self.predictor_name],
                repr,
                carry[..., self._flattened_h_states_shape[0] :].reshape(
                    (*carry_shape, *self._h_state_shapes[1])
                ),
            )
            carry = np.concatenate(
                (
                    repr_carry.reshape((*carry_shape, -1)),
                    pred_carry.reshape((*carry_shape, -1)),
                ),
                axis=-1,
            )

            return pred, carry

        return forward

    def make_encode(
        self,
    ) -> Callable[
        [
            Union[FrozenVariableDict, Dict[str, Any]],
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
                Union[FrozenVariableDict, Dict[str, Any]],
                chex.Array,
                chex.Array,
            ],
            Tuple[chex.Array, chex.Array],
        ]
        """

        def encode(
            params: Union[FrozenVariableDict, Dict[str, Any]],
            input: chex.Array,
            carry: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Forward call of the encoder.

            :param params: the model parameters
            :param input: the input
            :param carry: the hidden state (not used)
            :type params: Union[FrozenVariableDict
            :type input: chex.Array
            :type carry: chex.Array
            :return: the encoded input and next carry
            :rtype: Tuple[chex.Array, chex.Array]

            """
            carry_shape = carry.shape[:-1]
            repr, repr_carry = self.encoder.forward(
                params,
                input,
                carry[..., : self._flattened_h_states_shape[0]].reshape(
                    (*carry_shape, *self._h_state_shapes[0])
                ),
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

    def __init__(self, model: Model, num_models: int) -> None:
        self.model = model
        self.num_models = num_models
        self.forward = jax.jit(self.make_forward())

    def init(
        self, model_key: jrandom.PRNGKey, dummy_x: chex.Array
    ) -> Union[FrozenVariableDict, Dict[str, Any]]:
        """
        Initialize model parameters.

        :param model_key: the random number generation key for initializing parameters
        :param dummy_x: the input data
        :type model_key: jrandom.PRNGKey
        :type dummy_x: chex.Array
        :return: the initialized parameters for all of the models
        :rtype: Union[FrozenVariableDict, Dict[str, Any]]

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
    ) -> Callable[
        [
            Union[FrozenVariableDict, Dict[str, Any]],
            chex.Array,
            chex.Array,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the forward call of the ensemble model.

        :return: the forward call.
        :rtype: Callable[
            [
                Union[FrozenVariableDict, Dict[str, Any]],
                chex.Array,
                chex.Array,
            ],
            Tuple[chex.Array, chex.Array],
        ]
        """

        def forward(
            params: Union[FrozenVariableDict, Dict[str, Any]],
            input: chex.Array,
            carry: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Forward call of the ensemble.

            :param params: the model parameters
            :param input: the input
            :param carry: the hidden state (not used)
            :type params: Union[FrozenVariableDict
            :type input: chex.Array
            :type carry: chex.Array
            :return: the output and a pass-through carry
            :rtype: Tuple[chex.Array, chex.Array]

            """
            pred, carry = jax.vmap(self.model.forward)(params, input, carry)
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

    def __init__(self, layers: Sequence[int]) -> None:
        self.model = MLPModule(layers)
        self.forward = jax.jit(self.make_forward())

    def init(
        self, model_key: jrandom.PRNGKey, dummy_x: chex.Array
    ) -> Union[FrozenVariableDict, Dict[str, Any]]:
        """
        Initialize model parameters.

        :param model_key: the random number generation key for initializing parameters
        :param dummy_x: the input data
        :type model_key: jrandom.PRNGKey
        :type dummy_x: chex.Array
        :return: the initialized parameters
        :rtype: Union[FrozenVariableDict, Dict[str, Any]]

        """
        return self.model.init(model_key, dummy_x)

    def make_forward(
        self,
    ) -> Callable[
        [
            Union[FrozenVariableDict, Dict[str, Any]],
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
                Union[FrozenVariableDict, Dict[str, Any]],
                chex.Array,
                chex.Array,
            ],
            Tuple[chex.Array, chex.Array],
        ]
        """

        def forward(
            params: Union[FrozenVariableDict, Dict[str, Any]],
            input: chex.Array,
            carry: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Forward call of the MLP.

            :param params: the model parameters
            :param input: the input
            :param carry: the hidden state (not used)
            :type params: Union[FrozenVariableDict
            :type input: chex.Array
            :type carry: chex.Array
            :return: the output and a pass-through carry
            :rtype: Tuple[chex.Array, chex.Array]

            """
            # NOTE: Assume batch size is first dim
            input = input.reshape((input.shape[0], -1))
            return self.model.apply(params, input), carry

        return forward


class Policy(ABC):
    """Abstract policy class."""

    #: Compute action for interacting with the environment.
    compute_action: Callable[
        [
            Union[FrozenVariableDict, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]

    #: Compute deterministic action.
    deterministic_action: Callable[
        [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array],
        Tuple[chex.Array, chex.Array],
    ]

    def __init__(self, model: Model) -> None:
        self.reset = jax.jit(self.make_reset(model))

    @abstractmethod
    def make_deterministic_action(
        self, model: Model
    ) -> Callable[
        [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the function for taking deterministic action.

        :param model: the model
        :type model: Model
        :return: a function for taking deterministic action
        :rtype: Callable[
            [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array],
            Tuple[chex.Array, chex.Array],
        ]

        """
        raise NotImplementedError

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
            Union[FrozenVariableDict, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]

    @abstractmethod
    def make_random_action(
        self, model: Model
    ) -> Callable[
        [
            Union[FrozenVariableDict, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the function for taking random action.

        :param model: the policy
        :type model: Model
        :return: a function for taking random action
        :rtype: Callable[
            [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array, jrandom.PRNGKey],
            Tuple[chex.Array, chex.Array],
        ]

        """
        raise NotImplementedError

    @abstractmethod
    def make_act_lprob(
        self, model: Model
    ) -> Callable[
        [
            Union[FrozenVariableDict, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array, chex.Array],
    ]:
        """
        Makes the function for taking random action and computing its log probability.

        :param model: the policy
        :type model: Model
        :return: a function for taking random action and computing its log probability
        :rtype: Callable[
            [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array, jrandom.PRNGKey],
            Tuple[chex.Array, chex.Array, chex.Array],
        ]

        """
        raise NotImplementedError

    @abstractmethod
    def make_lprob(
        self, model: Model
    ) -> Callable[
        [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
        chex.Array,
    ]:
        """
        Makes the function for computing action log probability.

        :param model: the policy
        :type model: Model
        :return: a function for computing action log probability
        :rtype: Callable[
            [Union[FrozenVariableDict, Dict[str, Any]], chex.Array, chex.Array, chex.Array],
            chex.Array,
        ]

        """
        raise NotImplementedError
