from typing import Any, Callable, Dict, Union, Tuple

import chex
import jax
import jax.nn as nn
import optax

from jaxl.models.common import Model, QFunction


class StateActionInputQ(QFunction):
    """Q-function that takes in state-action pairs and output corresponding action values."""

    def __init__(
        self,
        encoding: nn.Module,
        model: Model,
    ):
        super().__init__()
        self.encoding = encoding
        self.model = model
        self.q_values = jax.jit(
            self.make_q_values(model, encoding)
        )

    def make_q_values(
        self, model: Model, encoding: Model,
    ) -> Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            Union[optax.Params, Dict[str, Any]],
            Union[optax.Params, Dict[str, Any]],
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the function for taking action during interaction

        :param model: the model
        :param encoding: the encoding of state-action pair
        :type model: Model
        :type encoding: Model
        :return: a function for taking action during interaction
        :rtype: Callable[
            [
                Union[optax.Params, Dict[str, Any]],
                Union[optax.Params, Dict[str, Any]],
                Union[optax.Params, Dict[str, Any]],
            ],
            Tuple[chex.Array, chex.Array],
        ]

        """

        def compute_q_value(
            params: Union[optax.Params, Dict[str, Any]],
            state_action: Union[optax.Params, Dict[str, Any]],
            h_state: Union[optax.Params, Dict[str, Any]],
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Compute action-value based on a state-aciton pair.

            :param params: the model parameters
            :param state_action: the state-action pair
            :param h_state: the hidden state
            :type params: Union[optax.Params, Dict[str, Any]]
            :type obs: chex.Array
            :type h_state: chex.Array
            :return: the action-value given the state-action pair
            :rtype: Tuple[chex.Array, chex.Array]

            """
            enc_state_action, h_state = encoding.forward(params, state_action, h_state)
            q_val, h_state = model.forward(params, enc_state_action, h_state)
            return q_val, h_state

        return compute_q_value
