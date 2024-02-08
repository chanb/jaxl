from abc import ABC
from flax import linen as nn
from types import SimpleNamespace
from typing import Any, Callable, Dict, Union, Tuple

import chex
import jax
import numpy as np
import optax

from jaxl.constants import *
from jaxl.models.common import Model


class QFunction(ABC):
    """Abstract Q-function class."""

    #: Compute action-value for state-action pairs.
    q_values: Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            Union[optax.Params, Dict[str, Any]],
            Union[optax.Params, Dict[str, Any]],
        ],
        Tuple[chex.Array, chex.Array],
    ]


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
        self, model: Model, encoding: nn.Module,
    ) -> Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
            Union[chex.Array, Dict[str, Any]],
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
                chex.Array,
                chex.Array,
                Union[chex.Array, Dict[str, Any]],
            ],
            Tuple[chex.Array, chex.Array],
        ]

        """

        def compute_q_value(
            params: Union[optax.Params, Dict[str, Any]],
            obs: chex.Array,
            act: chex.Array,
            h_state: Union[chex.Array, Dict[str, Any]],
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Compute action-value based on a state-aciton pair.

            :param params: the model parameters
            :param obs: the observation
            :param act: the action
            :param h_state: the hidden state
            :type params: Union[optax.Params, Dict[str, Any]]
            :type obs: chex.Array
            :type act: chex.Array
            :type h_state: chex.Array
            :return: the action-value given the state-action pair
            :rtype: Tuple[chex.Array, chex.Array]

            """
            state_action = encoding.forward({
                CONST_OBSERVATION: obs,
                CONST_ACTION: act,
            })
            q_val, h_state = model.forward(params, state_action, h_state)
            return q_val, h_state

        return compute_q_value


def q_function_dims(
    obs_dim: chex.Array,
    act_dim: chex.Array,
    qf_config: SimpleNamespace,
) -> Tuple[chex.Array, chex.Array]:
    """
    Gets the Q-function input/output dimensionalities

    :param obs_dim: the observation dimensionality
    :param act_dim: the action dimensionality
    :param qf_config: the Q-function configuration
    :type obs_dim: chex.Array
    :type act_dim: chex.Array
    :type qf_config: SimpleNamespace
    :return: the input/output dimensinalities of the Q-function
    :rtype: Tuple[chex.Array, chex.Array]

    """
    assert (
        qf_config.q_function in VALID_Q_FUNCTION
    ), f"{qf_config.q_function} is not supported (one of {VALID_Q_FUNCTION})"
    assert (
        qf_config.type in VALID_Q_ENCODING
    ), f"{qf_config.type} is not supported (one of {VALID_Q_ENCODING})"

    if qf_config.q_function == CONST_STATE_ACTION_INPUT:
        if qf_config.type == CONST_CONCATENATE_INPUTS_ENCODING:
            return (int(np.product(obs_dim) + np.product(act_dim)),), (1,)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def get_q_function(
    encoding: nn.Module,
    model: Model,
    qf_config: SimpleNamespace,
) -> QFunction:
    """
    Gets a Q-function

    :param encoding: the state-action encoding
    :param model: the model
    :param qf_config: the Q-function configuration
    :type encoding: nn.Module
    :type model: Model
    :type qf_config: SimpleNamespace
    :return: a Q-function
    :rtype: QFunction

    """
    assert (
        qf_config.q_function in VALID_Q_FUNCTION
    ), f"{qf_config.q_function} is not supported (one of {VALID_Q_FUNCTION})"

    if qf_config.q_function == CONST_STATE_ACTION_INPUT:
        return StateActionInputQ(
            encoding=encoding,
            model=model,
        )
    else:
        raise NotImplementedError
