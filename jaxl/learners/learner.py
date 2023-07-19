from abc import ABC
from orbax.checkpoint import PyTreeCheckpointer
from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple, Union

import _pickle as pickle
import chex
import numpy as np
import optax
import os

from jaxl.buffers import get_buffer, ReplayBuffer
from jaxl.constants import *
from jaxl.envs import get_environment, DefaultGymWrapper
from jaxl.models import Model


"""
Abstract learner classes.
XXX: Try not to modify this.
"""


class Learner(ABC):
    """
    Abstract learner class.
    """

    #: Replay buffer.
    _buffer: ReplayBuffer

    #: The model to train.
    _model: Union[Model, Dict[str, Model]]

    #: The optimizer to use.
    _optimizer: Union[
        optax.GradientTransformation, Dict[str, optax.GradientTransformation]
    ]

    #: The model and optimizer states.
    _model_dict: Dict[str, Any]

    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        self._config = config
        self._model_config = model_config
        self._optimizer_config = optimizer_config

    @property
    def model_dict(self):
        """
        Model dictionary
        """
        return self._model_dict

    @model_dict.setter
    def model_dict(self, model_dict: Dict[str, Any]):
        """
        Setter for model_dict

        :param model_dict: contains model states
        :type model_dict: Dict[str, Any]

        """
        self._model_dict = model_dict

    @property
    def buffer(self):
        """
        Buffer
        """
        return self._buffer

    def checkpoint(self, final=False) -> Dict[str, Any]:
        """
        Returns the parameters to checkpoint

        :param final: whether or not this is the final checkpoint
        :type final: bool (DefaultValue = False)
        :return: the checkpoint parameters
        :rtype: Dict[str, Any]

        """
        return {
            CONST_MODEL_DICT: self.model_dict,
        }

    def load_checkpoint(self, params: Dict[str, Any]):
        """
        Loads a model state from a saved checkpoint.

        :param params: the checkpointed parameters
        :type params: Dict[str, Any]

        """
        self.model_dict = params[CONST_MODEL_DICT]

    def save_env_config(self, checkpoint_path: str):
        """
        Saves the environment configuration.

        :param checkpoint_path: directory path to store the environment configuration to
        :type checkpoint_path: str

        """
        pass

    def save_buffer(self, checkpoint_path: str):
        """
        Saves the buffer.
        By default, we assume the buffer is fixed, thus nothing will be saved.

        :param checkpoint_path: directory path to store the buffer to
        :type checkpoint_path: str

        """
        pass

    def _generate_dummy_x(self, input_dim: chex.Array) -> chex.Array:
        """
        Generates an arbitrary input based on the input space.
        This is mainly for initializing the model.

        :param input_dim: the input dimension
        :type input_dim: chex.Array
        :return: an arbitrary input
        :rtype: chex.Array

        """
        return np.zeros((1, 1, *input_dim))

    def update(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Updates the model

        :param *args:
        :param **kwargs:
        :return: the update information
        :rtype: Dict[str, Any]

        """
        raise NotImplementedError

    def _initialize_buffer(self):
        """
        Construct the buffer
        """
        raise NotImplementedError

    def _initialize_model_and_opt(self, input_dim: chex.Array, output_dim: chex.Array):
        """
        Construct the model and the optimizer.

        :param input_dim: input dimension of the data point
        :param output_dim: output dimension of the data point
        :type input_dim: chex.Array
        :type output_dim: chex.Array

        """
        raise NotImplementedError


class OfflineLearner(Learner):
    """
    Offline learner class that extends the ``Learner`` class.
    Offline learner assumes the buffer to be fixed and has no interaction with any environments.
    """

    #: The loss function to use.
    _loss: Callable[..., Tuple[chex.Array, Dict]]

    #: The number of gradient steps per update call.
    _num_updates_per_epoch: int

    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        super().__init__(config, model_config, optimizer_config)

        self._initialize_buffer()
        self._initialize_model_and_opt(self._buffer.input_dim, self._buffer.output_dim)
        self._num_updates_per_epoch = self._config.num_updates_per_epoch

    def _initialize_buffer(self):
        """
        Construct the buffer
        """
        self._buffer = get_buffer(
            self._config.buffer_config, self._config.seeds.buffer_seed
        )


class OnlineLearner(Learner):
    """
    Online learner class that extends the ``Learner`` class.
    Online learner assumes the buffer to be changing and interacts with an environment.
    """

    #: The environment to interact with.
    _env: DefaultGymWrapper

    #: The number of environment interactions per update call.
    _num_steps_per_epoch: int

    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        super().__init__(config, model_config, optimizer_config)

        self._initialize_env()
        self._initialize_buffer()
        self._initialize_model_and_opt(self._buffer.input_dim, self._buffer.output_dim)
        self._num_steps_per_epoch = config.num_steps_per_epoch

    def _initialize_env(self):
        """
        Initialize the environment.
        """
        self._env = get_environment(self._config.env_config)

    def _initialize_buffer(self):
        """
        Construct the buffer
        """
        h_state_dim = (1,)
        if hasattr(self._model_config, "h_state_dim"):
            h_state_dim = self._model_config.h_state_dim
        self._buffer = get_buffer(
            self._config.buffer_config,
            self._config.seeds.buffer_seed,
            self._env,
            h_state_dim,
        )

    def save_buffer(self, checkpoint_path: str):
        """
        Saves the buffer.

        :param checkpoint_path: directory path to store the buffer to
        :type checkpoint_path: str

        """
        self._buffer.save(checkpoint_path)

    def save_env_config(self, checkpoint_path: str):
        """
        Saves the environment configuration.
        :param checkpoint_path: directory path to store the environment configuration to
        :type checkpoint_path: str

        """
        if hasattr(self._env, "get_config"):
            with open(checkpoint_path, "wb") as f:
                pickle.dump(self._env.get_config(), f)

    @property
    def env(self):
        """Environment."""
        return self._env
