from abc import ABC
from orbax.checkpoint import PyTreeCheckpointer
from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple, Union

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

    _buffer: ReplayBuffer
    _model: Union[Model, Dict[str, Model]]
    _optimizer: Union[
        optax.GradientTransformation, Dict[str, optax.GradientTransformation]
    ]
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
        Getter for model_dict
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

    def checkpoint(self, checkpoint_path: str, exist_ok: bool = False):
        """
        Saves the current model state.

        :param checkpoint_path: directory path to store the checkpoint to
        :param exist_ok: whether to overwrite the existing checkpoint path
        :type checkpoint_path: str
        :type exist_ok: bool (Default value = False)

        """
        assert not (
            os.path.isfile(checkpoint_path) and not exist_ok
        ), f"{checkpoint_path} already exists and experiment prevents from overwriting"

        checkpointer = PyTreeCheckpointer()
        checkpointer.save(checkpoint_path, self.model_dict)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Loads a model state from a saved checkpoint.

        :param checkpoint_path: directory path to load the checkpoint from
        :type checkpoint_path: str

        """
        assert os.path.isfile(checkpoint_path), f"{checkpoint_path} does not exist"

        checkpointer = PyTreeCheckpointer()
        self._model_dict = checkpointer.restore(checkpoint_path)

    def save_buffer(self, checkpoint_path: str):
        """
        Saves the buffer.
        By default, we assume the buffer is fixed, thus nothing will be saved.

        :param checkpoint_path: directory path to store the buffer to
        :type checkpoint_path: str

        """
        pass

    def _generate_dummy_x(self) -> chex.Array:
        """
        Generates an arbitrary input based on the buffer's input space.
        This is mainly for initializing the model.

        :return: an arbitrary input
        :rtype: chex.Array

        """
        return np.zeros((1, 1, *self._buffer.input_dim))

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

    _loss: Callable[..., Tuple[chex.Array, Dict]]

    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        super().__init__(config, model_config, optimizer_config)

        self._initialize_buffer()
        self._initialize_model_and_opt(self._buffer.input_dim, self._buffer.output_dim)

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

    _env: DefaultGymWrapper

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
