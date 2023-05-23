from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, Any

import chex


class NoSampleError(Exception):
    """Exception for when buffer has no samples."""

    pass


class LengthMismatchError(Exception):
    """Exception for when there is a length mismatch."""

    pass


class CheckpointIndexError(Exception):
    """Exception for when checkpointing is using invalid index."""

    pass


class ReplayBuffer(ABC):
    """Abstract replay buffer class"""

    @abstractproperty
    def buffer_size(self):
        """The buffer size."""
        raise NotImplementedError

    @abstractproperty
    def is_full(self):
        """Whether or not the buffer is full."""
        raise NotImplementedError

    @abstractproperty
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def push(
        self,
        obs: chex.Array,
        h_state: chex.Array,
        act: chex.Array,
        rew: float,
        terminated: bool,
        truncated: bool,
        info: dict,
        **kwargs
    ) -> bool:
        """
        Push data into buffer.

        :param obs: the observation
        :param h_state: the hidden state
        :param act: the action taken
        :param rew: the reward
        :param terminated: end of the episode
        :param truncated: early truncation due to time limit
        :param info: environment information
        :param **kwargs:
        :type obs: chex.Array
        :type h_state: chex.Array
        :type act: chex.Array
        :type rew: float
        :type terminated: bool
        :type truncated: bool
        :type info: dict
        :return: whether the sample is pushed successfully
        :rtype: bool

        """
        raise NotImplementedError

    @abstractmethod
    def clear(self, **kwargs):
        """
        Reset the buffer to be empty.

        :param **kwargs:

        """
        raise NotImplementedError

    @abstractmethod
    def sample(
        self, batch_size: int, idxes: Optional[chex.Array] = None, **kwargs
    ) -> Any:
        """
        Sample data from the buffer.

        :param batch_size: batch size
        :param idxes: the specified indices if needed.
        :param **kwargs:
        :type batch_size: int
        :param idxes: Optional[chex.Array]:  (Default value = None)
        :return: the data
        :rtype: Any

        """
        raise NotImplementedError

    @abstractmethod
    def sample_init_obs(self, batch_size: int, **kwargs) -> Any:
        """
        Sample initial observations from the buffer.

        :param batch_size: batch size
        :param **kwargs:
        :type batch_size: int
        :return: the initial observations
        :rtype: Any

        """
        raise NotImplementedError

    @abstractmethod
    def save(self, save_path: str, **kwargs):
        """
        Saves the replay buffer.

        :param save_path: the file name of the replay buffer
        :param **kwargs:
        :type save_path: str

        """
        raise NotImplementedError

    @abstractmethod
    def load(self, load_path: str, **kwargs):
        """
        Loads a replay buffer.

        :param load_path: the file name of the replay buffer
        :param **kwargs:
        :type load_path: str

        """
        raise NotImplementedError

    @abstractproperty
    def input_dim(self):
        """
        The input data dimension.
        """
        raise NotImplementedError

    @abstractproperty
    def output_dim(self):
        """
        The output data dimension.
        """
        raise NotImplementedError
