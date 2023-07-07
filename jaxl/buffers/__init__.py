import chex
import numpy as np

from copy import deepcopy
from types import SimpleNamespace

from jaxl.buffers.buffers import ReplayBuffer
from jaxl.buffers.ram_buffers import (
    MemoryEfficientNumPyBuffer,
    NextStateNumPyBuffer,
    TrajectoryNumPyBuffer,
)
from jaxl.constants import *
from jaxl.envs.wrappers import DefaultGymWrapper


def get_buffer(
    buffer_config: SimpleNamespace,
    buffer_seed: int = None,
    env: DefaultGymWrapper = None,
    h_state_dim: chex.Array = (1,),
) -> ReplayBuffer:
    """
    Gets a buffer.

    :param buffer_config: the buffer configuration
    :param buffer_seed: the seed for buffer sampling
    :param env: the environment
    :param h_state_dim: the hidden state dimension
    :type buffer_config: SimpleNamespace:
    :type buffer_seed: int:  (Default value = None)
    :type env: DefaultGymWrapper:  (Default value = None)
    :type h_state_dim: chex.Array:  (Default value = (1,)):
    :return: a replay buffer
    :rtype: ReplayBuffer

    """
    assert (
        buffer_config.buffer_type in VALID_BUFFER
    ), f"{buffer_config.buffer_type} is not supported (one of {VALID_BUFFER})"

    if hasattr(buffer_config, "load_buffer"):
        buffer_kwargs = deepcopy(DEFAULT_LOAD_BUFFER_KWARGS)
        buffer_kwargs["load_buffer"] = buffer_config.load_buffer
    else:
        buffer_kwargs = deepcopy(DEFAULT_LOAD_BUFFER_KWARGS)
        buffer_kwargs["buffer_size"] = buffer_config.buffer_size
        buffer_kwargs["obs_dim"] = env.observation_space.shape
        buffer_kwargs["act_dim"] = env.act_dim
        buffer_kwargs["rew_dim"] = env.reward_dim
        buffer_kwargs["h_state_dim"] = h_state_dim
        buffer_kwargs["rng"] = np.random.RandomState(buffer_seed)

        # TODO: Need to implement for trajectory buffer
        # TODO: Might need to change action dim based on environment

    if buffer_config.buffer_type == CONST_DEFAULT:
        buffer_constructor = NextStateNumPyBuffer
    elif buffer_config.buffer_type == CONST_MEMORY_EFFICIENT:
        buffer_constructor = MemoryEfficientNumPyBuffer
    elif buffer_config.buffer_type == CONST_TRAJECTORY:
        buffer_constructor = TrajectoryNumPyBuffer
    else:
        raise NotImplementedError

    buffer = buffer_constructor(**buffer_kwargs)

    buffer_size = getattr(buffer_config, "set_size", False)
    if buffer_size:
        buffer.set_size(buffer_size)

    return buffer
