import os

from gymnasium.envs.registration import register
from types import SimpleNamespace

from jaxl.constants import *
from jaxl.envs.wrappers import DefaultGymWrapper


def get_environment(env_config: SimpleNamespace) -> DefaultGymWrapper:
    """
    Gets an environment.

    :param env_config: the environment configration file
    :type env_config: SimpleNamespace
    :return: the environment
    :rtype: DefaultGymWrapper

    """
    assert (
        env_config.env_type in VALID_ENV_TYPE
    ), f"{env_config.env_type} is not supported (one of {VALID_ENV_TYPE})"

    if env_config.env_type == CONST_GYM:
        import gymnasium as gym

        env = gym.make(env_config.env_name, **vars(env_config.env_kwargs))
    elif env_config.env_type == CONST_DM_CONTROL:
        raise NotImplementedError
    else:
        raise NotImplementedError

    env = DefaultGymWrapper(env)

    return env


register(
    id="ParameterizedInvertedDoublePendulum-v0",
    entry_point="jaxl.envs.mujoco.inverted_double_pendulum:ParameterizedInvertedDoublePendulumEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
    kwargs={
        "parameter_config_path": os.path.join(
            os.path.dirname(__file__), "mujoco/configs/inverted_double_pendulum.json"
        )
    },
)

register(
    id="ParameterizedHopper-v0",
    entry_point="jaxl.envs.mujoco.hopper:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
    kwargs={
        "parameter_config_path": os.path.join(
            os.path.dirname(__file__), "mujoco/configs/hopper.json"
        )
    },
)
register(
    id="ParameterizedHalfCheetah-v0",
    entry_point="jaxl.envs.mujoco.half_cheetah:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
    kwargs={
        "parameter_config_path": os.path.join(
            os.path.dirname(__file__), "mujoco/configs/half_cheetah.json"
        )
    },
)
register(
    id="ParameterizedSwimmer-v0",
    entry_point="jaxl.envs.mujoco.swimmer:SwimmerEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
    kwargs={
        "parameter_config_path": os.path.join(
            os.path.dirname(__file__), "mujoco/configs/swimmer.json"
        )
    },
)
register(
    id="ParameterizedAnt-v0",
    entry_point="jaxl.envs.mujoco.ant:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
    kwargs={
        "parameter_config_path": os.path.join(
            os.path.dirname(__file__), "mujoco/configs/ant.json"
        )
    },
)
