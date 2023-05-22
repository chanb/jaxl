from types import SimpleNamespace

from jaxl.constants import *
from jaxl.envs.wrappers import DefaultGymWrapper


def get_environment(env_config: SimpleNamespace) -> DefaultGymWrapper:
    assert (
        env_config.env_type in VALID_ENV_TYPE
    ), f"{env_config.env_type} is not supported (one of {VALID_ENV_TYPE})"

    if env_config.env_type == CONST_GYM:
        import gymnasium as gym

        env = gym.make(env_config.env_name, **vars(env_config.env_kwargs))
    else:
        raise NotImplementedError

    env = DefaultGymWrapper(env)

    return env
