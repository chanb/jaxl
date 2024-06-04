import gymnasium as gym
import numpy as np

from jaxl.constants import *


class ManipulatorLearningEnv(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        include_absorbing_state: bool = False,
    ):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-np.ones(4), high=np.ones(4))
        obs_dim = 59 + int(include_absorbing_state)
        self.observation_space = gym.spaces.Box(
            low=-np.ones(obs_dim), high=np.ones(obs_dim)
        )

        self._get_obs = lambda x: x
        if include_absorbing_state:

            def get_obs(x):
                return np.concatenate((x, [0]), axis=-1)

            self._get_obs = get_obs

    def reset(self, seed: int = None, **kwargs):
        if seed is not None:
            self.env.seed(int(seed))
        observation = self.env.reset()
        return self._get_obs(observation), {}

    def step(self, action):
        observation, reward, terminated, info = self.env.step(action)
        return self._get_obs(observation), reward, terminated, False, info
