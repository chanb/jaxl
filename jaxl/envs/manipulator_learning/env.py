import gymnasium as gym
import numpy as np

from jaxl.constants import *


class ManipulatorLearningEnv(gym.Wrapper):
    def __init__(
        self,
        env,
    ):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-np.ones(4), high=np.ones(4))
        self.observation_space = gym.spaces.Box(low=-np.ones(59), high=np.ones(59))

    def reset(self, seed: int = None, **kwargs):
        if seed is not None:
            self.env.seed(int(seed))
        observation = self.env.reset()
        return observation, {}

    def step(self, action):
        observation, reward, terminated, info = self.env.step(action)
        return observation, reward, terminated, False, info
