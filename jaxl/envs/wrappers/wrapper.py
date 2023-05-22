import chex
import gymnasium as gym


class DefaultGymWrapper(gym.Wrapper):
    @property
    def reward_dim(self) -> chex.Array:
        return (1,)
