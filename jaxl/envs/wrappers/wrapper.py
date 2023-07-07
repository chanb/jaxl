import chex
import gymnasium as gym

from gymnasium import spaces


class DefaultGymWrapper(gym.Wrapper):
    """
    Gym wrapper that provides extra functionalities for environments.
    """

    @property
    def reward_dim(self) -> chex.Array:
        """
        Gets the reward dimension.
        """
        return (1,)

    @property
    def act_dim(self) -> chex.Array:
        """
        Gets the action dimension.
        """
        action_space = getattr(
            self.unwrapped, "agent_action_space", self.unwrapped.action_space
        )
        if isinstance(action_space, spaces.Discrete):
            return (1, action_space.n)
        else:
            return (*action_space.shape, 1)
