import chex
import gymnasium as gym


class DefaultGymWrapper(gym.Wrapper):
    """
    Gym wrapper that provides extra functionalities for environments.
    """

    @property
    def reward_dim(self) -> chex.Array:
        """
        Gets the number of reward dimension.
        """
        return (1,)
