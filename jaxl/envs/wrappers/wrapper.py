import chex
import gymnasium as gym

from gymnasium import spaces, Space


BANG_BANG = "bang_bang"
DISCRETE = "discrete"


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
        if isinstance(self.action_space, spaces.Discrete):
            control_mode = getattr(self, "control_mode", DISCRETE)
            if control_mode == BANG_BANG:
                return (self.action_space.n, 1)
            return (1, self.action_space.n)
        else:
            return (*self.action_space.shape, 1)

    @property
    def action_space(self) -> Space:
        return getattr(
            self.unwrapped, "agent_action_space", self.unwrapped.action_space
        )
