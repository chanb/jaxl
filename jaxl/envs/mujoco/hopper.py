__credits__ = ["Kallinteris-Andreas"]

import numpy as np
import os

from gymnasium.spaces import Box

from jaxl.envs.mujoco.parameterized_env import ParameterizedMujocoEnv
from dm_control.utils.rewards import tolerance


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class HopperEnv(ParameterizedMujocoEnv):
    r"""
    ## Description
    This environment is based on the work done by Erez, Tassa, and Todorov in
    ["Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks"](http://www.roboticsproceedings.org/rss07/p10.pdf). The environment aims to
    increase the number of independent state and control variables as compared to
    the classic control environments. The hopper is a two-dimensional
    one-legged figure that consist of four main body parts - the torso at the
    top, the thigh in the middle, the leg in the bottom, and a single foot on
    which the entire body rests. The goal is to make hops that move in the
    forward (right) direction by applying torques on the three hinges
    connecting the four body parts.


    ## Action Space
    The action space is a `Box(-1, 1, (3,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    |-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
    | 0   | Torque applied on the thigh rotor  | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
    | 1   | Torque applied on the leg rotor    | -1          | 1           | leg_joint                        | hinge | torque (N m) |
    | 2   | Torque applied on the foot rotor   | -1          | 1           | foot_joint                       | hinge | torque (N m) |


    ## Observation Space
    Observations consist of positional values of different body parts of the
    hopper, followed by the velocities of those individual parts
    (their derivatives) with all the positions ordered before all the velocities.

    By default, observations do not include the x-coordinate of the hopper. It may
    be included by passing `exclude_current_positions_from_observation=False` during construction.
    In that case, the observation space will be `Box(-Inf, Inf, (12,), float64)` where the first observation
    represents the x-coordinate of the hopper.
    Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x-coordinate
    will be returned in `info` with key `"x_position"`.

    However, by default, the observation is a `Box(-Inf, Inf, (11,), float64)` where the elements
    correspond to the following:

    | Num | Observation                                        | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | -------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | z-coordinate of the torso (height of hopper)       | -Inf | Inf | rootz                            | slide | position (m)             |
    | 1   | angle of the torso                                 | -Inf | Inf | rooty                            | hinge | angle (rad)              |
    | 2   | angle of the thigh joint                           | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
    | 3   | angle of the leg joint                             | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
    | 4   | angle of the foot joint                            | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
    | 5   | velocity of the x-coordinate of the torso          | -Inf | Inf | rootx                          | slide | velocity (m/s)           |
    | 6   | velocity of the z-coordinate (height) of the torso | -Inf | Inf | rootz                          | slide | velocity (m/s)           |
    | 7   | angular velocity of the angle of the torso         | -Inf | Inf | rooty                          | hinge | angular velocity (rad/s) |
    | 8   | angular velocity of the thigh hinge                | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
    | 9   | angular velocity of the leg hinge                  | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
    | 10  | angular velocity of the foot hinge                 | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
    | excluded | x-coordinate of the torso                     | -Inf | Inf | rootx                            | slide | position (m)             |


    ## Rewards
    The reward consists of three parts:
    - *healthy_reward*:
    Every timestep that the Hopper is healthy (see definition in section "Episode Termination"),
    it gets a reward of fixed value `healthy_reward`.
    - *forward_reward*:
    A reward for moving forward,
    this reward would be positive if the Hopper moves forward (in the positive $x$ direction / in the right direction).
    $w_{forward} \times \frac{dx}{dt}$, where
    $dx$ is the displacement of the "torso" ($x_{after-action} - x_{before-action}$),
    $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is 4),
    and `frametime` which is 0.002 - so the default is $dt = 4 \times 0.002 = 0.008$,
    $w_{forward}$ is the `forward_reward_weight` (default is $1$).
    - *ctrl_cost*:
    A negative reward to penalize the Hopper for taking actions that are too large.
    $w_{control} \times \\|action\\|_2^2$,
    where $w_{control}$ is `ctrl_cost_weight` (default is $10^{-3}$).

    The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost*,
    and `info` will also contain the individual reward terms.

    ## Starting State
    All observations start in state
    (0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) with a uniform noise
     in the range of [-`reset_noise_scale`, `reset_noise_scale`] added to the values for stochasticity.


    ## Episode End
    #### Termination
    If `terminate_when_unhealthy is True` (which is the default), the environment terminates when the Hopper is unhealthy.
    The Hopper is unhealthy if any of the following happens:

    1. An element of `observation[1:]` (if  `exclude_current_positions_from_observation=True`, else `observation[2:]`) is no longer contained in the closed interval specified by the argument `healthy_state_range`
    2. The height of the hopper (`observation[0]` if  `exclude_current_positions_from_observation=True`, else `observation[1]`) is no longer contained in the closed interval specified by the argument `healthy_z_range` (usually meaning that it has fallen)
    3. The angle of the torso (`observation[1]` if  `exclude_current_positions_from_observation=True`, else `observation[2]`) is no longer contained in the closed interval specified by the argument `healthy_angle_range`

    #### Truncation
    The maximum duration of an episode is 1000 timesteps.


    ## Arguments
    `gymnasium.make` takes additional arguments such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc.

    ```python
    import gymnasium as gym
    env = gym.make('Hopper-v5', ctrl_cost_weight=1e-3, ....)
    ```

    | Parameter                                    | Type      | Default               | Description                                                                                                                                                                     |
    | -------------------------------------------- | --------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   | `"hopper_v5.xml"`     | Path to a MuJoCo model                                                                                                                                                          |
    | `forward_reward_weight`                      | **float** | `1`                   | Weight for _forward_reward_ term (see section on reward)                                                                                                                        |
    | `ctrl_cost_weight`                           | **float** | `1e-3`                | Weight for _ctrl_cost_ reward (see section on reward)                                                                                                                           |
    | `healthy_reward`                             | **float** | `1`                   | Weight for _healthy_reward_ reward (see section on reward)                                                                                                                      |
    | `terminate_when_unhealthy`                   | **bool**  | `True`                | If true, issue a done signal if the hopper is no longer healthy                                                                                                                 |
    | `healthy_state_range`                        | **tuple** | `(-100, 100)`         | The elements of `observation[1:]` (if `exclude_current_positions_from_observation=True`, else `observation[2:]`) must be in this range for the hopper to be considered healthy  |
    | `healthy_z_range`                            | **tuple** | `(0.7, float("inf"))` | The z-coordinate must be in this range for the hopper to be considered healthy                                                                                                  |
    | `healthy_angle_range`                        | **tuple** | `(-0.2, 0.2)`         | The angle given by `observation[1]` (if `exclude_current_positions_from_observation=True`, else `observation[2]`) must be in this range for the hopper to be considered healthy |
    | `reset_noise_scale`                          | **float** | `5e-3`                | Scale of random perturbations of initial position and velocity (see section on Starting State)                                                                                  |
    | `exclude_current_positions_from_observation` | **bool**  | `True`                | Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies               |

    ## Version History
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3.
    * v3: Support for `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
    * v2: All continuous control environments now use mujoco-py >= 1.50.
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0).
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 125,
    }

    def __init__(
        self,
        parameter_config_path,
        frame_skip=4,
        default_camera_config=DEFAULT_CAMERA_CONFIG,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range=(0.7, float("inf")),
        healthy_angle_range=(-0.2, 0.2),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        seed=None,
        use_default=False,
        control_mode="default",
        **kwargs,
    ):
        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        super().__init__(
            parameter_config_path,
            os.path.join(os.path.dirname(__file__), "assets/hopper.xml"),
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            seed=seed,
            use_default=use_default,
            control_mode=control_mode,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        return terminated

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        action = self.process_action(action)
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        # DMC reward function
        z_position_after = self.data.qpos[1]
        standing = tolerance(z_position_after, (0.6, 2))
        hopping = tolerance(
            x_velocity,
            bounds=(1.5, float("inf")),
            margin=1.5 / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        shaped_reward = standing * hopping

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward * healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        terminated = self.terminated
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": x_position_after,
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
            "x_velocity": x_velocity,
        }
        if self.control_mode != "default":
            info["shaped_reward"] = shaped_reward

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }
