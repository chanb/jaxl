# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Planar Manipulator domain."""

import collections
import numpy as np
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from typing import Any, Union

from jaxl.envs.dmc.parameterized_env import (
    ParameterizedDMCEnv,
    randomize_env_xml,
    DEFAULT_ID,
    DEFAULT_RGB_ARRAY,
    DEFAULT_SIZE,
)

_CLOSE = 0.01  # (Meters) Distance below which a thing is considered close.
_CONTROL_TIMESTEP = 0.01  # (Seconds)
_TIME_LIMIT = 10  # (Seconds)
_P_IN_HAND = 0.1  # Probabillity of object-in-hand initial state
_P_IN_TARGET = 0.1  # Probabillity of object-in-target initial state
_ARM_JOINTS = [
    "arm_root",
    "arm_shoulder",
    "arm_elbow",
    "arm_wrist",
    "finger",
    "fingertip",
    "thumb",
    "thumbtip",
]
_ALL_PROPS = frozenset(["ball", "target_ball", "cup", "peg", "target_peg", "slot"])
_TOUCH_SENSORS = [
    "palm_touch",
    "finger_touch",
    "thumb_touch",
    "fingertip_touch",
    "thumbtip_touch",
]

SUITE = containers.TaggedTasks()


class Physics(mujoco.Physics):
    """Physics with additional features for the Planar Manipulator domain."""

    def bounded_joint_pos(self, joint_names):
        """Returns joint positions as (sin, cos) values."""
        joint_pos = self.named.data.qpos[joint_names]
        return np.vstack([np.sin(joint_pos), np.cos(joint_pos)]).T

    def joint_vel(self, joint_names):
        """Returns joint velocities."""
        return self.named.data.qvel[joint_names]

    def body_2d_pose(self, body_names, orientation=True):
        """Returns positions and/or orientations of bodies."""
        if not isinstance(body_names, str):
            body_names = np.array(body_names).reshape(-1, 1)  # Broadcast indices.
        pos = self.named.data.xpos[body_names, ["x", "z"]]
        if orientation:
            ori = self.named.data.xquat[body_names, ["qw", "qy"]]
            return np.hstack([pos, ori])
        else:
            return pos

    def touch(self):
        return np.log1p(self.named.data.sensordata[_TOUCH_SENSORS])

    def site_distance(self, site1, site2):
        site1_to_site2 = np.diff(self.named.data.site_xpos[[site2, site1]], axis=0)
        return np.linalg.norm(site1_to_site2)


class Bring(base.Task):
    """A Bring `Task`: bring the prop to the target."""

    def __init__(self, random=None):
        """Initialize an instance of the `Bring` task.

        Args:
          fully_observable: A `bool`, whether the observation should contain the
            position and velocity of the object being manipulated and the target
            location.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._target = "target_ball"
        self._object = "ball"
        self._object_joints = ["_".join([self._object, dim]) for dim in "xzy"]
        self._fully_observable = True
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # Local aliases
        choice = self.random.choice
        uniform = self.random.uniform
        model = physics.named.model
        data = physics.named.data

        # Find a collision-free random initial configuration.
        penetrating = True
        while penetrating:
            # Randomise angles of arm joints.
            is_limited = model.jnt_limited[_ARM_JOINTS].astype(bool)
            joint_range = model.jnt_range[_ARM_JOINTS]
            lower_limits = np.where(is_limited, joint_range[:, 0], -np.pi)
            upper_limits = np.where(is_limited, joint_range[:, 1], np.pi)
            angles = uniform(lower_limits, upper_limits)
            data.qpos[_ARM_JOINTS] = angles

            # Symmetrize hand.
            data.qpos["finger"] = data.qpos["thumb"]

            # Randomise target location.
            target_x = uniform(-0.4, 0.4)
            target_z = uniform(0.1, 0.4)
            target_angle = uniform(-np.pi, np.pi)

            model.body_pos[self._target, ["x", "z"]] = target_x, target_z
            model.body_quat[self._target, ["qw", "qy"]] = [
                np.cos(target_angle / 2),
                np.sin(target_angle / 2),
            ]

            # Randomise object location.
            object_init_probs = [
                _P_IN_HAND,
                _P_IN_TARGET,
                1 - _P_IN_HAND - _P_IN_TARGET,
            ]
            init_type = choice(["in_hand", "in_target", "uniform"], p=object_init_probs)
            if init_type == "in_target":
                object_x = target_x
                object_z = target_z
                object_angle = target_angle
            elif init_type == "in_hand":
                physics.after_reset()
                object_x = data.site_xpos["grasp", "x"]
                object_z = data.site_xpos["grasp", "z"]
                grasp_direction = data.site_xmat["grasp", ["xx", "zx"]]
                object_angle = np.pi - np.arctan2(
                    grasp_direction[1], grasp_direction[0]
                )
            else:
                object_x = uniform(-0.5, 0.5)
                object_z = uniform(0, 0.7)
                object_angle = uniform(0, 2 * np.pi)
                data.qvel[self._object + "_x"] = uniform(-5, 5)

            data.qpos[self._object_joints] = object_x, object_z, object_angle

            # Check for collisions.
            physics.after_reset()
            penetrating = physics.data.ncon > 0

        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns either features or only sensors (to be used with pixels)."""
        obs = collections.OrderedDict()
        obs["arm_pos"] = physics.bounded_joint_pos(_ARM_JOINTS)
        obs["arm_vel"] = physics.joint_vel(_ARM_JOINTS)
        obs["touch"] = physics.touch()
        if self._fully_observable:
            obs["hand_pos"] = physics.body_2d_pose("hand")
            obs["object_pos"] = physics.body_2d_pose(self._object)
            obs["object_vel"] = physics.joint_vel(self._object_joints)
            obs["target_pos"] = physics.body_2d_pose(self._target)
        return obs

    def _is_close(self, distance):
        return rewards.tolerance(distance, (0, _CLOSE), _CLOSE * 2)

    def _peg_reward(self, physics):
        """Returns a reward for bringing the peg prop to the target."""
        grasp = self._is_close(physics.site_distance("peg_grasp", "grasp"))
        pinch = self._is_close(physics.site_distance("peg_pinch", "pinch"))
        grasping = (grasp + pinch) / 2
        bring = self._is_close(physics.site_distance("peg", "target_peg"))
        bring_tip = self._is_close(physics.site_distance("target_peg_tip", "peg_tip"))
        bringing = (bring + bring_tip) / 2
        return max(bringing, grasping / 3)

    def _ball_reward(self, physics):
        """Returns a reward for bringing the ball prop to the target."""
        return self._is_close(physics.site_distance("ball", "target_ball"))

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        return self._ball_reward(physics)


class BringBallEnv(ParameterizedDMCEnv):
    def __init__(
        self,
        parameter_config_path,
        time_limit: int = _TIME_LIMIT,
        control_timestep: float = None,
        seed: Union[int, None] = None,
        use_default: bool = False,
        control_mode: str = "default",
        environment_kwargs: Any = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: int = DEFAULT_ID,
        render_mode: str = DEFAULT_RGB_ARRAY,
        **kwargs,
    ):
        rng = np.random.RandomState(seed)
        self.xml, self.modified_attributes = randomize_env_xml(
            os.path.join(os.path.dirname(__file__), "assets/bring_ball.xml"),
            parameter_config_path,
            use_default,
            rng,
        )
        physics = Physics.from_xml_string(self.xml, common.ASSETS)
        rng = np.random.RandomState(seed)

        self.task = Bring(random=rng)
        environment_kwargs = environment_kwargs or {}
        self.env = control.Environment(
            physics,
            self.task,
            control_timestep=_CONTROL_TIMESTEP,
            time_limit=time_limit,
            **environment_kwargs,
        )

        super().__init__(
            width,
            height,
            camera_id,
            render_mode,
            control_mode,
        )
