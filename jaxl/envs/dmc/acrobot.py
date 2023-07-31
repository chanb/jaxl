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

"""Acrobot domain."""

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

_DEFAULT_TIME_LIMIT = 10
SUITE = containers.TaggedTasks()


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Acrobot domain."""

    def horizontal(self):
        """Returns horizontal (x) component of body frame z-axes."""
        return self.named.data.xmat[["upper_arm", "lower_arm"], "xz"]

    def vertical(self):
        """Returns vertical (z) component of body frame z-axes."""
        return self.named.data.xmat[["upper_arm", "lower_arm"], "zz"]

    def to_target(self):
        """Returns the distance from the tip to the target."""
        tip_to_target = (
            self.named.data.site_xpos["target"] - self.named.data.site_xpos["tip"]
        )
        return np.linalg.norm(tip_to_target)

    def orientations(self):
        """Returns the sines and cosines of the pole angles."""
        return np.concatenate((self.horizontal(), self.vertical()))


class Balance(base.Task):
    """An Acrobot `Task` to swing up and balance the pole."""

    def __init__(self, sparse, random=None):
        """Initializes an instance of `Balance`.

        Args:
          sparse: A `bool` specifying whether to use a sparse (indicator) reward.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._sparse = sparse
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Shoulder and elbow are set to a random position between [-pi, pi).

        Args:
          physics: An instance of `Physics`.
        """
        physics.named.data.qpos[["shoulder", "elbow"]] = self.random.uniform(
            -np.pi, np.pi, 2
        )
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of pole orientation and angular velocities."""
        obs = collections.OrderedDict()
        obs["orientations"] = physics.orientations()
        obs["velocity"] = physics.velocity()
        return obs

    def _get_reward(self, physics, sparse):
        target_radius = physics.named.model.site_size["target", 0]
        return rewards.tolerance(
            physics.to_target(), bounds=(0, target_radius), margin=0 if sparse else 1
        )

    def get_reward(self, physics):
        """Returns a sparse or a smooth reward, as specified in the constructor."""
        return self._get_reward(physics, sparse=self._sparse)


class AcrobotEnv(ParameterizedDMCEnv):
    def __init__(
        self,
        parameter_config_path,
        time_limit: int = _DEFAULT_TIME_LIMIT,
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
            os.path.join(os.path.dirname(__file__), "assets/acrobot.xml"),
            parameter_config_path,
            use_default,
            rng,
        )
        physics = Physics.from_xml_string(self.xml, common.ASSETS)

        rng = np.random.RandomState(seed)
        self.task = Balance(sparse=False, random=rng)
        environment_kwargs = environment_kwargs or {}
        self.env = control.Environment(
            physics, self.task, time_limit=time_limit, **environment_kwargs
        )

        super().__init__(
            width,
            height,
            camera_id,
            render_mode,
            control_mode,
        )
