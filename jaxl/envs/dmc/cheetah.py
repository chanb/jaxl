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

"""Cheetah Domain."""

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


# How long the simulation will run, in seconds.
_DEFAULT_TIME_LIMIT = 10

# Running speed above which reward is 1.
_RUN_SPEED = 10

SUITE = containers.TaggedTasks()


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Cheetah domain."""

    def speed(self):
        """Returns the horizontal speed of the Cheetah."""
        return self.named.data.sensordata["torso_subtreelinvel"][0]


class Cheetah(base.Task):
    """A `Task` to train a running Cheetah."""

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # The indexing below assumes that all joints have a single DOF.
        assert physics.model.nq == physics.model.njnt
        is_limited = physics.model.jnt_limited == 1
        lower, upper = physics.model.jnt_range[is_limited].T
        physics.data.qpos[is_limited] = self.random.uniform(lower, upper)

        # Stabilize the model before the actual simulation.
        physics.step(nstep=200)

        physics.data.time = 0
        self._timeout_progress = 0
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state, ignoring horizontal position."""
        obs = collections.OrderedDict()
        # Ignores horizontal position to maintain translational invariance.
        obs["position"] = physics.data.qpos[1:].copy()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        return rewards.tolerance(
            physics.speed(),
            bounds=(_RUN_SPEED, float("inf")),
            margin=_RUN_SPEED,
            value_at_margin=0,
            sigmoid="linear",
        )


class CheetahEnv(ParameterizedDMCEnv):
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
            os.path.join(os.path.dirname(__file__), "assets/cheetah.xml"),
            parameter_config_path,
            use_default,
            rng,
        )
        physics = Physics.from_xml_string(self.xml, common.ASSETS)

        rng = np.random.RandomState(seed)
        self.task = Cheetah(random=rng)
        environment_kwargs = environment_kwargs or {}
        self.env = control.Environment(
            physics,
            self.task,
            time_limit=time_limit,
            control_timestep=control_timestep,
            **environment_kwargs,
        )

        super().__init__(
            width,
            height,
            camera_id,
            render_mode,
            control_mode,
        )
