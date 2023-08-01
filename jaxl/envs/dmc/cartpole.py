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

"""Cartpole domain."""

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
    """Physics simulation with additional features for the Cartpole domain."""

    def cart_position(self):
        """Returns the position of the cart."""
        return self.named.data.qpos["slider"][0]

    def angular_vel(self):
        """Returns the angular velocity of the pole."""
        return self.data.qvel[1:]

    def pole_angle_cosine(self):
        """Returns the cosine of the pole angle."""
        return self.named.data.xmat[2:, "zz"]

    def bounded_position(self):
        """Returns the state, with pole angle split into sin/cos."""
        return np.hstack(
            (self.cart_position(), self.named.data.xmat[2:, ["zz", "xz"]].ravel())
        )


class Balance(base.Task):
    """A Cartpole `Task` to balance the pole.

    State is initialized either close to the target configuration or at a random
    configuration.
    """

    _CART_RANGE = (-0.25, 0.25)
    _ANGLE_COSINE_RANGE = (0.995, 1)

    def __init__(self, random=None):
        """Initializes an instance of `Balance`.

        Args:
          swing_up: A `bool`, which if `True` sets the cart to the middle of the
            slider and the pole pointing towards the ground. Otherwise, sets the
            cart to a random position on the slider and the pole to a random
            near-vertical position.
          sparse: A `bool`, whether to return a sparse or a smooth reward.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Initializes the cart and pole according to `swing_up`, and in both cases
        adds a small random initial velocity to break symmetry.

        Args:
          physics: An instance of `Physics`.
        """
        nv = physics.model.nv
        physics.named.data.qpos["slider"] = 0.01 * self.random.randn()
        physics.named.data.qpos["hinge_1"] = np.pi + 0.01 * self.random.randn()
        physics.named.data.qpos[2:] = 0.1 * self.random.randn(nv - 2)
        physics.named.data.qvel[:] = 0.01 * self.random.randn(physics.model.nv)
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the (bounded) physics state."""
        obs = collections.OrderedDict()
        obs["position"] = physics.bounded_position()
        obs["velocity"] = physics.velocity()
        return obs

    def _get_reward(self, physics):
        upright = (physics.pole_angle_cosine() + 1) / 2
        centered = rewards.tolerance(physics.cart_position(), margin=2)
        centered = (1 + centered) / 2
        small_control = rewards.tolerance(
            physics.control(), margin=1, value_at_margin=0, sigmoid="quadratic"
        )[0]
        small_control = (4 + small_control) / 5
        small_velocity = rewards.tolerance(physics.angular_vel(), margin=5).min()
        small_velocity = (1 + small_velocity) / 2
        return upright.mean() * small_control * small_velocity * centered

    def get_reward(self, physics):
        """Returns a sparse or a smooth reward, as specified in the constructor."""
        return self._get_reward(physics)


class CartpoleEnv(ParameterizedDMCEnv):
    def __init__(
        self,
        parameter_config_path,
        time_limit: int = _DEFAULT_TIME_LIMIT,
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
            os.path.join(os.path.dirname(__file__), "assets/cartpole.xml"),
            parameter_config_path,
            use_default,
            rng,
        )

        physics = Physics.from_xml_string(self.xml, common.ASSETS)
        rng = np.random.RandomState(seed)
        self.task = Balance(random=rng)
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
