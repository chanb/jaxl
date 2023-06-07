import mujoco
import numpy as np
import os
import xml.etree.ElementTree as et

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv, mujoco_env
from gymnasium.spaces import Box


class ParameterizedInvertedDoublePendulumEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, gravity=-9.81, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            "inverted_double_pendulum.xml",
            5,
            observation_space=observation_space,
            **kwargs,
        )
        reference_path = os.path.join(
            os.path.dirname(mujoco_env.__file__), "assets/inverted_double_pendulum.xml"
        )
        reference_xml = et.parse(reference_path)
        root = reference_xml.getroot()
        root.find(".//option[@gravity]").set("gravity", "0 0 {}".format(gravity))
        new_xml = et.tostring(root, encoding="unicode", method="xml")
        self.model = mujoco.MjModel.from_xml_string(new_xml)
        self.data = mujoco.MjData(self.model)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        ob = self._get_obs()
        x, _, y = self.data.site_xpos[0]
        dist_penalty = 0.01 * x**2 + (y - 2) ** 2
        v1, v2 = self.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        terminated = bool(y <= 1)

        if self.render_mode == "human":
            self.render()
        return ob, r, terminated, False, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos[:1],  # cart x pos
                np.sin(self.data.qpos[1:]),  # link angles
                np.cos(self.data.qpos[1:]),
                np.clip(self.data.qvel, -10, 10),
                np.clip(self.data.qfrc_constraint, -10, 10),
            ]
        ).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1,
        )
        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]
