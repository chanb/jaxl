import json
import mujoco
import xml.etree.ElementTree as et

from gymnasium import utils, Space
from gymnasium.envs.mujoco import MujocoEnv, mujoco_env
from gymnasium.envs.mujoco.mujoco_env import DEFAULT_SIZE
from typing import Any, Union


class ParameterizedMujocoEnv(MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        parameter_config_path: Any,
        model_path: Any,
        frame_skip: Any,
        observation_space: Space,
        render_mode: Union[str, None] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Union[int, None] = None,
        camera_name: Union[str, None] = None,
        default_camera_config: Union[dict, None] = None,
        **kwargs
    ):
        self._update_env_xml(model_path, parameter_config_path)
        super().__init__(
            model_path=model_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_id=camera_id,
            camera_name=camera_name,
            default_camera_config=default_camera_config,
            **kwargs,
        )

    # TODO: Fix this
    # Make sure we can retrieve the randomized parameters in the future?
    def _update_env_xml(self, model_path: Any, parameter_config_path: Any):
        reference_xml = et.parse(model_path)
        root = reference_xml.getroot()

        with open(parameter_config_path, "r") as f:
            parameter_config = json.load(f)

        for key, value in parameter_config.items():
            pass
        # root.find(".//option[@gravity]").set("gravity", "0 0 {}".format(gravity))

        self.xml = et.tostring(root, encoding="unicode", method="xml")

    def _initialize_simulation(self):
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        # MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)
