import chex
import json
import mujoco
import numpy as np
import xml.etree.ElementTree as et

from gymnasium import utils, Space, spaces
from gymnasium.envs.mujoco import MujocoEnv, mujoco_env
from gymnasium.envs.mujoco.mujoco_env import DEFAULT_SIZE
from typing import Any, Union


DEFAULT = "default"
JOINT = "joint"
GEOM = "geom"
MIN = "min"
MAX = "max"
OPTION = "option"


def sample_data(attr_data: dict, np_random: np.random.Generator) -> chex.Array:
    min_vals = np.array(attr_data[MIN])
    max_vals = np.array(attr_data[MAX])
    sampled_vals = np_random.uniform(size=len(attr_data[DEFAULT]))
    sampled_vals = (max_vals - min_vals) * sampled_vals + min_vals
    return sampled_vals


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
        seed: Union[int, None] = None,
        use_default: bool = False,
        bang_bang_control: bool = False,
        **kwargs
    ):
        self._rng = np.random.RandomState(seed)
        self._update_env_xml(model_path, parameter_config_path, use_default)
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

        if bang_bang_control:
            self.action_space = spaces.MultiDiscrete(np.ones(self.action_space.shape) * 2)
            def process_action(action):
                return (-1) ** (action + 1)
        else:
            def process_action(action):
                return action
        self.process_action = process_action

    def _update_env_xml(
        self, model_path: Any, parameter_config_path: Any, use_default: bool
    ):
        reference_xml = et.parse(model_path)
        root = reference_xml.getroot()

        self.modified_attributes = {}
        if not use_default:
            with open(parameter_config_path, "r") as f:
                parameter_config = json.load(f)

            for tag, configs in parameter_config.items():
                self.modified_attributes.setdefault(tag, {})
                if tag in [OPTION]:
                    for attr, attr_data in configs.items():
                        sampled_vals = sample_data(attr_data, self._rng)
                        root.find(".//{}[@{}]".format(tag, attr)).set(
                            attr, " ".join(map(lambda x: str(x), sampled_vals))
                        )
                        self.modified_attributes[tag][attr] = sampled_vals
                elif tag in [GEOM, JOINT]:
                    for name, attr_dict in configs.items():
                        for attr, attr_data in attr_dict.items():
                            sampled_vals = sample_data(attr_data, self._rng)
                            root.find(".//{}[@name='{}']".format(tag, name)).set(
                                attr, " ".join(map(lambda x: str(x), sampled_vals))
                            )
                        self.modified_attributes[tag].setdefault(name, {})
                        self.modified_attributes[tag][name][attr] = sampled_vals

        self.xml = et.tostring(root, encoding="unicode", method="xml")

    def _initialize_simulation(self):
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        # MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def get_config(self):
        return {
            "xml": self.xml,
            "modified_attributes": self.modified_attributes,
        }
