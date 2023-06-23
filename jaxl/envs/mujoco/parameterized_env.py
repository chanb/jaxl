from gymnasium import utils, Space
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_env import DEFAULT_SIZE
from typing import Any, Union

class ParameterizedMujocoEnv(MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        model_path: Any,
        frame_skip: Any,
        observation_space: Space,
        render_mode: Union[str, None] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Union[int, None] = None,
        camera_name: Union[str, None] = None,
        default_camera_config: Union[dict, None] = None
    ):
        pass
