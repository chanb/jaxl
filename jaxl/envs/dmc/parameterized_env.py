from dm_control.rl import control
from dm_control.suite import base
from dm_env import TimeStep
from typing import Any, Dict, Iterable, Optional, Tuple

import chex
import gymnasium as gym
import jax
import json
import numpy as np
import xml.etree.ElementTree as et

from gymnasium import spaces
from itertools import product

BANG_BANG = "bang_bang"
CONTINUOUS = "continuous"
DEFAULT = "default"
DISCRETE = "discrete"
JOINT = "joint"
GEOM = "geom"
MIN = "min"
MAX = "max"
OPTION = "option"

VALID_CONTROL_MODE = [
    CONTINUOUS,
    BANG_BANG,
    DISCRETE,
    DEFAULT,
]

DEFAULT_ID = 0
DEFAULT_RGB_ARRAY = "rgb_array"
DEFAULT_SIZE = 480


def sample_data(attr_data: dict, np_random: np.random.Generator) -> chex.Array:
    min_vals = np.array(attr_data[MIN])
    max_vals = np.array(attr_data[MAX])
    sampled_vals = np_random.uniform(size=len(attr_data[DEFAULT]))
    sampled_vals = (max_vals - min_vals) * sampled_vals + min_vals
    return sampled_vals


def randomize_env_xml(
    model_path: Any,
    parameter_config_path: Any,
    use_default: bool,
    rng: np.random.RandomState,
) -> Tuple[str, Dict[str, Any]]:
    reference_xml = et.parse(model_path)
    root = reference_xml.getroot()

    modified_attributes = {}
    if not use_default:
        with open(parameter_config_path, "r") as f:
            parameter_config = json.load(f)

        for tag, configs in parameter_config.items():
            modified_attributes.setdefault(tag, {})
            if tag in [OPTION]:
                for attr, attr_data in configs.items():
                    sampled_vals = sample_data(attr_data, rng)
                    root.find(".//{}[@{}]".format(tag, attr)).set(
                        attr, " ".join(map(lambda x: str(x), sampled_vals))
                    )
                    modified_attributes[tag][attr] = sampled_vals
            elif tag in [GEOM, JOINT]:
                for name, attr_dict in configs.items():
                    for attr, attr_data in attr_dict.items():
                        sampled_vals = sample_data(attr_data, rng)
                        root.find(".//{}[@name='{}']".format(tag, name)).set(
                            attr, " ".join(map(lambda x: str(x), sampled_vals))
                        )
                    modified_attributes[tag].setdefault(name, {})
                    modified_attributes[tag][name][attr] = sampled_vals

    xml = et.tostring(root, encoding="unicode", method="xml")
    return xml, modified_attributes


class ParameterizedDMCEnv(gym.Env):
    env: control.Environment
    task: base.Task

    metadata = {
        "render_modes": [
            DEFAULT_RGB_ARRAY,
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: int = DEFAULT_ID,
        render_mode: str = None,
        control_mode: str = CONTINUOUS,
        **kwargs,
    ):
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.camera_id = camera_id
        self.action_space = spaces.Box(
            low=self.env.action_spec().minimum,
            high=self.env.action_spec().maximum,
            shape=self.env.action_spec().shape,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                int(
                    sum(
                        [
                            np.product(val.shape) if len(val.shape) else 1
                            for val in self.env.observation_spec().values()
                        ]
                    )
                ),
            ),
            dtype=np.float64,
        )

        self.control_mode = control_mode
        assert control_mode in VALID_CONTROL_MODE
        if control_mode == BANG_BANG:
            n_dim = int(np.prod(self.action_space.shape))
            self.agent_action_space = spaces.Discrete(n_dim)

            def process_action(action):
                return (-1) ** (np.array(action) + 1)

        elif control_mode == DISCRETE:
            n_dim = int(np.prod(self.action_space.shape))
            if n_dim > 1:
                self.agent_action_space = spaces.Discrete(2**n_dim)

                action_map = np.array(list(product(np.array([-1.0, 1.0]), repeat=n_dim)))

                def process_action(action):
                    return action_map[int(action)]
            else:
                actions = (
                    [2.0] ** np.arange(-4, 1)[:, None]
                ).flatten()
                action_map = np.concatenate([-actions, [0], actions])
                self.agent_action_space = spaces.Discrete(len(action_map))

                def process_action(u):
                    return action_map[u].item()


        else:
            self.agent_action_space = self.action_space

            def process_action(action):
                return action

        self.process_action = process_action

    def get_config(self):
        return {
            "xml": self.xml,
            "modified_attributes": self.modified_attributes,
        }

    def _get_obs(self, timestep: TimeStep):
        return np.concatenate(
            [
                val.flatten() if isinstance(val, np.ndarray) else np.array([val])
                for val in jax.tree_util.tree_leaves(timestep.observation)
            ],
            axis=0,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[chex.Array, Dict[str, Any]]:
        self.task._random = np.random.RandomState(seed)
        timestep = self.env.reset()
        return self._get_obs(timestep), {}

    def step(
        self,
        action: chex.Array,
    ) -> Tuple[chex.Array, float, bool, bool, Dict[str, Any]]:
        action = self.process_action(action)
        timestep = self.env.step(action)
        next_obs = self._get_obs(timestep)
        truncated = timestep.last()
        return (next_obs, timestep.reward, False, truncated, {})

    def render(self):
        return self.env.physics.render(
            camera_id=self.camera_id, height=self.height, width=self.width
        )
