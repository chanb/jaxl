from dm_control.rl import control
from dm_control.suite import base
from dm_env import TimeStep
from typing import Any, Dict, Optional, Tuple

import chex
import gymnasium as gym
import jax
import json
import numpy as np
import xml.etree.ElementTree as et

from gymnasium import spaces
from itertools import product

BANG_BANG = "bang_bang"
DEFAULT = "default"
DISCRETE = "discrete"
JOINT = "joint"
GEOM = "geom"
MIN = "min"
MAX = "max"
OPTION = "option"

VALID_CONTROL_MODE = [
    DEFAULT,
    BANG_BANG,
    DISCRETE,
]


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

    def __init__(self, control_mode: str = DEFAULT, **kwargs):
        self.action_space = spaces.Box(
            low=self.env.action_spec().minimum,
            high=self.env.action_spec().maximum,
            shape=self.env.action_spec().shape,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(sum(
                [np.product(val.shape) for val in self.env.observation_spec().values()]
            ),),
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
            self.agent_action_space = spaces.Discrete(3**n_dim)

            action_map = np.array(list(product(np.arange(-1, 2), repeat=n_dim)))

            def process_action(action):
                return action_map[int(action)]

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
        return np.concatenate(jax.tree_util.tree_leaves(timestep.observation), axis=0)

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
        return (next_obs, timestep.reward, truncated, False, {})
