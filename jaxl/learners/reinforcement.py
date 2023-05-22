import _pickle as pickle
import chex
import numpy as np
import os

from types import SimpleNamespace
from typing import Union

from jaxl.constants import *
from jaxl.envs.rollouts import Rollout
from jaxl.learners.learner import OnlineLearner
from jaxl.utils import RunningMeanStd


class ReinforcementLearner(OnlineLearner):
    _update_frequency: int
    _gamma: float
    _value_rms: Union[bool, RunningMeanStd]

    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        super().__init__(config, model_config, optimizer_config)
        self._update_frequency = config.buffer_config.buffer_size
        self._gamma = config.gamma
        self._val_rms = False
        if getattr(config, CONST_VALUE_RMS, False):
            self._val_rms = RunningMeanStd(shape=self._env.reward_dim)

    @property
    def val_rms(self):
        return self._val_rms

    @property
    def update_frequency(self):
        return self._update_frequency

    def checkpoint(self, checkpoint_path: str, exist_ok: bool = False):
        super().checkpoint(checkpoint_path, exist_ok)
        learner_dict = {
            CONST_VALUE_RMS: self.val_rms,
        }

        with open(os.path.join(checkpoint_path, "learner_dict.pkl"), "wb") as f:
            pickle.dump(learner_dict, f)

    def load_checkpoint(self, checkpoint_path: str):
        super().load_checkpoint(checkpoint_path)

        with open(os.path.join(checkpoint_path, "learner_dict.pkl"), "rb") as f:
            learner_dict = pickle.load(f)
            self._val_rms = learner_dict[CONST_VALUE_RMS]


class OnPolicyLearner(ReinforcementLearner):
    _sample_idxes: chex.Array
    _rollout: Rollout

    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        super().__init__(config, model_config, optimizer_config)
        self._sample_idxes = np.arange(self._update_frequency)
        self._rollout = Rollout(self._env)

    def checkpoint(self, checkpoint_path: str, exist_ok: bool = False):
        super().checkpoint(checkpoint_path, exist_ok)

        with open(os.path.join(checkpoint_path, "train_returns.pkl"), "wb") as f:
            pickle.dump(
                {
                    CONST_EPISODE_LENGTHS: self._rollout.episode_lengths,
                    CONST_EPISODIC_RETURNS: self._rollout.episodic_returns,
                },
                f,
            )
