import _pickle as pickle
import chex
import numpy as np
import os

from types import SimpleNamespace
from typing import Any, Dict, Union

from jaxl.constants import *
from jaxl.envs.rollouts import Rollout
from jaxl.learners.learner import OnlineLearner
from jaxl.utils import RunningMeanStd


class ReinforcementLearner(OnlineLearner):
    _update_frequency: int
    _gamma: float
    _value_rms: Union[bool, RunningMeanStd]
    _obs_rms: Union[bool, RunningMeanStd]

    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        super().__init__(config, model_config, optimizer_config)
        self._update_frequency = config.buffer_config.buffer_size
        self._gamma = config.gamma

        self._obs_rms = False
        if getattr(config, CONST_OBS_RMS, False):
            self._obs_rms = RunningMeanStd(shape=self._env.observation_space.shape)

        self._val_rms = False
        if getattr(config, CONST_VALUE_RMS, False):
            self._val_rms = RunningMeanStd(shape=self._env.reward_dim)

    @property
    def obs_rms(self):
        return self._obs_rms

    @property
    def val_rms(self):
        return self._val_rms

    @property
    def update_frequency(self):
        return self._update_frequency

    def checkpoint(self, checkpoint_path: str, exist_ok: bool = False):
        super().checkpoint(checkpoint_path, exist_ok)
        learner_dict = {
            CONST_OBS_RMS: self.obs_rms,
            CONST_VALUE_RMS: self.val_rms,
        }

        with open(os.path.join(checkpoint_path, "learner_dict.pkl"), "wb") as f:
            pickle.dump(learner_dict, f)

    def load_checkpoint(self, checkpoint_path: str):
        super().load_checkpoint(checkpoint_path)

        with open(os.path.join(checkpoint_path, "learner_dict.pkl"), "rb") as f:
            learner_dict = pickle.load(f)
            self._obs_rms = learner_dict[CONST_OBS_RMS]
            self._val_rms = learner_dict[CONST_VALUE_RMS]

    def gather_rms(self, aux: Dict[str, Any]):
        if self.obs_rms:
            max_norm_mean_idx = np.argmax(np.abs(self.obs_rms.mean))
            max_norm_var_idx = np.argmax(np.abs(self.obs_rms.var))
            aux[CONST_LOG][f"{CONST_OBS_RMS}/mean_max_norm-mean"] = self.obs_rms.mean[max_norm_mean_idx]
            aux[CONST_LOG][f"{CONST_OBS_RMS}/mean_max_norm-var"] = self.obs_rms.var[max_norm_mean_idx]
            aux[CONST_LOG][f"{CONST_OBS_RMS}/mean_max_norm-idx"] = max_norm_mean_idx
            aux[CONST_LOG][f"{CONST_OBS_RMS}/var_max_norm-mean"] = self.obs_rms.mean[max_norm_var_idx]
            aux[CONST_LOG][f"{CONST_OBS_RMS}/var_max_norm-var"] = self.obs_rms.var[max_norm_var_idx]
            aux[CONST_LOG][f"{CONST_OBS_RMS}/var_max_norm-idx"] = max_norm_var_idx

        if self.val_rms:
            aux[CONST_LOG][f"{CONST_VALUE_RMS}-mean"] = self.val_rms.mean.item()
            aux[CONST_LOG][f"{CONST_VALUE_RMS}-var"] = self.val_rms.var.item()

        return aux


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
