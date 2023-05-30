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
    """
    Reinforcement learner class that extends the ``OnlineLearner`` class.
    This is the general learner for reinforcement learning agents.
    """

    _update_frequency: int
    _num_steps_per_epoch: int
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
        self._num_steps_per_epoch = config.num_steps_per_epoch
        self._gamma = config.gamma

        self._obs_rms = False
        if getattr(config, CONST_OBS_RMS, False):
            self._obs_rms = RunningMeanStd(shape=self._env.observation_space.shape)

        self._val_rms = False
        if getattr(config, CONST_VALUE_RMS, False):
            self._val_rms = RunningMeanStd(shape=self._env.reward_dim)

    @property
    def policy(self):
        """
        Policy.
        """
        raise NotImplementedError

    @property
    def policy_params(self):
        """
        Policy parameters.
        """
        raise NotImplementedError

    @property
    def obs_rms(self):
        """
        Running statistics for observations.
        """
        return self._obs_rms

    @property
    def val_rms(self):
        """
        Running statistics for value estimates.
        """
        return self._val_rms

    @property
    def update_frequency(self):
        """
        The frequency to perform model update after environment steps.
        """
        return self._update_frequency

    def checkpoint(self, checkpoint_path: str, exist_ok: bool = False):
        """
        Saves the current model state.

        :param checkpoint_path: directory path to store the checkpoint to
        :param exist_ok: whether to overwrite the existing checkpoint path
        :type checkpoint_path: str
        :type exist_ok: bool (Default value = False)

        """
        super().checkpoint(checkpoint_path, exist_ok)
        learner_dict = {
            CONST_OBS_RMS: self.obs_rms,
            CONST_VALUE_RMS: self.val_rms,
        }

        with open(os.path.join(checkpoint_path, "learner_dict.pkl"), "wb") as f:
            pickle.dump(learner_dict, f)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Loads a model state from a saved checkpoint.

        :param checkpoint_path: directory path to load the checkpoint from
        :type checkpoint_path: str

        """
        super().load_checkpoint(checkpoint_path)

        with open(os.path.join(checkpoint_path, "learner_dict.pkl"), "rb") as f:
            learner_dict = pickle.load(f)
            self._obs_rms = learner_dict[CONST_OBS_RMS]
            self._val_rms = learner_dict[CONST_VALUE_RMS]

    def update_value_rms_and_normalize(self, rets: chex.Array) -> chex.Array:
        """
        Updates the running statistics for value estimate
        and normalize the returns.

        :param rets: the returns for updating the statistics
        :type rets: chex.Array
        :return: the normalized returns
        :rtype: chex.Array

        """
        if self.val_rms:
            self.val_rms.update(rets)
            rets = self.val_rms.normalize(rets)
        return rets

    def update_obs_rms_and_normalize(
        self, obss: chex.Array, lengths: chex.Array
    ) -> chex.Array:
        """
        Updates the running statistics for observations
        and normalize the observations.
        If the observations are stacked with previous timesteps,
        then update using only the current timestep.

        :param obss: the observations for updating the statistics
        :param lengths: the lengths of each observation sequence
        :type obss: chex.Array
        :type lengths: chex.Array
        :return: the normalized observations
        :rtype: chex.Array

        """
        if self.obs_rms:
            idxes = lengths.reshape((-1, *[1 for _ in range(obss.ndim - 1)])) - 1
            update_obss = np.take_along_axis(obss, idxes, 1)
            self.obs_rms.update(update_obss)
            obss = self.obs_rms.normalize(obss)
        return obss

    def gather_rms(self, aux: Dict[str, Any]) -> Dict[str, Any]:
        """
        Logs the running statistics of observations and value estimate.

        :param aux: the auxiliary dictionary to put the runnint statistics into
        :type aux: Dict[str, Any]
        :return: the auxiliary dictionary with running statistics
        :rtype: Dict[str, Any]
        """
        if self.obs_rms:
            max_norm_mean_idx = np.argmax(np.abs(self.obs_rms.mean))
            max_norm_var_idx = np.argmax(np.abs(self.obs_rms.var))
            aux[CONST_LOG][f"{CONST_OBS_RMS}/mean_max_norm-mean"] = self.obs_rms.mean[
                max_norm_mean_idx
            ]
            aux[CONST_LOG][f"{CONST_OBS_RMS}/mean_max_norm-var"] = self.obs_rms.var[
                max_norm_mean_idx
            ]
            aux[CONST_LOG][f"{CONST_OBS_RMS}/mean_max_norm-idx"] = max_norm_mean_idx
            aux[CONST_LOG][f"{CONST_OBS_RMS}/var_max_norm-mean"] = self.obs_rms.mean[
                max_norm_var_idx
            ]
            aux[CONST_LOG][f"{CONST_OBS_RMS}/var_max_norm-var"] = self.obs_rms.var[
                max_norm_var_idx
            ]
            aux[CONST_LOG][f"{CONST_OBS_RMS}/var_max_norm-idx"] = max_norm_var_idx

        if self.val_rms:
            aux[CONST_LOG][f"{CONST_VALUE_RMS}/mean"] = self.val_rms.mean.item()
            aux[CONST_LOG][f"{CONST_VALUE_RMS}/var"] = self.val_rms.var.item()

        return aux


class OnPolicyLearner(ReinforcementLearner):
    """
    On-policy learner class that extends the ``ReinforcementLearner`` class.
    This is the general learner for on-policy reinforcement learning agents.
    """

    _num_update_steps: int
    _sample_idxes: chex.Array
    _rollout: Rollout

    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        super().__init__(config, model_config, optimizer_config)
        assert (
            self._num_steps_per_epoch % self._update_frequency == 0
        ), "num_steps_per_epoch {} should be divisible by update_frequency {} for on-policy algorithms".format(
            self._num_steps_per_epoch, self._update_frequency
        )
        self._num_update_steps = self._num_steps_per_epoch // self._update_frequency
        self._sample_idxes = np.arange(self._update_frequency)
        self._rollout = Rollout(self._env)

    def checkpoint(self, checkpoint_path: str, exist_ok: bool = False):
        """
        Saves the current model state.

        :param checkpoint_path: directory path to store the checkpoint to
        :param exist_ok: whether to overwrite the existing checkpoint path
        :type checkpoint_path: str
        :type exist_ok: bool (Default value = False)

        """
        super().checkpoint(checkpoint_path, exist_ok)

        with open(os.path.join(checkpoint_path, "train_returns.pkl"), "wb") as f:
            pickle.dump(
                {
                    CONST_EPISODE_LENGTHS: self._rollout.episode_lengths,
                    CONST_EPISODIC_RETURNS: self._rollout.episodic_returns,
                },
                f,
            )
