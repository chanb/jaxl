from gymnasium import spaces
from skimage.transform import resize
from tqdm import tqdm
from typing import Any, Dict, Union

import jax.random as jrandom
import math
import numpy as np
import optax

from jaxl.buffers import ReplayBuffer
from jaxl.envs.rollouts import EvaluationRollout
from jaxl.envs.wrappers.wrapper import DefaultGymWrapper
from jaxl.models import Policy
from jaxl.utils import RunningMeanStd


class MetaWorldRollout(EvaluationRollout):
    def __init__(self, env: DefaultGymWrapper, num_scrambling_steps: int=10, seed: int = 0):
        super().__init__(env, seed)
        self.num_scrambling_steps = num_scrambling_steps

    def rollout_with_subsampling(
        self,
        params: Union[optax.Params, Dict[str, Any]],
        policy: Policy,
        obs_rms: Union[bool, RunningMeanStd],
        buffer: ReplayBuffer,
        num_samples: int,
        subsampling_length: int,
        max_episode_length: int = None,
        use_image_for_inference: bool = False,
        get_image: bool = False,
        width: int = 84,
        height: int = 84,
        use_tqdm: bool = True,
    ):
        """
        Executes the policy in the environment and store them with a subsampling scheme.

        :param params: the model parameters
        :param policy: the policy
        :param obs_rms: the running statistics for observations
        :param buffer: the buffer to store the transitions with
        :param num_sampels: the number of samples to store
        :param subsampling_length: the length of the subtrajectory per episode
        :param max_episode_length: the maximum episode length
        :param get_image: whether or not to get image observation
        :param width: width of the image
        :param height: height of the image
        :param use_tqdm: whether or not to show progress bar
        :type params: Union[optax.Params, Dict[str, Any]]
        :type policy: Policy
        :type obs_rms: Union[bool, RunningMeanStd]
        :type buffer: ReplayBuffer
        :type num_samples: int
        :type subsampling_length: int
        :type max_episode_length: int (DefaultValue = None)
        :type get_image: bool (DefaultValue = False)
        :type width: int (DefaultValue = 84)
        :type height: int (DefaultValue = 84)
        :type use_tqdm: bool (DefaultValue = False)

        """
        num_episodes = math.ceil(num_samples / subsampling_length)

        termination_steps = None
        if max_episode_length is not None and max_episode_length > subsampling_length:
            termination_steps = jrandom.randint(
                jrandom.split(self._reset_key, 1)[0],
                (num_episodes,),
                subsampling_length,
                max_episode_length,
            )

        it = range(num_episodes)
        if use_tqdm:
            it = tqdm(it)
        for ep_i in it:
            termination_step = None
            if termination_steps is not None:
                termination_step = termination_steps[ep_i]
            self._episodic_returns.append(0)
            self._episode_lengths.append(0)
            seed = int(jrandom.randint(self._reset_key, (1,), 0, 2**16 - 1))
            self._reset_key = jrandom.split(self._reset_key, 1)[0]
            self._curr_obs, self._curr_info = self._env.reset(seed=seed)

            self._env.action_space.seed(seed)
            for _ in range(self.num_scrambling_steps):
                self._curr_obs, _, _, _, self._curr_info = self._env.step(self._env.action_space.sample())

            self._curr_h_state = policy.reset()

            save_curr_obs = self._curr_obs
            if get_image:
                save_curr_obs = np.transpose(resize(
                    self._env.render(),
                    (height, width)
                ), axes=(2, 0, 1))

            curr_episode = []
            done = False
            while not done:
                if use_image_for_inference:
                    self._curr_obs = save_curr_obs
                normalize_obs = np.array([self._curr_obs])
                if obs_rms:
                    normalize_obs = obs_rms.normalize(normalize_obs)

                act, next_h_state = policy.deterministic_action(
                    params,
                    normalize_obs,
                    np.array([self._curr_h_state]),
                )
                act = act[0]
                next_h_state = next_h_state[0]

                if isinstance(self._env.action_space, spaces.Box):
                    env_act = np.clip(
                        act, self._env.action_space.low, self._env.action_space.high
                    )
                env_act = np.array(act)
                next_obs, rew, terminated, truncated, info = self._env.step(env_act)
                self._episodic_returns[-1] += float(rew)
                self._episode_lengths[-1] += 1

                done = terminated or truncated
                if termination_step is not None:
                    done = done or len(curr_episode) == termination_step

                save_next_obs = next_obs
                if get_image:
                    save_next_obs = np.transpose(resize(
                        self._env.render(),
                        (height, width)
                    ), axes=(2, 0, 1))

                curr_episode.append(
                    (
                        save_curr_obs,
                        self._curr_h_state,
                        act,
                        rew,
                        terminated,
                        truncated,
                        info,
                        save_next_obs,
                        next_h_state,
                    )
                )

                save_curr_obs = save_next_obs
                self._curr_obs = next_obs
                self._curr_h_state = next_h_state

            idx_start = 0
            idx_end = len(curr_episode)
            if termination_step is not None:
                idx_start = termination_step - subsampling_length
                idx_end = termination_step
            else:
                if subsampling_length < len(curr_episode):
                    idx_start = jrandom.randint(
                        self._reset_key,
                        (),
                        minval=0,
                        maxval=len(curr_episode) - subsampling_length,
                    )
                    idx_end = idx_start + subsampling_length

            for idx in range(idx_start, idx_end):
                buffer.push(*curr_episode[idx])
                if buffer.is_full:
                    break

        self._env.reset()