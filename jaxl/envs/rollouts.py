from abc import ABC, abstractclassmethod
from gymnasium import spaces
from tqdm import tqdm
from typing import Any, Dict, Iterable, Tuple, Union

import chex
import jax.random as jrandom
import math
import numpy as np
import optax

from jaxl.buffers import ReplayBuffer
from jaxl.envs.wrappers.wrapper import DefaultGymWrapper
from jaxl.models import Policy
from jaxl.utils import RunningMeanStd


class Rollout(ABC):
    """
    Interconnection between policy and environment.
    This executes the provided policy in the specified environment.
    """

    #: The environment.
    _env: DefaultGymWrapper

    #: The current observation.
    _curr_obs: chex.Array

    #: The current hidden state.
    _curr_h_state: chex.Array

    #: Episode lengths.
    _episode_lengths: Iterable

    #: Episodic returns
    _episodic_returns: Iterable

    #: Whether or not the current trajectory is done (terminated or truncated).
    _done: bool

    def __init__(self, env: DefaultGymWrapper):
        self._env = env
        self._curr_obs = None
        self._curr_h_state = None
        self._episodic_returns = []
        self._episode_lengths = []
        self._done = True

    @abstractclassmethod
    def rollout(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @property
    def episodic_returns(self):
        """All episodic returns."""
        return self._episodic_returns

    @property
    def episode_lengths(self):
        """All episode lengths."""
        return self._episode_lengths

    @property
    def latest_return(self):
        """Latest episodic return."""
        if self._done:
            return self._episodic_returns[-1]
        if len(self._episodic_returns) > 2:
            return self._episodic_returns[-2]
        return 0

    @property
    def latest_episode_length(self):
        """Latest episode length."""
        if self._done:
            return self._episode_lengths[-1]
        if len(self._episode_lengths) > 2:
            return self._episode_lengths[-2]
        return 0

    def latest_average_return(self, num_episodes: int = 5) -> chex.Array:
        """
        Gets the average return of the last few episodes

        :param num_episodes: the number of episodes to smooth over.
        :type params: int:  (Default Value = 5)
        :return: the average return over the last `num_episodes` episodes
        :rtype: chex.Array

        """
        latest_returns = self.episodic_returns[-num_episodes - 1 :]
        if self._done:
            latest_returns = latest_returns[1:]
        else:
            latest_returns = latest_returns[:-1]
        return np.mean(latest_returns)

    def latest_average_episode_length(self, num_episodes: int = 5) -> chex.Array:
        """
        Gets the average episode length of the last few episodes

        :param num_episodes: the number of episodes to smooth over.
        :type params: int int:  (Default Value = 5)
        :return: the average episode length over the last `num_episodes` episodes
        :rtype: chex.Array

        """
        latest_episode_lengths = self.episode_lengths[-num_episodes - 1 :]
        if self._done:
            latest_episode_lengths = latest_episode_lengths[1:]
        else:
            latest_episode_lengths = latest_episode_lengths[:-1]
        return np.mean(latest_episode_lengths)


class EvaluationRollout(Rollout):
    """
    Interconnection between policy and environment.
    This executes the provided policy in the specified environment
    without any exploration. That is, it uses `deterministic_action`,
    which is usually implemented as the most-likely action.
    """

    def __init__(self, env: DefaultGymWrapper, seed: int = 0):
        super().__init__(env)
        self._reset_key = jrandom.split(jrandom.PRNGKey(seed), 1)[0]

    def rollout(
        self,
        params: Union[optax.Params, Dict[str, Any]],
        policy: Policy,
        obs_rms: Union[bool, RunningMeanStd],
        num_episodes: int,
        buffer: ReplayBuffer = None,
        use_tqdm: bool = True,
    ):
        """
        Executes the policy in the environment.

        :param params: the model parameters
        :param policy: the policy
        :param obs_rms: the running statistics for observations
        :param num_episodes: the number of interaction episodes with the environment
        :param buffer: the buffer to store the transitions with
        :param use_tqdm: whether or not to show progress bar
        :type params: Union[optax.Params, Dict[str, Any]]
        :type policy: Policy
        :type obs_rms: Union[bool, RunningMeanStd]
        :type num_episodes: int
        :type buffer: ReplayBuffer (DefaultValue = None)
        :type use_tqdm: bool (DefaultValue = False)

        """
        it = range(num_episodes)
        if use_tqdm:
            it = tqdm(it)
        for _ in it:
            self._episodic_returns.append(0)
            self._episode_lengths.append(0)
            seed = int(jrandom.randint(self._reset_key, (1,), 0, 2**16 - 1))
            self._reset_key = jrandom.split(self._reset_key, 1)[0]
            self._curr_obs, self._curr_info = self._env.reset(seed=seed)
            self._curr_h_state = policy.reset()

            done = False
            while not done:
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

                env_act = act
                if isinstance(self._env.action_space, spaces.Box):
                    env_act = np.clip(
                        act, self._env.action_space.low, self._env.action_space.high
                    )
                next_obs, rew, terminated, truncated, info = self._env.step(env_act)
                self._episodic_returns[-1] += float(rew)
                self._episode_lengths[-1] += 1

                done = terminated or truncated

                if buffer is not None:
                    buffer.push(
                        self._curr_obs,
                        self._curr_h_state,
                        act,
                        rew,
                        terminated,
                        truncated,
                        info,
                        next_obs,
                        next_h_state,
                    )

                self._curr_obs = next_obs
                self._curr_h_state = next_h_state
        self._env.reset()

    def random_sample_rollout(
        self,
        num_episodes: int,
        buffer: ReplayBuffer = None,
        use_tqdm: bool = True,
    ):
        """
        Executes a random policy in the environment.

        :param num_episodes: the number of interaction episodes with the environment
        :param buffer: the buffer to store the transitions with
        :param use_tqdm: whether or not to show progress bar
        :type num_episodes: int
        :type buffer: ReplayBuffer (DefaultValue = None)
        :type use_tqdm: bool (DefaultValue = False)

        """
        it = range(num_episodes)
        if use_tqdm:
            it = tqdm(it)
        for _ in it:
            self._episodic_returns.append(0)
            self._episode_lengths.append(0)
            seed = int(jrandom.randint(self._reset_key, (1,), 0, 2**16 - 1))
            self._reset_key = jrandom.split(self._reset_key, 1)[0]
            self._curr_obs, self._curr_info = self._env.reset(seed=seed)
            self._curr_h_state = np.ones((1,))

            done = False
            while not done:
                act = self._env.action_space.sample()
                next_h_state = self._curr_h_state
                env_act = act
                next_obs, rew, terminated, truncated, info = self._env.step(env_act)
                self._episodic_returns[-1] += float(rew)
                self._episode_lengths[-1] += 1

                done = terminated or truncated

                if buffer is not None:
                    buffer.push(
                        self._curr_obs,
                        self._curr_h_state,
                        act,
                        rew,
                        terminated,
                        truncated,
                        info,
                        next_obs,
                        next_h_state,
                    )

                self._curr_obs = next_obs
                self._curr_h_state = next_h_state
        self._env.reset()

    def rollout_with_subsampling(
        self,
        params: Union[optax.Params, Dict[str, Any]],
        policy: Policy,
        obs_rms: Union[bool, RunningMeanStd],
        buffer: ReplayBuffer,
        num_samples: int,
        subsampling_length: int,
        max_episode_length: int = None,
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
        :param use_tqdm: whether or not to show progress bar
        :type params: Union[optax.Params, Dict[str, Any]]
        :type policy: Policy
        :type obs_rms: Union[bool, RunningMeanStd]
        :type buffer: ReplayBuffer
        :type num_samples: int
        :type subsampling_length: int
        :type max_episode_length: int (DefaultValue = None)
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
            self._curr_h_state = policy.reset()

            curr_episode = []
            done = False
            while not done:
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

                env_act = act
                if isinstance(self._env.action_space, spaces.Box):
                    env_act = np.clip(
                        act, self._env.action_space.low, self._env.action_space.high
                    )
                next_obs, rew, terminated, truncated, info = self._env.step(env_act)
                self._episodic_returns[-1] += float(rew)
                self._episode_lengths[-1] += 1

                done = terminated or truncated
                if termination_step is not None:
                    done = done or len(curr_episode) == termination_step

                curr_episode.append(
                    (
                        self._curr_obs,
                        self._curr_h_state,
                        act,
                        rew,
                        terminated,
                        truncated,
                        info,
                        next_obs,
                        next_h_state,
                    )
                )

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


class StandardRollout(Rollout):
    """
    Interconnection between policy and environment.
    This executes the provided policy in the specified environment
    using `compute_action`.
    """

    def __init__(self, env: DefaultGymWrapper, seed: int = 0):
        super().__init__(env)
        self._reset_key, self._exploration_key = jrandom.split(jrandom.PRNGKey(seed))

    def rollout(
        self,
        params: Union[optax.Params, Dict[str, Any]],
        policy: Policy,
        obs_rms: Union[bool, RunningMeanStd],
        buffer: ReplayBuffer,
        num_steps: int,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Executes the policy in the environment.

        :param params: the model parameters
        :param policy: the policy
        :param obs_rms: the running statistics for observations
        :param buffer: the buffer
        :param num_steps: the number of interactions to have with the environment
        :type params: Union[optax.Params, Dict[str, Any]]
        :type policy: Policy
        :type obs_rms: Union[bool, RunningMeanStd]
        :type buffer: ReplayBuffer
        :type num_steps: int
        :return: the current observation and the current hidden state
        :rtype: Tuple[chex.Array, chex.Array]

        """
        for _ in range(num_steps):
            if self._done:
                self._done = False
                self._episodic_returns.append(0)
                self._episode_lengths.append(0)
                seed = int(jrandom.randint(self._reset_key, (1,), 0, 2**16 - 1))
                self._reset_key = jrandom.split(self._reset_key, 1)[0]
                self._curr_obs, self._curr_info = self._env.reset(seed=seed)
                self._curr_h_state = policy.reset()

            normalize_obs = np.array([self._curr_obs])
            if obs_rms:
                normalize_obs = obs_rms.normalize(normalize_obs)

            act, next_h_state = policy.compute_action(
                params,
                normalize_obs,
                np.array([self._curr_h_state]),
                self._exploration_key,
            )
            act = act[0]
            next_h_state = next_h_state[0]
            self._exploration_key = jrandom.split(self._exploration_key, 1)[0]

            env_act = act
            if isinstance(self._env.action_space, spaces.Box):
                env_act = np.clip(
                    act, self._env.action_space.low, self._env.action_space.high
                )
            next_obs, rew, terminated, truncated, info = self._env.step(env_act)

            self._episodic_returns[-1] += float(rew)
            self._episode_lengths[-1] += 1

            self._done = terminated or truncated

            rew = info.get("shaped_reward", rew)

            buffer.push(
                self._curr_obs,
                self._curr_h_state,
                act,
                rew,
                terminated,
                truncated,
                info,
                next_obs,
                next_h_state,
            )
            self._curr_obs = next_obs
            self._curr_h_state = next_h_state
        return self._curr_obs, self._curr_h_state
