from flax.core.scope import FrozenVariableDict
from typing import Any, Dict, Tuple, Union

import chex
import jax.random as jrandom
import numpy as np

from jaxl.buffers import ReplayBuffer
from jaxl.envs.wrappers.wrapper import DefaultGymWrapper
from jaxl.models import Policy
from jaxl.utils import RunningMeanStd


class Rollout:
    def __init__(self, env: DefaultGymWrapper, seed: int = 0):
        self._env = env
        self._curr_obs = None
        self._curr_h_state = None
        self._curr_info = None
        self._done = True
        self._episodic_returns = []
        self._episode_lengths = []
        self._reset_key, self._exploration_key = jrandom.split(jrandom.PRNGKey(seed))

    def rollout(
        self,
        params: Union[FrozenVariableDict, Dict[str, Any]],
        policy: Policy,
        obs_rms: Union[bool, RunningMeanStd],
        buffer: ReplayBuffer,
        num_steps: int,
    ) -> Tuple[chex.Array, chex.Array]:
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
            self._exploration_key = jrandom.split(self._exploration_key, 1)[0]

            env_act = np.clip(
                act, self._env.action_space.low, self._env.action_space.high
            )
            next_obs, rew, terminated, truncated, info = self._env.step(env_act)
            self._episodic_returns[-1] += float(rew)
            self._episode_lengths[-1] += 1

            self._done = terminated or truncated

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

    @property
    def episodic_returns(self):
        return self._episodic_returns

    @property
    def episode_lengths(self):
        return self._episode_lengths

    @property
    def latest_return(self):
        if self._done:
            return self._episodic_returns[-1]
        if len(self._episodic_returns) > 2:
            return self._episodic_returns[-2]
        return 0

    @property
    def latest_episode_length(self):
        if self._done:
            return self._episode_lengths[-1]
        if len(self._episode_lengths) > 2:
            return self._episode_lengths[-2]
        return 0
