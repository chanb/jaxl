from types import SimpleNamespace
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import timeit

from jaxl.constants import *
from jaxl.learners.reinforcement import OnPolicyLearner
from jaxl.losses.reinforcement import monte_carlo_returns, make_reinforce_loss
from jaxl.models import (
    get_model,
    get_optimizer,
    get_policy,
    policy_output_dim,
)
from jaxl.utils import l2_norm


"""
Standard REINFORCE.
"""


class REINFORCE(OnPolicyLearner):
    """
    REINFORCE algorithm. This extends `OnPolicyLearner`.
    """

    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        super().__init__(config, model_config, optimizer_config)

        self._policy = get_policy(self._model, config)
        self._loss = make_reinforce_loss(self._policy, self._config.pi_loss_setting)
        self.policy_step = jax.jit(self.make_policy_step())

    def _initialize_model_and_opt(self, input_dim: chex.Array, output_dim: chex.Array):
        """
        Construct the policy and the optimizer.

        :param input_dim: input dimension of the data point
        :param output_dim: output dimension of the data point
        :type input_dim: chex.Array
        :type output_dim: chex.Array

        """
        output_dim = policy_output_dim(output_dim, self._config)
        self._model = get_model(input_dim, output_dim, self._model_config)
        self._optimizer = get_optimizer(self._optimizer_config)

        model_key = jrandom.PRNGKey(self._config.seeds.model_seed)
        dummy_x = self._generate_dummy_x()
        params = self._model.init(model_key, dummy_x)
        opt_state = self._optimizer.init(params)
        self._model_dict = {
            CONST_MODEL: {CONST_POLICY: params},
            CONST_OPT_STATE: {CONST_POLICY: opt_state},
        }

    def make_policy_step(self):
        """
        Makes the training step for policy update.
        """

        def _policy_step(
            model_dict: Dict[str, Any],
            obss: chex.Array,
            h_states: chex.Array,
            acts: chex.Array,
            rets: chex.Array,
            *args,
            **kwargs,
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            """
            The training step that computes the REINFORCE loss and performs policy update.

            :param model_dict: the model state and optimizer state
            :param obss: the training observations
            :param h_states: the training hidden states for memory-based models
            :param acts: the training actions
            :param rets: the Monte-Carlo returns
            :param *args:
            :param **kwargs:
            :type model_dict: Dict[str, Any]
            :type obss: chex.Array
            :type h_states: chex.Array
            :type acts: chex.Array
            :type rets: chex.Array
            :return: the updated model state and optimizer state, and auxiliary information
            :rtype: Tuple[Dict[str, Any], Dict[str, Any]]

            """
            (agg_loss, aux), grads = jax.value_and_grad(self._loss, has_aux=True)(
                model_dict[CONST_MODEL][CONST_POLICY],
                obss,
                h_states,
                acts,
                rets,
                jnp.zeros(rets.shape),
            )
            aux[CONST_AGG_LOSS] = agg_loss
            updates, opt_state = self._optimizer.update(
                grads,
                model_dict[CONST_OPT_STATE][CONST_POLICY],
                model_dict[CONST_MODEL][CONST_POLICY],
            )
            aux[CONST_GRAD_NORM] = {CONST_POLICY: l2_norm(grads)}
            params = optax.apply_updates(model_dict[CONST_MODEL][CONST_POLICY], updates)
            return {
                CONST_MODEL: {CONST_POLICY: params},
                CONST_OPT_STATE: {CONST_POLICY: opt_state},
            }, aux

        return _policy_step

    def update(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Updates the policy.

        :param *args:
        :param **kwargs:
        :return: the update information
        :rtype: Dict[str, Any]

        """

        auxes = []
        for _ in range(self._num_update_steps):
            auxes.append({})
            tic = timeit.default_timer()
            self._rollout.rollout(
                self._model_dict[CONST_MODEL][CONST_POLICY],
                self._policy,
                self._obs_rms,
                self._buffer,
                self._update_frequency,
            )
            rollout_time = timeit.default_timer() - tic

            tic = timeit.default_timer()
            (
                obss,
                h_states,
                acts,
                rews,
                dones,
                _,
                _,
                _,
                lengths,
                _,
            ) = self._buffer.sample(
                batch_size=self._update_frequency, idxes=self._sample_idxes
            )

            obss = self.update_obs_rms_and_normalize(obss, lengths)

            rets = monte_carlo_returns(rews, dones, self._gamma)
            rets = self.update_value_rms_and_normalize(rets)

            self.model_dict, aux = self.policy_step(
                self._model_dict, obss, h_states, acts, rets
            )
            update_time = timeit.default_timer() - tic
            assert np.isfinite(aux[CONST_AGG_LOSS]), f"Loss became NaN\naux: {aux}"

            auxes[-1][CONST_ROLLOUT_TIME] = rollout_time
            auxes[-1][CONST_UPDATE_TIME] = update_time
            auxes[-1][CONST_AUX] = aux

        auxes = jax.tree_util.tree_map(lambda *args: np.mean(args), *auxes)
        aux[CONST_LOG] = {
            f"losses/{CONST_AGG_LOSS}": auxes[CONST_AUX][CONST_AGG_LOSS].item(),
            f"{CONST_GRAD_NORM}/pi": auxes[CONST_AUX][CONST_GRAD_NORM][
                CONST_POLICY
            ].item(),
            f"{CONST_PARAM_NORM}/pi": l2_norm(
                self.model_dict[CONST_MODEL][CONST_POLICY]
            ).item(),
            f"interaction/{CONST_AVERAGE_RETURN}": self._rollout.latest_average_return(),
            f"interaction/{CONST_AVERAGE_EPISODE_LENGTH}": self._rollout.latest_average_episode_length(),
            f"time/{CONST_ROLLOUT_TIME}": auxes[CONST_ROLLOUT_TIME].item(),
            f"time/{CONST_UPDATE_TIME}": auxes[CONST_UPDATE_TIME].item(),
        }

        self.gather_rms(aux)
        return aux
