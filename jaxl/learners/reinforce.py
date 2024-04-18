from types import SimpleNamespace
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import timeit

from jaxl.constants import *
from jaxl.learners.reinforcement import OnPolicyLearner
from jaxl.losses.reinforcement import monte_carlo_returns, make_reinforce_loss
from jaxl.models import (
    get_model,
    get_policy,
    get_update_function,
    policy_output_dim,
)
from jaxl.optimizers import get_optimizer
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

        model_key = jrandom.PRNGKey(self._config.seeds.model_seed)
        dummy_x = self._generate_dummy_x(input_dim)
        params = self._model.init(model_key, dummy_x)

        self._optimizer, opt_state = get_optimizer(
            self._optimizer_config, self._model, params
        )

        self._model_dict = {
            CONST_MODEL: {CONST_POLICY: params},
            CONST_OPT_STATE: {CONST_POLICY: opt_state},
        }

    @property
    def policy(self):
        """
        Policy.
        """
        return self._policy

    @property
    def policy_params(self):
        """
        Policy parameters.
        """
        return self._model_dict[CONST_MODEL][CONST_POLICY]

    def make_policy_step(self):
        """
        Makes the training step for policy update.
        """

        update_function = get_update_function(self._model)

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

            params, opt_state = update_function(
                self._optimizer,
                grads,
                model_dict[CONST_OPT_STATE][CONST_POLICY],
                model_dict[CONST_MODEL][CONST_POLICY],
                aux[CONST_AUX][CONST_UPDATES],
            )

            aux[CONST_GRAD_NORM] = {CONST_POLICY: l2_norm(grads)}
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
        total_rollout_time = 0
        total_update_time = 0

        carried_steps = self._global_step % self._update_frequency
        num_update_steps = (
            self._num_steps_per_epoch + carried_steps
        ) // self._update_frequency

        for update_i in range(num_update_steps):
            auxes.append({})
            tic = timeit.default_timer()
            if update_i > 0:
                step_count = self._update_frequency
            else:
                step_count = self._update_frequency - carried_steps

            self._rollout.rollout(
                self._model_dict[CONST_MODEL][CONST_POLICY],
                self._policy,
                self._obs_rms,
                self._buffer,
                step_count,
            )
            self._global_step += step_count
            total_rollout_time += timeit.default_timer() - tic

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
            total_update_time += timeit.default_timer() - tic
            assert np.isfinite(aux[CONST_AGG_LOSS]), f"Loss became NaN\naux: {aux}"

            auxes[-1][CONST_AUX] = aux
            auxes[-1][CONST_ACTION] = {
                i: {
                    CONST_SATURATION: np.abs(acts[..., i]).max(),
                    CONST_MEAN: np.abs(acts[..., i]).mean(),
                }
                for i in range(acts.shape[-1])
            }

        auxes = jax.tree_util.tree_map(lambda *args: np.mean(args), *auxes)
        aux[CONST_LOG] = {
            f"losses/{CONST_AGG_LOSS}": auxes[CONST_AUX][CONST_AGG_LOSS].item(),
            f"{CONST_GRAD_NORM}/pi": auxes[CONST_AUX][CONST_GRAD_NORM][
                CONST_POLICY
            ].item(),
            f"losses_info/pi_log_prob": auxes[CONST_AUX][CONST_LOG_PROBS].item(),
            f"{CONST_PARAM_NORM}/pi": l2_norm(
                self.model_dict[CONST_MODEL][CONST_POLICY]
            ).item(),
            f"interaction/{CONST_AVERAGE_RETURN}": self._rollout.latest_average_return(),
            f"interaction/{CONST_AVERAGE_EPISODE_LENGTH}": self._rollout.latest_average_episode_length(),
            f"time/{CONST_ROLLOUT_TIME}": total_rollout_time,
            f"time/{CONST_UPDATE_TIME}": total_update_time,
        }

        for act_i in range(acts.shape[-1]):
            aux[CONST_LOG][
                f"{CONST_ACTION}/{CONST_ACTION}_{act_i}_{CONST_SATURATION}"
            ] = auxes[CONST_ACTION][act_i][CONST_SATURATION]
            aux[CONST_LOG][f"{CONST_ACTION}/{CONST_ACTION}_{act_i}_{CONST_MEAN}"] = (
                auxes[CONST_ACTION][act_i][CONST_MEAN]
            )

        self.gather_rms(aux)
        return aux
