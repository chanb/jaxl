from types import SimpleNamespace
from typing import Any, Dict, Tuple

import chex
import jax
import jax.random as jrandom
import numpy as np
import optax
import timeit

from jaxl.constants import *
from jaxl.learners.reinforcement import OnPolicyLearner
from jaxl.losses.reinforcement import (
    scan_gae_lambda_returns,
    make_ppo_pi_loss,
    make_ppo_vf_loss,
)
from jaxl.losses.supervised import make_squared_loss
from jaxl.models import (
    get_model,
    get_optimizer,
    get_policy,
    policy_output_dim,
)
from jaxl.utils import l2_norm


"""
Standard PPO.
"""


class PPO(OnPolicyLearner):
    """
    Proximal Policy Optimization (PPO) algorithm. This extends `OnPolicyLearner`.
    """

    _gae_lambda: float
    _normalize_advantage: bool
    _eps: float
    _opt_epochs: int
    _opt_batch_size: int

    def __init__(
        self,
        config: SimpleNamespace,
        model_config: SimpleNamespace,
        optimizer_config: SimpleNamespace,
    ):
        super().__init__(config, model_config, optimizer_config)
        self._gae_lambda = self._config.gae_lambda
        self._normalize_advantage = self._config.normalize_advantage
        self._eps = self._config.eps
        self._opt_epochs = self._config.opt_epochs
        self._opt_batch_size = self._config.opt_batch_size
        self._sample_key = jrandom.split(
            jrandom.PRNGKey(self._config.seeds.buffer_seed)
        )[1]

        assert (
            self._opt_batch_size <= self._update_frequency
        ), "optimization batch size {} cannot be larger than update frequency {}".format(
            self._opt_batch_size, self._update_frequency
        )

        self._pi = get_policy(self._model[CONST_POLICY], config)
        self._vf = self._model[CONST_VF]

        self._pi_loss = make_ppo_pi_loss(self._pi, self._config.pi_loss_setting)

        if self._config.vf_loss_setting.clip_param:
            _vf_loss = make_ppo_vf_loss(self._vf, self._config.vf_loss_setting)

            def vf_loss(params, obss, h_states, rets, vals):
                return _vf_loss(params, obss, h_states, rets, vals)

        else:
            _vf_loss = make_squared_loss(self._vf, self._config.vf_loss_setting)

            def vf_loss(params, obss, h_states, rets, vals):
                return _vf_loss(params, obss, h_states, rets)

        self._vf_loss = vf_loss

        def joint_loss(
            model_dicts: Dict[str, Any],
            obss: chex.Array,
            h_states: chex.Array,
            acts: chex.Array,
            rets: chex.Array,
            advs: chex.Array,
            vals: chex.Array,
            old_lprobs: chex.Array,
            *args,
            **kwargs,
        ) -> Tuple[chex.Array, Dict]:
            """
            Aggregates the actor loss and the critic loss.

            :param model_dict: the actor and critic states and their optimizers state
            :param obss: the training observations
            :param h_states: the training hidden states for memory-based models
            :param acts: the training actions
            :param rets: the Monte-Carlo returns
            :param advs: the advantages
            :param vals: the predicted values from the critic
            :param old_lprobs: the action log probabilities
            :param *args:
            :param **kwargs:
            :type model_dict: Dict[str, Any]
            :type obss: chex.Array
            :type h_states: chex.Array
            :type acts: chex.Array
            :type rets: chex.Array
            :types advs: chex.Array
            :types vals: chex.Array
            :types old_lprobs: chex.Array
            :return: the aggregate loss and auxiliary information
            :rtype: Tuple[chex.Array, Dict[str, Any]]

            """
            pi_loss, pi_aux = self._pi_loss(
                model_dicts[CONST_POLICY],
                obss,
                h_states,
                acts,
                advs,
                old_lprobs,
            )
            vf_loss, vf_aux = self._vf_loss(
                model_dicts[CONST_VF],
                obss,
                h_states,
                rets,
                vals,
            )

            agg_loss = (
                self._config.pi_loss_setting.coefficient * pi_loss
                + self._config.vf_loss_setting.coefficient * vf_loss
            )

            aux = {
                CONST_POLICY: {
                    CONST_LOSS: pi_loss,
                    CONST_NUM_CLIPPED: pi_aux[CONST_NUM_CLIPPED],
                    CONST_IS_RATIO: pi_aux[CONST_IS_RATIO],
                    CONST_LOG_PROB: pi_aux[CONST_LOG_PROB],
                },
                CONST_VF: {
                    CONST_LOSS: vf_loss,
                    CONST_NUM_CLIPPED: vf_aux.get(CONST_NUM_CLIPPED, 0),
                },
            }
            return agg_loss, aux

        self._joint_loss = joint_loss
        self.joint_step = jax.jit(self.make_joint_step())

    @property
    def policy(self):
        """
        Policy.
        """
        return self._pi

    @property
    def policy_params(self):
        """
        Policy parameters.
        """
        return self._model_dict[CONST_MODEL][CONST_POLICY]

    def _initialize_model_and_opt(self, input_dim: chex.Array, output_dim: chex.Array):
        """
        Construct the actor and critic, and their corresponding optimizers.

        :param input_dim: input dimension of the data point
        :param output_dim: output dimension of the data point
        :type input_dim: chex.Array
        :type output_dim: chex.Array

        """
        act_dim = policy_output_dim(output_dim, self._config)
        self._model = {
            CONST_POLICY: get_model(input_dim, act_dim, self._model_config.policy),
            CONST_VF: get_model(input_dim, (1,), self._model_config.vf),
        }

        model_keys = jrandom.split(jrandom.PRNGKey(self._config.seeds.model_seed))
        dummy_x = self._generate_dummy_x(input_dim)
        pi_params = self._model[CONST_POLICY].init(model_keys[0], dummy_x)
        vf_params = self._model[CONST_VF].init(model_keys[1], dummy_x)

        pi_opt, pi_opt_state = get_optimizer(
            self._optimizer_config.policy, self._model[CONST_POLICY], pi_params
        )
        vf_opt, vf_opt_state = get_optimizer(
            self._optimizer_config.vf, self._model[CONST_VF], vf_params
        )

        self._optimizer = {
            CONST_POLICY: pi_opt,
            CONST_VF: vf_opt,
        }

        self._model_dict = {
            CONST_MODEL: {
                CONST_POLICY: pi_params,
                CONST_VF: vf_params,
            },
            CONST_OPT_STATE: {
                CONST_POLICY: pi_opt_state,
                CONST_VF: vf_opt_state,
            },
        }

    def make_joint_step(self):
        """
        Makes the training step for both actor update and critic update.
        """

        def _joint_step(
            model_dict: Dict[str, Any],
            obss: chex.Array,
            h_states: chex.Array,
            acts: chex.Array,
            rets: chex.Array,
            advs: chex.Array,
            vals: chex.Array,
            old_lprobs: chex.Array,
            *args,
            **kwargs,
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            """
            The training step that computes the PPO loss and performs actor update and critic update.

            :param model_dict: the actor and critic states and their optimizers state
            :param obss: the training observations
            :param h_states: the training hidden states for memory-based models
            :param acts: the training actions
            :param rets: the Monte-Carlo returns
            :param advs: the advantages
            :param vals: the predicted values from the critic
            :param old_lprobs: the action log probabilities
            :param *args:
            :param **kwargs:
            :type model_dict: Dict[str, Any]
            :type obss: chex.Array
            :type h_states: chex.Array
            :type acts: chex.Array
            :type rets: chex.Array
            :types advs: chex.Array
            :types vals: chex.Array
            :types old_lprobs: chex.Array
            :return: the aggregate loss and auxiliary information
            :rtype: Tuple[Dict[str, Any], Dict[str, Any]]

            """
            (agg_loss, aux), grads = jax.value_and_grad(self._joint_loss, has_aux=True)(
                model_dict[CONST_MODEL],
                obss,
                h_states,
                acts,
                rets,
                advs,
                vals,
                old_lprobs,
            )
            aux[CONST_AGG_LOSS] = agg_loss
            aux[CONST_GRAD_NORM] = {
                CONST_POLICY: l2_norm(grads[CONST_POLICY]),
                CONST_VF: l2_norm(grads[CONST_VF]),
            }

            updates, pi_opt_state = self._optimizer[CONST_POLICY].update(
                grads[CONST_POLICY],
                model_dict[CONST_OPT_STATE][CONST_POLICY],
                model_dict[CONST_MODEL][CONST_POLICY],
            )
            pi_params = optax.apply_updates(
                model_dict[CONST_MODEL][CONST_POLICY], updates
            )

            updates, vf_opt_state = self._optimizer[CONST_VF].update(
                grads[CONST_VF],
                model_dict[CONST_OPT_STATE][CONST_VF],
                model_dict[CONST_MODEL][CONST_VF],
            )
            vf_params = optax.apply_updates(model_dict[CONST_MODEL][CONST_VF], updates)

            return {
                CONST_MODEL: {
                    CONST_POLICY: pi_params,
                    CONST_VF: vf_params,
                },
                CONST_OPT_STATE: {
                    CONST_POLICY: pi_opt_state,
                    CONST_VF: vf_opt_state,
                },
            }, aux

        return _joint_step

    def update(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Updates the actor and the critic.

        :param *args:
        :param **kwargs:
        :return: the update information
        :rtype: Dict[str, Any]

        """

        auxes = []
        total_rollout_time = 0
        total_update_time = 0
        for _ in range(self._num_update_steps):
            auxes.append({})
            tic = timeit.default_timer()
            next_obs, next_h_state = self._rollout.rollout(
                self._model_dict[CONST_MODEL][CONST_POLICY],
                self._pi,
                self._obs_rms,
                self._buffer,
                self._update_frequency,
            )
            total_rollout_time += timeit.default_timer() - tic

            tic = timeit.default_timer()
            (
                obss,
                h_states,
                acts,
                rews,
                _,
                dones,
                _,
                _,
                lengths,
                _,
            ) = self._buffer.sample(
                batch_size=self._update_frequency, idxes=self._sample_idxes
            )

            obss = self.update_obs_rms_and_normalize(obss, lengths)

            # Get value predictions for computing GAE return
            vals, _ = self._model[CONST_VF].forward(
                self._model_dict[CONST_MODEL][CONST_VF],
                np.concatenate((obss, np.array([[next_obs]])), axis=0),
                np.concatenate((h_states, np.array([[next_h_state]])), axis=0),
            )

            unnormalized_vals = vals
            if self.val_rms:
                unnormalized_vals = self.val_rms.unnormalize(vals)

            rets = scan_gae_lambda_returns(
                rews, unnormalized_vals, dones, self._gamma, self._gae_lambda
            )[:, None]
            rets = self.update_value_rms_and_normalize(np.array(rets))

            vals = vals[:-1]
            advs = rets - vals
            if self._normalize_advantage:
                advs = (advs - advs.mean()) / (advs.std() + self._eps)

            # Get action log probabilities for importance sampling
            old_lprobs, old_act_params = self._pi.lprob(
                self._model_dict[CONST_MODEL][CONST_POLICY], obss, h_states, acts
            )

            auxes_per_epoch = []
            curr_sample_keys = jrandom.split(self._sample_key, num=self._opt_epochs + 1)
            self._sample_key = curr_sample_keys[0]
            permutation_keys = curr_sample_keys[1:]
            sample_idxes = jax.vmap(jrandom.permutation, in_axes=[0, None])(
                permutation_keys, self._sample_idxes
            ).flatten()
            for opt_i in range(self._opt_epochs):
                minibatch_idxes = sample_idxes[
                    opt_i * self._opt_batch_size : (opt_i + 1) * self._opt_batch_size
                ]
                self.model_dict, aux = self.joint_step(
                    self._model_dict,
                    obss[minibatch_idxes],
                    h_states[minibatch_idxes],
                    acts[minibatch_idxes],
                    rets[minibatch_idxes],
                    advs[minibatch_idxes],
                    vals[minibatch_idxes],
                    old_lprobs[minibatch_idxes],
                )
                assert np.isfinite(aux[CONST_AGG_LOSS]), f"Loss became NaN\naux: {aux}"
                auxes_per_epoch.append(aux)
            total_update_time += timeit.default_timer() - tic

            auxes[-1][CONST_RETURNS] = rets.mean().item()
            auxes[-1][CONST_VALUES] = vals.mean().item()
            auxes[-1][CONST_ADVANTAGES] = advs.mean().item()

            auxes_per_epoch = jax.tree_util.tree_map(
                lambda *args: np.mean(args), *auxes_per_epoch
            )
            auxes[-1][CONST_AUX] = auxes_per_epoch
            auxes[-1][CONST_ACTION] = {
                i: {
                    CONST_SATURATION: np.abs(acts[..., i]).max(),
                    CONST_MEAN: np.abs(acts[..., i]).mean(),
                }
                for i in range(acts.shape[-1])
            }
            auxes[-1][CONST_POLICY] = {
                i: {
                    CONST_MEAN: np.abs(old_act_params[CONST_MEAN][..., i]).mean(),
                    CONST_STD: np.abs(old_act_params[CONST_STD][..., i]).mean(),
                }
                for i in range(acts.shape[-1])
            }

        auxes = jax.tree_util.tree_map(lambda *args: np.mean(args), *auxes)
        aux[CONST_LOG] = {
            f"losses/{CONST_AGG_LOSS}": auxes[CONST_AUX][CONST_AGG_LOSS].item(),
            f"losses/pi": auxes[CONST_AUX][CONST_POLICY][CONST_LOSS].item(),
            f"losses/vf": auxes[CONST_AUX][CONST_VF][CONST_LOSS].item(),
            f"losses_info/{CONST_RETURN}": auxes[CONST_RETURNS].item(),
            f"losses_info/{CONST_ADVANTAGE}": auxes[CONST_ADVANTAGES].item(),
            f"losses_info/{CONST_VALUE}": auxes[CONST_VALUES].item(),
            f"losses_info/pi_num_clipped": auxes[CONST_AUX][CONST_POLICY][
                CONST_NUM_CLIPPED
            ].item(),
            f"losses_info/pi_log_prob": auxes[CONST_AUX][CONST_POLICY][
                CONST_LOG_PROB
            ].item(),
            f"losses_info/vf_num_clipped": auxes[CONST_AUX][CONST_VF][
                CONST_NUM_CLIPPED
            ].item(),
            f"losses_info/is_ratio": auxes[CONST_AUX][CONST_POLICY][CONST_IS_RATIO]
            .mean()
            .item(),
            f"{CONST_GRAD_NORM}/pi": auxes[CONST_AUX][CONST_GRAD_NORM][
                CONST_POLICY
            ].item(),
            f"{CONST_PARAM_NORM}/pi": l2_norm(
                self.model_dict[CONST_MODEL][CONST_POLICY]
            ).item(),
            f"{CONST_GRAD_NORM}/vf": auxes[CONST_AUX][CONST_GRAD_NORM][CONST_VF].item(),
            f"{CONST_PARAM_NORM}/vf": l2_norm(
                self.model_dict[CONST_MODEL][CONST_VF]
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
            aux[CONST_LOG][
                f"{CONST_ACTION}/{CONST_ACTION}_{act_i}_{CONST_MEAN}"
            ] = auxes[CONST_ACTION][act_i][CONST_MEAN]
            aux[CONST_LOG][
                f"{CONST_POLICY}/{CONST_ACTION}_{act_i}_{CONST_MEAN}"
            ] = auxes[CONST_POLICY][act_i][CONST_MEAN]
            aux[CONST_LOG][
                f"{CONST_POLICY}/{CONST_ACTION}_{act_i}_{CONST_STD}"
            ] = auxes[CONST_POLICY][act_i][CONST_STD]

        self.gather_rms(aux)
        return aux
