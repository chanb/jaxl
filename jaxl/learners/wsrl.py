from types import SimpleNamespace
from typing import Any, Dict, Tuple

import chex
import dill
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import os
import timeit

from jaxl.constants import *
from jaxl.learners.reinforce import REINFORCE
from jaxl.learners.ppo import PPO
from jaxl.models import (
    get_model,
    get_wsrl_model,
    get_policy,
    get_update_function,
    load_params,
    policy_output_dim,
)
from jaxl.optimizers import (
    get_scheduler,
    get_optimizer,
)
from jaxl.utils import l2_norm


"""
Standard Warm-start RL.
"""


class WSRLPPO(PPO):
    """
    WSRL with Proximal Policy Optimization (PPO) algorithm.
    """

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
            CONST_POLICY: get_wsrl_model(input_dim, act_dim, self._model_config.policy),
            CONST_VF: get_model(input_dim, (1,), self._model_config.vf),
        }

        model_keys = jrandom.split(jrandom.PRNGKey(self._config.seeds.model_seed))
        dummy_x = self._generate_dummy_x(input_dim)
        pi_params = self._model[CONST_POLICY].init(model_keys[0], dummy_x)
        vf_params = self._model[CONST_VF].init(model_keys[1], dummy_x)

        if getattr(self._config, "load_pretrain", False):
            all_params = load_params(self._config.load_pretrain.checkpoint_path)
            if CONST_POLICY in self._config.load_pretrain.load_components:
                pi_params[CONST_MEAN] = all_params[CONST_MODEL_DICT][CONST_MODEL][
                    CONST_POLICY
                ]
            if CONST_VF in self._config.load_pretrain.load_components:
                vf_params = all_params[CONST_MODEL_DICT][CONST_MODEL][CONST_VF]
            if getattr(self._config, CONST_OBS_RMS, False):
                self._obs_rms_state = all_params[CONST_OBS_RMS]

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


class WSRLREINFORCE(REINFORCE):
    """
    WSRL with REINFORCE algorithm.
    """

    def _initialize_model_and_opt(self, input_dim: chex.Array, output_dim: chex.Array):
        """
        Construct the policy and the optimizer.

        :param input_dim: input dimension of the data point
        :param output_dim: output dimension of the data point
        :type input_dim: chex.Array
        :type output_dim: chex.Array

        """
        output_dim = policy_output_dim(output_dim, self._config)
        self._model = get_wsrl_model(input_dim, output_dim, self._model_config)

        model_key = jrandom.PRNGKey(self._config.seeds.model_seed)
        dummy_x = self._generate_dummy_x(input_dim)
        params = self._model.init(model_key, dummy_x)
        if getattr(self._config, "load_pretrain", False):
            all_params = load_params(self._config.load_pretrain.checkpoint_path)
            if CONST_POLICY in self._config.load_pretrain.load_components:
                params[CONST_MEAN] = all_params[CONST_MODEL_DICT][CONST_MODEL][
                    CONST_POLICY
                ]

        self._optimizer, opt_state = get_optimizer(
            self._optimizer_config, self._model, params
        )

        self._model_dict = {
            CONST_MODEL: {CONST_POLICY: params},
            CONST_OPT_STATE: {CONST_POLICY: opt_state},
        }
