from flax import linen as nn
from types import SimpleNamespace

import chex
import optax

from jaxl.constants import *
from jaxl.models.common import MLP, Model, Policy
from jaxl.models.policies import GaussianPolicy


"""
Getters for the models and the learners.
XXX: Feel free to add new components as needed.
"""


def get_optimizer(opt_config: SimpleNamespace) -> optax.GradientTransformation:
    if opt_config.optimizer == CONST_ADAM:
        return optax.adam(learning_rate=opt_config.lr)
    elif opt_config.optimizer == CONST_SGD:
        return optax.sgd(learning_rate=opt_config.lr)
    else:
        raise NotImplementedError


def get_model(
    input_dim: chex.Array, output_dim: chex.Array, model_config: SimpleNamespace
) -> Model:
    assert (
        model_config.architecture in VALID_ARCHITECTURE
    ), f"{model_config.architecture} is not supported (one of {VALID_ARCHITECTURE})"
    if model_config.architecture == CONST_MLP:
        return MLP(model_config.layers + list(output_dim))
    else:
        raise NotImplementedError


def get_policy(policy: Model, config: SimpleNamespace) -> Policy:
    assert (
        config.policy_distribution in VALID_POLICY_DISTRIBUTION
    ), f"{config.policy_distribution} is not supported (one of {VALID_POLICY_DISTRIBUTION})"
    if config.policy_distribution == CONST_GAUSSIAN:
        return GaussianPolicy(policy, config.min_std)
    else:
        raise NotImplementedError


def policy_output_dim(output_dim: chex.Array, config: SimpleNamespace) -> chex.Array:
    assert (
        config.policy_distribution in VALID_POLICY_DISTRIBUTION
    ), f"{config.policy_distribution} is not supported (one of {VALID_POLICY_DISTRIBUTION})"
    if config.policy_distribution == CONST_GAUSSIAN:
        return [dim * 2 for dim in output_dim]
    return output_dim
