from types import SimpleNamespace
from typing import Any, Dict, Tuple, Union

import chex
import optax

from jaxl.constants import *
from jaxl.models.common import MLP, Model, Policy, EnsembleModel, EncoderPredictorModel
from jaxl.models.policies import DeterministicPolicy, GaussianPolicy, SoftmaxPolicy


"""
Getters for the models and the learners.
XXX: Feel free to add new components as needed.
"""


def get_optimizer(
    opt_config: SimpleNamespace,
    model: Model,
    params: Union[optax.Params, Dict[str, Any]],
) -> Union[
    Tuple[Dict[str, Any], Dict[str, Any]],
    Tuple[optax.GradientTransformation, optax.OptState],
]:
    """
    Gets an optimizer and its optimizer state.

    :param opt_config: the optimizer configuration
    :param model: the model
    :param params: the model parameters
    :type opt_config: SimpleNamespace
    :type model: Model
    :type params: Union[optax.Params, Dict[str, Any]]
    :return: an optimizer and its optimizer state
    :rtype: Union[
        Tuple[Dict[str, Any], Dict[str, Any]],
        Tuple[optax.GradientTransformation, optax.OptState]
    ]

    """
    if isinstance(model, EncoderPredictorModel):
        encoder_opt, encoder_opt_state = get_optimizer(
            opt_config.encoder, model.encoder, params[CONST_ENCODER]
        )
        predictor_opt, predictor_opt_state = get_optimizer(
            opt_config.predictor, model.predictor, params[CONST_PREDICTOR]
        )
        return {
            CONST_ENCODER: encoder_opt,
            CONST_PREDICTOR: predictor_opt,
        }, {
            CONST_ENCODER: encoder_opt_state,
            CONST_PREDICTOR: predictor_opt_state,
        }

    assert (
        opt_config.optimizer in VALID_OPTIMIZER
    ), f"{opt_config.optimizer} is not supported (one of {VALID_OPTIMIZER})"

    opt_transforms = []
    if opt_config.optimizer == CONST_FROZEN:
        opt_transforms.append(optax.set_to_zero())
    else:
        opt_transforms.append(optax.zero_nans())
        if opt_config.max_grad_norm:
            opt_transforms.append(optax.clip_by_global_norm(opt_config.max_grad_norm))
        if opt_config.optimizer == CONST_ADAM:
            opt_transforms.append(optax.scale_by_adam())
        elif opt_config.optimizer == CONST_SGD:
            pass
        else:
            raise NotImplementedError

        opt_transforms.append(optax.scale(-opt_config.lr))
    opt = optax.chain(*opt_transforms)
    opt_state = opt.init(params)
    return opt, opt_state


def get_model(
    input_dim: chex.Array, output_dim: chex.Array, model_config: SimpleNamespace
) -> Model:
    """
    Gets a model.

    :param input_dim: the input dimension
    :param output_dim: the output dimension
    :param model_config: the model configuration
    :type input_dim: chex.Array
    :type output_dim: chex.Array
    :type model_config: SimpleNamespace
    :return: a model
    :rtype: Model

    """
    assert (
        model_config.architecture in VALID_ARCHITECTURE
    ), f"{model_config.architecture} is not supported (one of {VALID_ARCHITECTURE})"
    if model_config.architecture == CONST_MLP:
        return MLP(
            model_config.layers + list(output_dim),
            getattr(model_config, "activation", CONST_RELU),
        )
    elif model_config.architecture == CONST_ENCODER_PREDICTOR:
        encoder = get_model(input_dim, model_config.encoder_dim, model_config.encoder)
        predictor = get_model(
            model_config.encoder_dim, output_dim, model_config.predictor
        )
        return EncoderPredictorModel(
            encoder,
            predictor,
        )
    elif model_config.architecture == CONST_ENSEMBLE:
        model = get_model(
            getattr(model_config, "input_dim", input_dim),
            getattr(model_config, "output_dim", output_dim),
            model_config.model,
        )
        return EnsembleModel(
            model, model_config.num_models, getattr(model_config, "vmap_all", True)
        )
    else:
        raise NotImplementedError


def get_policy(model: Model, config: SimpleNamespace) -> Policy:
    """
    Gets a policy

    :param model: a model
    :param config: the policy configuration
    :type model: Model
    :type config: SimpleNamespace
    :return: a policy
    :rtype: Policy

    """
    assert (
        config.policy_distribution in VALID_POLICY_DISTRIBUTION
    ), f"{config.policy_distribution} is not supported (one of {VALID_POLICY_DISTRIBUTION})"

    if config.policy_distribution == CONST_GAUSSIAN:
        return GaussianPolicy(model, getattr(config, CONST_MIN_STD, DEFAULT_MIN_STD))
    elif config.policy_distribution == CONST_DETERMINISTIC:
        return DeterministicPolicy(model)
    elif config.policy_distribution == CONST_SOFTMAX:
        return SoftmaxPolicy(model)
    else:
        raise NotImplementedError


def policy_output_dim(output_dim: chex.Array, config: SimpleNamespace) -> chex.Array:
    """
    Gets the policy output dimension based on its distribution.

    :param output_dim: the original output dimension
    :param config: the policy configuration
    :type output_dim: chex.Array
    :type config: SimpleNamespace
    :return: the output dimension of the policy
    :rtype: chex.Array

    """
    assert (
        config.policy_distribution in VALID_POLICY_DISTRIBUTION
    ), f"{config.policy_distribution} is not supported (one of {VALID_POLICY_DISTRIBUTION})"
    if config.policy_distribution == CONST_GAUSSIAN:
        return [dim * 2 for dim in output_dim]
    return output_dim
