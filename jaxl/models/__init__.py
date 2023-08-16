from types import SimpleNamespace
from typing import Any, Dict, Tuple, Union

import chex
import numpy as np
import optax

from jaxl.constants import *
from jaxl.models.common import MLP, Model, Policy, EnsembleModel, EncoderPredictorModel, GPT
from jaxl.models.policies import *


"""
Getters for the models and the learners.
XXX: Feel free to add new components as needed.
"""


def get_scheduler(
    scheduler_config: SimpleNamespace,
) -> optax.Schedule:
    """
    Gets a scheduler.

    :param scheduler_config: the scheduler configuration
    :type scheduler_config: SimpleNamespace
    :return: the scheduler
    :rtype: optax.Schedule
    """
    assert (
        scheduler_config.scheduler in VALID_SCEHDULER
    ), f"{scheduler_config.scheduler} is not supported (one of {VALID_SCEHDULER})"

    kwargs = scheduler_config.scheduler_kwargs
    if scheduler_config.scheduler == CONST_CONSTANT_SCHEDULE:
        return optax.constant_schedule(kwargs.value)
    elif scheduler_config.scheduler == CONST_LINEAR_SCHEDULE:
        return optax.linear_schedule(
            kwargs.init_value,
            kwargs.end_value,
            kwargs.transition_steps,
            kwargs.transition_begin,
        )
    elif scheduler_config.scheduler == CONST_EXPONENTIAL_DECAY:
        return optax.exponential_decay(
            kwargs.init_value,
            kwargs.transition_steps,
            kwargs.decay_rate,
            kwargs.transition_begin,
            kwargs.staircase,
            kwargs.end_value,
        )
    else:
        raise NotImplementedError


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
        # opt_transforms.append(optax.zero_nans())
        if opt_config.max_grad_norm:
            opt_transforms.append(optax.clip_by_global_norm(opt_config.max_grad_norm))
        if opt_config.optimizer == CONST_ADAM:
            if hasattr(opt_config, "weight_decay"):
                opt_transforms.append(
                    optax.adamw(
                        get_scheduler(opt_config.lr),
                        weight_decay=opt_config.weight_decay,
                    )
                )
            else:
                opt_transforms.append(optax.adam(get_scheduler(opt_config.lr)))
        elif opt_config.optimizer == CONST_SGD:
            opt_transforms.append(optax.sgd(get_scheduler(opt_config.lr)))
        else:
            raise NotImplementedError
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
            model_config.layers + list(np.prod(output_dim, keepdims=True)),
            getattr(model_config, "activation", CONST_RELU),
            getattr(model_config, "output_activation", CONST_IDENTITY),
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
    elif model_config.architecture == CONST_GPT:
        return GPT(
            model_config.num_blocks,
            model_config.num_heads,
            model_config.num_embeddings,
            model_config.embed_dim,
            getattr(model_config, "output_dim", output_dim),
            model_config.positional_encoding,
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
        return SoftmaxPolicy(
            model, getattr(config, CONST_TEMPERATURE, DEFAULT_TEMPERATURE)
        )
    elif config.policy_distribution == CONST_BANG_BANG:
        return BangBangPolicy(
            model, getattr(config, CONST_TEMPERATURE, DEFAULT_TEMPERATURE)
        )
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
        return [*output_dim[:-1], output_dim[-1] * 2]
    return output_dim


def get_update_function(
    model: Model,
) -> Callable[
    [
        optax.GradientTransformation,
        Union[Dict[str, Any], chex.Array],
        optax.OptState,
        optax.Params,
    ],
    Tuple[Any, Any],
]:
    """
    Gets the update function based on the model architecture

    :param model: the model
    :type model: Model
    :return: the update function
    :rtype: Callable[
        [optax.GradientTransformation,
        Union[Dict[str, Any], chex.Array],
        optax.OptState,
        optax.Params],
        Tuple[Any, Any]
    ]

    """

    if isinstance(model, EncoderPredictorModel):

        def update_encoder_predictor(optimizer, grads, opt_state, params):
            updates, encoder_opt_state = optimizer[CONST_ENCODER].update(
                grads[CONST_ENCODER],
                opt_state[CONST_ENCODER],
                params[CONST_ENCODER],
            )
            encoder_params = optax.apply_updates(params[CONST_ENCODER], updates)

            updates, predictor_opt_state = optimizer[CONST_PREDICTOR].update(
                grads[CONST_PREDICTOR],
                opt_state[CONST_PREDICTOR],
                params[CONST_PREDICTOR],
            )
            predictor_params = optax.apply_updates(params[CONST_PREDICTOR], updates)

            return {
                CONST_ENCODER: encoder_params,
                CONST_PREDICTOR: predictor_params,
            }, {
                CONST_ENCODER: encoder_opt_state,
                CONST_PREDICTOR: predictor_opt_state,
            }

        return update_encoder_predictor
    else:

        def update_default(optimizer, grads, opt_state, params):
            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                params,
            )
            params = optax.apply_updates(params, updates)
            return params, opt_state

        return update_default
