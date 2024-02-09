from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager
from types import SimpleNamespace
from typing import Any, Dict, Tuple, Union, Sequence

import chex
import json
import numpy as np
import optax
import os

from jaxl.constants import *
from jaxl.models.common import (
    CNN,
    MLP,
    Model,
    EnsembleModel,
    EncoderPredictorModel,
)
from jaxl.models.transformers import (
    InContextSupervisedTransformer,
    CustomTokenizerICSupervisedTransformer,
)
from jaxl.models.policies import *
from jaxl.models.q_functions import *
from jaxl.utils import parse_dict


"""
Getters for the models and the learners.
XXX: Feel free to add new components as needed.
"""


def get_param_mask_by_name(p: optax.Params, mask_names: list) -> Any:
    """
    Mask parameters based on the layer name.

    :param p: the parameters
    :param mask_names: the list of layer names to mask
    :type p: optax.Params
    :type mask_names: list
    :return: a mask indicating which layer to filter
    :rtype: Any
    """
    return jax.tree_util.tree_map_with_path(
        lambda key_path, _: key_path[0].key in mask_names, p
    )


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
    mask_names = getattr(opt_config, CONST_MASK_NAMES, [])
    if len(mask_names):
        mask = get_param_mask_by_name(params, mask_names)
        set_to_zero = optax.masked(optax.set_to_zero(), mask)
        opt_transforms.insert(0, set_to_zero)
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
    elif model_config.architecture == CONST_CNN:
        return CNN(
            model_config.features,
            model_config.kernel_sizes,
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
    elif model_config.architecture == CONST_ICL_GPT:
        if hasattr(model_config, CONST_INPUT_TOKENIZER) and hasattr(
            model_config, CONST_OUTPUT_TOKENIZER
        ):
            return CustomTokenizerICSupervisedTransformer(
                output_dim,
                model_config.num_contexts,
                model_config.num_blocks,
                model_config.num_heads,
                model_config.embed_dim,
                model_config.positional_encoding,
                model_config.input_tokenizer,
                model_config.output_tokenizer,
                getattr(model_config, "query_pred_only", False),
            )
        return InContextSupervisedTransformer(
            output_dim,
            model_config.num_contexts,
            model_config.num_blocks,
            model_config.num_heads,
            model_config.embed_dim,
            model_config.positional_encoding,
            getattr(model_config, "query_pred_only", False),
        )
    else:
        raise NotImplementedError


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


def load_config(learner_path) -> Tuple[Dict, SimpleNamespace]:
    """
    Loads the configuration file of an experiment

    :param learner_path: the path that stores the experiment configuation
    :type learner_path: str
    :return: the experiment configuration
    :rtype: Tuple[Dict, SimpleNamespace]

    """
    config_path = os.path.join(learner_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        config = parse_dict(config_dict)

    return config_dict, config


def load_model(
    input_dim: Sequence[int],
    output_dim: Sequence[int],
    learner_path: str,
    checkpoint_i: int,
) -> Tuple[Dict, Model]:
    """
    Loads the model and the parameters

    :param input_dim: the input dimensionality
    :param output_dim: the output dimensionality
    :param learner_path: the path that stores the experiment configuation
    :param checkpoint_i: the i'th checkpoint to load from
    :type input_dim: Sequence[int]
    :type output_dim: Sequence[int]
    :type learner_path: str
    :type checkpoint_i: int
    :return: the parameters, the model, and the experiment configuration
    :rtype: Tuple[Dict, Model]
    """
    config_path = os.path.join(learner_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        config = parse_dict(config_dict)

    model = get_model(input_dim, output_dim, config.model_config)

    checkpoint_manager = CheckpointManager(
        os.path.join(learner_path, "models"),
        PyTreeCheckpointer(),
    )

    all_steps = checkpoint_manager.all_steps()
    params = checkpoint_manager.restore(
        all_steps[min(len(all_steps) - 1, checkpoint_i)]
    )

    return params, model
