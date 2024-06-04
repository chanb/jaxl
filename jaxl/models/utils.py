from types import SimpleNamespace
from typing import Any, Dict, Tuple, Union, Sequence, Iterable

import chex
import dill
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
    AsyncCustomTokenizerICSupervisedTransformer,
    NoTokenizerICSupervisedTransformer,
)
from jaxl.models.policies import *
from jaxl.models.q_functions import *
from jaxl.optimizers import *
from jaxl.utils import parse_dict


"""
Getters for the models and the learners.
XXX: Feel free to add new components as needed.
"""


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
            getattr(model_config, "use_batch_norm", False),
            getattr(model_config, "use_bias", True),
            getattr(model_config, "flatten", False),
        )
    elif model_config.architecture == CONST_CNN:
        return CNN(
            model_config.features,
            model_config.kernel_sizes,
            model_config.layers + list(np.prod(output_dim, keepdims=True)),
            getattr(model_config, "activation", CONST_RELU),
            getattr(model_config, "output_activation", CONST_IDENTITY),
            getattr(model_config, "use_batch_norm", False),
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
            if model_config.type == "async":
                constructor = AsyncCustomTokenizerICSupervisedTransformer
            elif model_config.type == "default":
                constructor = CustomTokenizerICSupervisedTransformer
            elif model_config.type == "no_tokenizer":
                constructor = NoTokenizerICSupervisedTransformer
            return constructor(
                output_dim,
                model_config.num_contexts,
                model_config.num_blocks,
                model_config.num_heads,
                model_config.embed_dim,
                getattr(model_config, "widening_factor", 1),
                model_config.positional_encoding,
                model_config.input_tokenizer,
                model_config.output_tokenizer,
                getattr(model_config, "query_pred_only", False),
                getattr(model_config, "input_output_same_encoding", True),
            )
        return InContextSupervisedTransformer(
            output_dim,
            model_config.num_contexts,
            model_config.num_blocks,
            model_config.num_heads,
            model_config.embed_dim,
            getattr(model_config, "widening_factor", 1),
            model_config.positional_encoding,
            getattr(model_config, "query_pred_only", False),
            getattr(model_config, "input_output_same_encoding", True),
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
        Any,
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
        optax.Params,
        Any],
        Tuple[Any, Any]
    ]

    """

    if isinstance(model, EncoderPredictorModel):

        def update_encoder_predictor(optimizer, grads, opt_state, params, batch_stats):
            updates, encoder_opt_state = optimizer[CONST_ENCODER].update(
                grads[CONST_ENCODER],
                opt_state[CONST_ENCODER],
                params[CONST_ENCODER],
            )
            encoder_params = optax.apply_updates(params[CONST_ENCODER], updates)
            encoder_params = model.encoder.update_batch_stats(
                encoder_params,
                batch_stats[CONST_ENCODER],
            )

            updates, predictor_opt_state = optimizer[CONST_PREDICTOR].update(
                grads[CONST_PREDICTOR],
                opt_state[CONST_PREDICTOR],
                params[CONST_PREDICTOR],
            )
            predictor_params = optax.apply_updates(params[CONST_PREDICTOR], updates)
            predictor_params = model.encoder.update_batch_stats(
                predictor_params,
                batch_stats[CONST_PREDICTOR],
            )

            return {
                CONST_ENCODER: encoder_params,
                CONST_PREDICTOR: predictor_params,
            }, {
                CONST_ENCODER: encoder_opt_state,
                CONST_PREDICTOR: predictor_opt_state,
            }

        return update_encoder_predictor
    else:

        def update_default(optimizer, grads, opt_state, params, batch_stats):
            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                params,
            )
            params = optax.apply_updates(params, updates)
            params = model.update_batch_stats(
                params,
                batch_stats,
            )
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
    :return: the model and the parameters
    :rtype: Tuple[Dict, Model]
    """
    config_path = os.path.join(learner_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        config = parse_dict(config_dict)

    model = get_model(input_dim, output_dim, config.model_config)

    all_steps = sorted(os.listdir(os.path.join(learner_path, "models")))
    to_load = min(len(all_steps) - 1, checkpoint_i)
    print("Loading checkpoint: {}".format(all_steps[to_load]))
    params = dill.load(
        open(os.path.join(learner_path, "models", all_steps[to_load]), "rb")
    )

    return params, model


def load_params(
    learner_path: str,
) -> Tuple[Model, Callable]:
    learner_path, checkpoint_i = learner_path.split(":")

    all_steps = sorted(os.listdir(os.path.join(learner_path, "models")))
    if checkpoint_i == "latest":
        step = all_steps[-1]
    else:
        step = np.argmin(
            np.abs(
                np.array([int(step.split(".")[0]) for step in all_steps]) - checkpoint_i
            )
        )
    return dill.load(open(os.path.join(learner_path, "models", step), "rb"))


def iterate_models(
    input_dim: Sequence[int],
    output_dim: Sequence[int],
    learner_path: str,
) -> Iterable[Tuple[Dict, Model, int]]:
    """
    An iterator that yields the model and the each checkpointed parameters

    :param input_dim: the input dimensionality
    :param output_dim: the output dimensionality
    :param learner_path: the path that stores the experiment configuation
    :type input_dim: Sequence[int]
    :type output_dim: Sequence[int]
    :type learner_path: str
    :return: an iterable of the model, the parameters, and the i'th checkpoint
    :rtype: Iterable[Tuple[Dict, Model, int]]
    """
    config_path = os.path.join(learner_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        config = parse_dict(config_dict)

    model = get_model(input_dim, output_dim, config.model_config)

    all_steps = sorted(os.listdir(os.path.join(learner_path, "models")))
    for step in all_steps:
        params = dill.load(open(os.path.join(learner_path, "models", step), "rb"))
        yield params, model, int(step.split(".dill")[0])
