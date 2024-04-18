from types import SimpleNamespace
from typing import Any, Dict, Tuple, Union

import jax
import optax

from jaxl.constants import *
from jaxl.models.common import (
    Model,
    EncoderPredictorModel,
)
from jaxl.optimizers.schedules import *


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
    elif scheduler_config.scheduler == CONST_LINEAR_WARMUP_SQRT_DECAY:
        return linear_warmup_sqrt_decay(
            kwargs.max_lr,
            kwargs.warmup_steps,
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
                    optax.inject_hyperparams(optax.adamw)(
                        get_scheduler(opt_config.lr),
                        weight_decay=opt_config.weight_decay,
                    )
                )
            else:
                opt_transforms.append(
                    optax.inject_hyperparams(optax.adam)(
                        get_scheduler(opt_config.lr), b1=getattr(opt_config, "b1", 0.9)
                    )
                )
        elif opt_config.optimizer == CONST_SGD:
            opt_transforms.append(
                optax.inject_hyperparams(optax.sgd)(get_scheduler(opt_config.lr))
            )
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
