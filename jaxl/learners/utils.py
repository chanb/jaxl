from typing import Any, Dict, Union

import jax
import optax

from jaxl.constants import CONST_PARAM_NORM
from jaxl.utils import per_leaf_l2_norm


def gather_per_leaf_l2_norm(
    aux: Dict,
    model_name: str,
    model_dict: Union[optax.Params, Dict[str, Any]],
):
    """
    Gathers the L2 norm of a given model

    :param aux: the auxiliary object that contains the information
    :param model_name: the model name
    :param model_dict: the model parameters
    :type aux: Dict
    :type model_name: str
    :type model_dict: Union[optax.Params, Dict[str, Any]]

    """
    param_norm = per_leaf_l2_norm(model_dict)
    for k, v in jax.tree_util.tree_flatten_with_path(param_norm)[0]:
        k = ".".join([layer.key for layer in k])
        aux[f"{CONST_PARAM_NORM}/{model_name}_{k}"] = v.item()
