from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jaxl.constants import *


def get_model_outputs(params, obss, acts, model_config, model, param_key):
    multi_output = False
    if param_key == CONST_QF:
        if model_config.qf.architecture == CONST_ENSEMBLE:
            out, state = jax.vmap(
                partial(
                    model.model.model.apply,
                    capture_intermediates=True,
                    mutable=["intermediates"],
                    eval=True,
                ),
                in_axes=[0, None],
            )(
                params[param_key],
                np.concatenate((obss, acts[:, None]), axis=-1),
            )
            multi_output = True
        else:
            out, state = model.model.model.apply(
                params[param_key],
                np.concatenate((obss, acts[:, None]), axis=-1),
                capture_intermediates=True,
                mutable=["intermediates"],
                eval=True,
            )
    else:
        out, state = model.model.apply(
            params[param_key],
            obss,
            capture_intermediates=True,
            mutable=["intermediates"],
            eval=True,
        )
    return out, state, multi_output


def compute_dormant(
    params, obss, acts, model_config, model, param_key, dormant_threshold=0.025
):
    out, state, multi_output = get_model_outputs(
        params, obss, acts, model_config, model, param_key
    )
    dormant_score = dict()
    is_dormant = dict()
    for kp, val in jax.tree_util.tree_flatten_with_path(state["intermediates"])[0]:
        if getattr(kp[0], "key", False) == "__call__":
            continue
        per_neuron_score = jnp.mean(jnp.abs(val), axis=1 if multi_output else 0)
        curr_key = "/".join(
            [
                curr_kp.key if hasattr(curr_kp, "key") else str(curr_kp.idx)
                for curr_kp in kp
            ][:-2]
        )
        # XXX: https://github.com/google/dopamine/issues/209
        dormant_score[curr_key] = per_neuron_score / jnp.mean(
            per_neuron_score, axis=-1, keepdims=True
        )
        is_dormant[curr_key] = dormant_score[curr_key] <= dormant_threshold
    return dormant_score, is_dormant, multi_output


def compute_dormant_score_stats(dormant_score, multi_output):
    if multi_output:
        flattened_scores = np.concatenate(
            [
                (dormant_score[key]).reshape((len(dormant_score[key]), -1))
                for key in list(dormant_score.keys())[:-1]
            ],
            axis=-1,
        )
        return {
            "std": np.std(flattened_scores, axis=1),
            "max": np.max(flattened_scores, axis=1),
            "min": np.min(flattened_scores, axis=1),
        }
    else:
        flattened_scores = np.concatenate(
            [(dormant_score[key]).flatten() for key in list(dormant_score.keys())[:-1]]
        )
        return {
            "std": np.std(flattened_scores),
            "max": np.max(flattened_scores),
            "min": np.min(flattened_scores),
        }


def compute_dormant_percentage(is_dormant, multi_output):
    return jax.tree_util.tree_reduce(
        lambda x, y: x + jnp.sum(y, axis=-1), is_dormant, 0
    ) / jax.tree_util.tree_reduce(
        lambda x, y: x + np.prod(y.shape[int(multi_output) :]), is_dormant, 0
    )
