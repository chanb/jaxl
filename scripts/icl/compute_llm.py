from jaxl.constants import *
from jaxl.learning_utils import get_learner
from jaxl.models.svm import *
from jaxl.utils import parse_dict

import _pickle as pickle
import argparse
import jax
import json
import numpy as np
import os

from functools import partial
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager
from types import SimpleNamespace


def make_model_specific(config: SimpleNamespace):
    query_pred_only = getattr(config.model_config, "query_pred_only", False)
    use_sigmoid = config.learner_config.losses[0] == CONST_SIGMOID_BCE
    if query_pred_only:
        if use_sigmoid:

            def process_prediction(preds):
                probs = jax.nn.sigmoid(preds.flatten())
                return np.eye(2)[(probs >= 0.5).astype(int)]

        else:

            def process_prediction(preds):
                return preds[:, -1]

    else:

        def process_prediction(preds):
            return preds[:, 0, -1]

    return process_prediction


def load_llm(learner_path: str):
    config_path = os.path.join(learner_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        config = parse_dict(config_dict)

    learner = get_learner(
        config.learner_config, config.model_config, config.optimizer_config
    )

    checkpoint_manager = CheckpointManager(
        os.path.join(learner_path, "models"),
        PyTreeCheckpointer(),
    )

    llm_params = checkpoint_manager.restore(checkpoint_manager.latest_step())
    llm_params[CONST_MODEL_DICT][CONST_MODEL][CONST_POSITIONAL_ENCODING] = dict()
    llm_model = learner._model
    return llm_params, llm_model, config


def examplar_len(
    llm_params, llm_model, context_inputs, context_outputs, queries, process_prediction
):
    results = {}
    for examplar_len in range(1, len(context_inputs) + 1):
        mask = (np.arange(len(context_inputs)) >= len(context_inputs) - examplar_len)[
            :, None
        ]
        curr_context_inputs = context_inputs * mask
        curr_context_outputs = context_outputs * mask

        llm_preds, _ = jax.vmap(llm_model.forward, in_axes=[None, 0, None])(
            llm_params[CONST_MODEL_DICT][CONST_MODEL],
            queries[:, None, None],
            {
                CONST_CONTEXT_INPUT: curr_context_inputs[None, :],
                CONST_CONTEXT_OUTPUT: curr_context_outputs[None, :],
            },
        )
        llm_preds = process_prediction(llm_preds)
        results[examplar_len] = llm_preds


def get_agent_result(context_data, queries, agent_path):
    llm_params, llm_model, config = load_llm(agent_path)
    process_prediction = make_model_specific(config)

    agent_result = {}
    for task_i in context_data:
        context_inputs = context_data[task_i][CONST_CONTEXT_INPUT]
        context_outputs = context_data[task_i][CONST_CONTEXT_OUTPUT]

        llm_preds = examplar_len(
            llm_params,
            llm_model,
            context_inputs,
            context_outputs,
            queries,
            process_prediction,
        )

        agent_result[task_i] = llm_preds
    return agent_result


def main(baseline_path, agent_path_dirs):
    assert os.path.isdir(baseline_path)
    context_data = pickle.load(
        open(os.path.join(baseline_path, "context_data.pkl"), "rb")
    )

    ground_truth_data = pickle.load(
        open(os.path.join(baseline_path, "ground_truth.pkl"), "rb")
    )

    _get_agent_result = partial(
        get_agent_result, context_data=context_data, queries=ground_truth_data["inputs"]
    )
    agent_results = {}
    for dir_i, agent_path_dir in enumerate(agent_path_dirs):
        print(
            "Processing: {} ({}/{} agent directories)".format(
                agent_path_dir, dir_i + 1, len(agent_path_dirs)
            )
        )

        for agent_i, rel_agent_path in enumerate(os.listdir(agent_path_dir)):
            agent_path = os.path.join(agent_path_dir, rel_agent_path)
            print(
                "Processing: {} ({}/{} agents)".format(
                    rel_agent_path, agent_i + 1, len(os.listdir(agent_path_dir))
                )
            )
            agent_results[agent_path] = _get_agent_result(agent_path=agent_path)

    with open(os.path.join(baseline_path, "agents.pkl"), "wb") as f:
        pickle.dump(agent_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_path",
        type=str,
        required=True,
        help="The path that stores the baseline results",
    )
    parser.add_argument(
        "--agent_path_dirs",
        nargs="+",
        help="The directories containing one or more agents",
    )

    args = parser.parse_args()
    main(
        args.baseline_path,
        args.agent_path_dirs,
    )
