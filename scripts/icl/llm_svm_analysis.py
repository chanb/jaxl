from jaxl.constants import *
from jaxl.learning_utils import get_learner
from jaxl.models.svm import *
from jaxl.utils import parse_dict

import _pickle as pickle
import argparse
import jax
import json
import os

from functools import partial
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager


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


def get_agent_repr(context_data, queries, agent_path, use_input_token_repr=False):
    llm_params, llm_model, _ = load_llm(agent_path)

    agent_result = {}
    for task_i in context_data:
        context_inputs = context_data[task_i][CONST_CONTEXT_INPUT]
        context_outputs = context_data[task_i][CONST_CONTEXT_OUTPUT]

        if queries is None:
            queries = context_inputs

        repr, _ = jax.vmap(llm_model.get_latent, in_axes=[None, 0, None])(
            llm_params[CONST_MODEL_DICT][CONST_MODEL],
            queries[:, None, None],
            {
                CONST_CONTEXT_INPUT: context_inputs[None],
                CONST_CONTEXT_OUTPUT: context_outputs[None],
            },
        )

        if use_input_token_repr:
            agent_result[task_i] = repr[0, 0, :-1:2]
        else:
            agent_result[task_i] = repr[:, 0, -1]
    return agent_result


def get_svm_sol(inputs, outputs):
    svm_sols = {}

    loss, sol = primal_svm(inputs, outputs)
    svm_sols["primal"] = {
        "loss": loss,
        "sol": sol,
    }

    loss, sol = dual_svm(inputs, outputs)
    svm_sols["dual"] = {
        "loss": loss,
        "sol": sol,
    }
    return svm_sols


def get_svms(repr_dict, context_data):
    svm_results = {}

    reprs = ["input", "context_reprs", "input_token_context_reprs"]
    for task_i in context_data:
        svm_results.setdefault(task_i, {})
        context_inputs = context_data[task_i][CONST_CONTEXT_INPUT]
        context_outputs = context_data[task_i][CONST_CONTEXT_OUTPUT]

        train_y = np.argmax(context_outputs, axis=-1)
        train_y[train_y == 0] = -1

        for repr in reprs:
            if repr == "input":
                inputs = context_inputs
            else:
                inputs = repr_dict[repr][task_i]

            svm_results[task_i][repr] = get_svm_sol(inputs, train_y)

    return svm_results


def main(baseline_path):
    assert os.path.isdir(baseline_path)
    context_data = pickle.load(
        open(os.path.join(baseline_path, "context_data.pkl"), "rb")
    )

    gt = pickle.load(open(os.path.join(baseline_path, "ground_truth.pkl"), "rb"))

    agents_data = pickle.load(open(os.path.join(baseline_path, "agents.pkl"), "rb"))

    agent_results = {}

    for agent_i, agent_path in enumerate(agents_data):
        print(
            "Processing: {} ({}/{} agents)".format(
                agent_path, agent_i + 1, len(agents_data)
            )
        )
        agent_results[agent_path] = {
            "context_reprs": get_agent_repr(context_data, None, agent_path),
            "input_token_context_reprs": get_agent_repr(
                context_data, None, agent_path, use_input_token_repr=True
            ),
            "query_reprs": get_agent_repr(context_data, gt["inputs"], agent_path),
        }

        agent_results[agent_path]["svms"] = get_svms(
            agent_results[agent_path], context_data
        )

    with open(os.path.join(baseline_path, "agent_reprs.pkl"), "wb") as f:
        pickle.dump(agent_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_path",
        type=str,
        required=True,
        help="The path that stores the baseline results",
    )

    args = parser.parse_args()
    main(
        args.baseline_path,
    )
