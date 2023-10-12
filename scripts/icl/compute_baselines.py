from jaxl.constants import *
from jaxl.datasets import get_dataset
from jaxl.models.svm import *
from jaxl.utils import parse_dict

import _pickle as pickle
import argparse
import numpy as np
import os

from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier


KNN = "knn"
SVM = "svm"
LR = "lr"


def make_test_dataset(
    num_tasks, seq_len, seed, input_range=[-1.0, 1.0], params_bound=[-0.5, 0.5]
):
    dataset_config = {
        "dataset_name": "multitask_nd_linear_classification",
        "dataset_kwargs": {
            "input_dim": 2,
            "input_range": input_range,
            "num_sequences": num_tasks,
            "sequence_length": seq_len,
            "noise": 0.0,
            "params_bound": params_bound,
            "num_active_params": None,
            "val_frac": 0.0005,
            "margin": 0.2,
        },
    }
    dataset_config["dataset_kwargs"] = parse_dict(dataset_config["dataset_kwargs"])
    ns_test_config = parse_dict(dataset_config)
    return get_dataset(ns_test_config, seed=seed)


def make_knn(inputs, outputs, num_neighbours):
    knn = make_pipeline(
        KNeighborsClassifier(
            n_neighbors=num_neighbours,
        )
    )
    knn.fit(inputs, np.argmax(outputs, axis=1))
    return knn


def make_svm(inputs, outputs, reg_coef):
    svm = make_pipeline(
        LinearSVC(C=reg_coef, max_iter=2000, dual="auto"),
    )
    svm.fit(inputs, np.argmax(outputs, axis=1))
    return svm


def make_lr(inputs, outputs, reg_coef, penalty="l2"):
    logistic_regression = make_pipeline(
        LogisticRegression(
            penalty=penalty,
            C=reg_coef,
            max_iter=2000,
        )
    )
    logistic_regression.fit(inputs, np.argmax(outputs, axis=1))
    return logistic_regression


def build_baseline_models(context_data, num_tasks, hyperparams, make_model):
    model_res = {}
    for task_i in range(num_tasks):
        models = {}
        context_inputs = context_data[task_i][CONST_CONTEXT_INPUT]
        context_outputs = context_data[task_i][CONST_CONTEXT_OUTPUT]

        for hyperparam in hyperparams:
            models[hyperparam] = make_model(context_inputs, context_outputs, hyperparam)

        model_res[task_i] = models
    return model_res


def get_ground_truth(dataset, num_tasks, delta=0.01, input_range=[-1.0, 1.0]):
    xs_grid = np.arange(input_range[0], input_range[1] + delta, delta)
    test_queries = np.stack(np.meshgrid(xs_grid, xs_grid)).reshape((2, -1)).T
    test_data = {"inputs": test_queries, "outputs": {}, "decision_boundary": {}}

    for task_i in range(num_tasks):
        test_data["outputs"][task_i] = np.eye(2)[
            (
                (test_queries @ dataset.params[task_i, 1:] + dataset.params[task_i, :1])
                >= 0
            )
            .flatten()
            .astype(int)
        ]

        gt = dataset.params[task_i]
        test_data["decision_boundary"][task_i] = -np.array(input_range) * gt[1] / gt[2]
    return test_data


def compute_baseline_results(context_data, baseline_models, input_range=[-1.0, 1.0]):
    baseline_results = {}
    for model_class in baseline_models:
        baseline_results.setdefault(model_class, {})
        for task_i, models_per_task in baseline_models[model_class].items():
            baseline_results[model_class].setdefault(task_i, {})
            for hyperparam, model in models_per_task.items():
                model_out = (
                    -(
                        np.array(input_range) * model[0].coef_[0, 0]
                        + model[0].intercept_[0]
                    )
                    / model[0].coef_[0, 1]
                )
                baseline_results[model_class][task_i][hyperparam] = {
                    "decision_boundary": model_out,
                    "preds": model.predict(context_data[task_i][CONST_CONTEXT_INPUT]),
                }

                if model_class == SVM:
                    decision_function = model.decision_function(
                        context_data[task_i][CONST_CONTEXT_INPUT]
                    )
                    support_vector_indices = np.where(
                        np.abs(decision_function) <= 1 + 1e-15
                    )[0]

                    baseline_results[model_class][task_i][hyperparam]["support_vector_indices"] = support_vector_indices
    return baseline_results


BASELINES = {
    SVM: (make_svm, [1e-2, 1e-1, 5e-1, 1.0, 2.0, 10.0, 100.0, 1000.0]),
    LR: (make_lr, [1e-2, 1e-1, 5e-1, 1.0, 2.0, 10.0, 100.0, 1000.0]),
}


def main(
    save_path_prefix: str,
    num_tasks: int,
    seq_len: int,
    seed: int,
    input_range: list = [-1.0, 1.0],
):
    time_tag = datetime.strftime(datetime.now(), "%m-%d-%y_%H_%M_%S")
    save_path = "{}-num_tasks_{}-seq_len_{}-seed_{}-{}".format(
        save_path_prefix,
        num_tasks,
        seq_len,
        seed,
        time_tag,
    )
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "config.pkl"), "wb") as f:
        pickle.dump(
            {
                "num_tasks": num_tasks,
                "seq_len": seq_len,
                "seed": seed,
                "input_range": input_range,
            },
            f,
        )

    test_dataset = make_test_dataset(
        num_tasks,
        seq_len,
        seed,
        input_range=input_range,
    )

    context_data = {}
    for task_i in range(num_tasks):
        context_data[task_i] = {
            CONST_CONTEXT_INPUT: test_dataset._inputs[task_i],
            CONST_CONTEXT_OUTPUT: test_dataset._targets[task_i],
        }

    with open(os.path.join(save_path, "context_data.pkl"), "wb") as f:
        pickle.dump(context_data, f)

    baseline_models = {}
    for model_class, (make_model_func, hyperparams) in BASELINES.items():
        baseline_models[model_class] = build_baseline_models(
            context_data, num_tasks, hyperparams, make_model_func
        )

    with open(os.path.join(save_path, "baseline_models.pkl"), "wb") as f:
        pickle.dump(baseline_models, f)

    baseline_results = compute_baseline_results(
        context_data, baseline_models, input_range=input_range
    )

    with open(os.path.join(save_path, "baseline_results.pkl"), "wb") as f:
        pickle.dump(baseline_results, f)

    ground_truth = get_ground_truth(test_dataset, num_tasks, input_range=input_range)

    with open(os.path.join(save_path, "ground_truth.pkl"), "wb") as f:
        pickle.dump(ground_truth, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path_prefix",
        type=str,
        required=True,
        help="The prefix of the path to save the results to",
    )

    parser.add_argument(
        "--num_tasks",
        type=int,
        default=5,
        help="The number of tasks to evaluate on",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=40,
        help="The number of examples per evaluation task",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=9999,
        help="The seed for generating the test set",
    )

    args = parser.parse_args()
    main(
        args.save_path_prefix,
        args.num_tasks,
        args.seq_len,
        args.seed,
    )
