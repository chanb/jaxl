import warnings

warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader
from tqdm import tqdm
from types import SimpleNamespace
from typing import Dict, Any

import _pickle as pickle
import argparse
import copy
import os

from jaxl.constants import *
from jaxl.datasets import get_dataset, FixedLengthContextDataset
from jaxl.models import load_config, iterate_models
from jaxl.utils import parse_dict, get_device

from utils import *


def get_eval_datasets(
    config_dict: Dict[str, Any],
    num_test_tasks: int,
    test_data_seed: int,
    num_samples_per_task: int,
    context_len: int,
    num_workers: int,
):
    # Same Pretraining
    same_pretraining_config_dict = copy.deepcopy(
        config_dict["learner_config"]["dataset_config"]
    )
    same_pretraining_config_dict["dataset_kwargs"]["task_config"][
        "num_sequences"
    ] = num_test_tasks
    same_pretraining_config = parse_dict(same_pretraining_config_dict)

    # In-weight
    in_weight_config_dict = copy.deepcopy(
        config_dict["learner_config"]["dataset_config"]
    )
    in_weight_config_dict["dataset_kwargs"]["task_config"]["p_bursty"] = 0.0
    in_weight_config_dict["dataset_kwargs"]["task_config"][
        "num_sequences"
    ] = num_test_tasks
    in_weight_config_dict["dataset_kwargs"]["task_config"]["unique_classes"] = True
    in_weight_config = parse_dict(in_weight_config_dict)

    # Complete OOD
    ood_config_dict = copy.deepcopy(config_dict["learner_config"]["dataset_config"])
    ood_config_dict["dataset_kwargs"]["train"] = False
    ood_config_dict["dataset_kwargs"]["task_config"]["num_sequences"] = num_test_tasks
    ood_config = parse_dict(ood_config_dict)

    # Variable-length context example
    variable_length_context_config_dict = copy.deepcopy(
        config_dict["learner_config"]["dataset_config"]
    )
    variable_length_context_config_dict["dataset_kwargs"]["train"] = False
    variable_length_context_config_dict["dataset_kwargs"]["task_config"][
        "num_sequences"
    ] = num_test_tasks

    variable_length_context_config_dict["dataset_wrapper"]["type"] = "ContextDataset"
    variable_length_context_config_dict["dataset_wrapper"]["kwargs"][
        "include_query_class"
    ] = True
    variable_length_context_config = parse_dict(variable_length_context_config_dict)

    # Hierarchy
    hierarchy_config_dict = copy.deepcopy(
        config_dict["learner_config"]["dataset_config"]
    )
    hierarchy_config_dict["dataset_kwargs"]["train"] = False
    hierarchy_config_dict["dataset_kwargs"]["remap"] = True
    hierarchy_config_dict["dataset_kwargs"]["task_config"][
        "num_sequences"
    ] = num_test_tasks
    hierarchy_config = parse_dict(hierarchy_config_dict)

    # Enforce no same base class image in context
    unique_classes_hierarchy_config_dict = copy.deepcopy(
        config_dict["learner_config"]["dataset_config"]
    )
    unique_classes_hierarchy_config_dict["dataset_kwargs"]["train"] = False
    unique_classes_hierarchy_config_dict["dataset_kwargs"]["remap"] = True
    unique_classes_hierarchy_config_dict["dataset_kwargs"]["task_config"][
        "num_sequences"
    ] = num_test_tasks
    unique_classes_hierarchy_config_dict["dataset_kwargs"]["task_config"][
        "p_bursty"
    ] = 0.0
    unique_classes_hierarchy_config_dict["dataset_kwargs"]["task_config"][
        "unique_classes"
    ] = True
    unique_classes_hierarchy_config_dict["dataset_wrapper"]["kwargs"][
        "include_query_class"
    ] = False
    unique_classes_hierarchy_config = parse_dict(unique_classes_hierarchy_config_dict)

    # Hierarchy variable length
    variable_length_hierarchy_config_dict = copy.deepcopy(
        config_dict["learner_config"]["dataset_config"]
    )
    variable_length_hierarchy_config_dict["dataset_kwargs"]["train"] = False
    variable_length_hierarchy_config_dict["dataset_kwargs"]["remap"] = True
    variable_length_hierarchy_config_dict["dataset_kwargs"]["task_config"][
        "num_sequences"
    ] = num_test_tasks

    variable_length_hierarchy_config_dict["dataset_wrapper"]["type"] = "ContextDataset"
    variable_length_hierarchy_config_dict["dataset_wrapper"]["kwargs"][
        "include_query_class"
    ] = False
    variable_length_hierarchy_config = parse_dict(variable_length_hierarchy_config_dict)

    configs = {
        "same_pretraining": same_pretraining_config,
        "in_weight": in_weight_config,
        "ood": ood_config,
        "variable_length_context": variable_length_context_config,
        "hierarchy": hierarchy_config,
        "unique_classes_hierarchy": unique_classes_hierarchy_config,
        "variable_length_hierarchy": variable_length_hierarchy_config,
    }

    return {
        eval_name: get_data_loader(
            config,
            test_data_seed,
            (
                num_samples_per_task
                if config.dataset_wrapper.type in ["FixedLengthContextDataset"]
                else context_len
            ),
            num_workers,
        )
        for eval_name, config in configs.items()
    }, configs


def main(args: SimpleNamespace):
    device = args.device
    get_device(device)

    runs_dir = args.runs_dir
    test_data_seed = args.test_data_seed
    num_workers = args.num_workers
    num_visualize = args.num_visualize
    num_train_tasks = args.num_train_tasks
    num_test_tasks = args.num_test_tasks

    ablation_name = os.path.basename(runs_dir)

    all_results = {}
    save_path = os.path.join(args.save_path, ablation_name)
    os.makedirs(os.path.join(save_path, "agg_data"), exist_ok=True)
    for curr_run_path in tqdm(os.listdir(runs_dir)):
        learner_path = os.path.join(runs_dir, curr_run_path)
        exp_name = "-".join(curr_run_path.split("-")[:-8])
        all_results.setdefault(exp_name, {})

        config_dict, config = load_config(learner_path)

        train_dataset = get_dataset(
            config.learner_config.dataset_config,
            config.learner_config.seeds.data_seed,
        )

        context_len = config.model_config.num_contexts
        num_samples_per_task = train_dataset._dataset.sequence_length - context_len

        datasets, dataset_configs = get_eval_datasets(
            config_dict,
            num_test_tasks,
            test_data_seed,
            num_samples_per_task,
            context_len,
            num_workers,
        )
        datasets["pretraining"] = (
            train_dataset,
            DataLoader(
                train_dataset,
                batch_size=(
                    num_samples_per_task
                    if isinstance(train_dataset, FixedLengthContextDataset)
                    else context_len
                ),
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
            ),
        )
        dataset_configs["pretraining"] = config.learner_config.dataset_config

        if num_visualize > 0:
            print("Plot examples")
            os.makedirs(os.path.join(save_path, "plots", exp_name), exist_ok=True)
            for eval_name in datasets:
                plot_examples(
                    datasets[eval_name][0],
                    num_visualize,
                    save_path,
                    exp_name,
                    eval_name,
                )

        accuracies = {eval_name: [] for eval_name in datasets}
        auxes = {eval_name: [] for eval_name in datasets}
        checkpoint_steps = []
        for params, model, checkpoint_step in iterate_models(
            train_dataset.input_dim, train_dataset.output_dim, learner_path
        ):
            checkpoint_steps.append(checkpoint_step)
            for eval_name in datasets:
                dataset, data_loader = datasets[eval_name]
                fixed_length = isinstance(dataset, FixedLengthContextDataset)
                factor = 1
                if not fixed_length:
                    factor = context_len
                num_tasks = (
                    num_train_tasks if eval_name == "pretraining" else num_test_tasks
                ) * factor
                acc, aux = evaluate(
                    model=model,
                    params=params,
                    dataset=dataset,
                    data_loader=data_loader,
                    num_tasks=num_tasks,
                    max_label=2 if eval_name.endswith("2_way") else None,
                    context_len=context_len,
                    fixed_length=fixed_length,
                )
                accuracies[eval_name].append(acc)
                auxes[eval_name].append(aux)
        all_results[exp_name][curr_run_path] = {
            "checkpoint_steps": checkpoint_steps,
            "accuracies": accuracies,
            "auxes": auxes,
        }
    pickle.dump(
        all_results,
        open(
            os.path.join(save_path, "agg_data", "accuracies.pkl"),
            "wb",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, required=True, help="The device to run the model on"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="The location to save the results",
    )
    parser.add_argument(
        "--runs_dir",
        type=str,
        required=True,
        help="The experiment runs to load from",
    )
    parser.add_argument(
        "--num_test_tasks", type=int, default=50, help="The number of evaluation tasks"
    )
    parser.add_argument(
        "--test_data_seed",
        type=int,
        default=1000,
        help="The seed for generating the test data",
    )
    parser.add_argument(
        "--num_train_tasks",
        type=int,
        default=50,
        help="The number of training tasks to evaluate on",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="The number of workers for fetching the batches of data",
    )
    parser.add_argument(
        "--num_visualize",
        type=int,
        default=0,
        help="Visualize the examples per dataset",
    )
    args = parser.parse_args()

    main(args)
