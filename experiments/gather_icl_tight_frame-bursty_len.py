import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm
from types import SimpleNamespace
from typing import Dict, Any

import _pickle as pickle
import argparse
import copy
import os
import timeit

from jaxl.constants import *
from jaxl.datasets import get_dataset
from jaxl.models import load_config, iterate_models
from jaxl.utils import parse_dict, get_device

from utils import *


def get_eval_datasets(
    config_dict: Dict[str, Any],
    num_test_tasks: int,
    test_data_seed: int,
    context_len: int,
):
    configs = {}

    # Cosine Similarity: Abstract hierarchy
    for bursty_len in range(1, context_len + 1):
        for cos_threshold in [0.0, 0.2, 0.3]:
            n_abstract_hierarchy_config_dict = copy.deepcopy(config_dict)
            n_abstract_hierarchy_config_dict["learner_config"]["dataset_config"][
                "dataset_kwargs"
            ]["split"] = "train"
            n_abstract_hierarchy_config_dict["learner_config"]["dataset_config"][
                "dataset_kwargs"
            ]["task_name"] = "abstract_class"
            n_abstract_hierarchy_config_dict["learner_config"]["dataset_config"][
                "dataset_kwargs"
            ]["num_sequences"] = num_test_tasks
            n_abstract_hierarchy_config_dict["learner_config"]["dataset_config"][
                "dataset_kwargs"
            ]["abstraction"] = "{}-cos".format(cos_threshold)
            n_abstract_hierarchy_config_dict["learner_config"]["dataset_config"][
                "dataset_kwargs"
            ]["bursty_len"] = bursty_len
            n_abstract_hierarchy_config = parse_dict(n_abstract_hierarchy_config_dict)
            configs["bursty_len_{}-{}_cos".format(bursty_len, cos_threshold)] = (
                n_abstract_hierarchy_config
            )

    return {
        eval_name: get_data_loader(config, test_data_seed)
        for eval_name, config in configs.items()
    }, configs


def main(args: SimpleNamespace):
    device = args.device
    get_device(device)

    runs_dir = args.runs_dir
    num_train_tasks = args.num_train_tasks
    num_test_tasks = args.num_test_tasks
    test_data_seed = args.test_data_seed

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

        fixed_length = False
        if hasattr(config.learner_config.dataset_config, "dataset_wrapper"):
            fixed_length = (
                config.learner_config.dataset_config.dataset_wrapper.type
                in ["FixedLengthContextDataset"]
            )

        datasets, dataset_configs = get_eval_datasets(
            config_dict,
            num_test_tasks,
            test_data_seed,
            context_len,
        )

        accuracies = {eval_name: [] for eval_name in datasets}
        auxes = {eval_name: [] for eval_name in datasets}
        checkpoint_steps = []
        for params, model, checkpoint_step in iterate_models(
            train_dataset.input_dim, train_dataset.output_dim, learner_path
        ):
            checkpoint_steps.append(checkpoint_step)
            for eval_name in datasets:
                tic = timeit.default_timer()
                print(curr_run_path, checkpoint_step, eval_name)
                dataset, data_loader = datasets[eval_name]
                acc, aux = evaluate(
                    model=model,
                    params=params,
                    dataset=dataset,
                    data_loader=data_loader,
                    num_tasks=(
                        num_train_tasks
                        if eval_name == "pretraining"
                        else num_test_tasks
                    ),
                    max_label=(
                        2
                        if eval_name.endswith("2_way")
                        or eval_name.endswith("cos")
                        or eval_name.endswith("l2")
                        else None
                    ),
                    context_len=context_len,
                    fixed_length=fixed_length,
                )
                accuracies[eval_name].append(acc)
                auxes[eval_name].append(aux)
                toc = timeit.default_timer()
                print("Takes {}s".format(toc - tic))

        all_results[exp_name][curr_run_path] = {
            "checkpoint_steps": checkpoint_steps,
            "accuracies": accuracies,
            "auxes": auxes,
        }
    pickle.dump(
        all_results,
        open(
            os.path.join(save_path, "agg_data", "accuracies-bursty_len.pkl"),
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
        "--num_test_tasks", type=int, default=30, help="The number of evaluation tasks"
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
        default=100,
        help="The number of training tasks to evaluate on",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="The number of workers for fetching the batches of data",
    )
    args = parser.parse_args()

    main(args)
