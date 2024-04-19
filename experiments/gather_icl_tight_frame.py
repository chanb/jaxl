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
):
    # Same Pretraining
    same_pretraining_config_dict = copy.deepcopy(config_dict)
    same_pretraining_config_dict["learner_config"]["dataset_config"]["dataset_kwargs"][
        "num_sequences"
    ] = num_test_tasks
    same_pretraining_config = parse_dict(same_pretraining_config_dict)

    # In-weight
    in_weight_config_dict = copy.deepcopy(config_dict)
    in_weight_config_dict["learner_config"]["dataset_config"]["dataset_kwargs"][
        "p_bursty"
    ] = 0.0
    in_weight_config_dict["learner_config"]["dataset_config"]["dataset_kwargs"][
        "num_sequences"
    ] = num_test_tasks
    in_weight_config_dict["learner_config"]["dataset_config"]["dataset_kwargs"][
        "unique_classes"
    ] = True
    in_weight_config = parse_dict(in_weight_config_dict)

    # Pretrain N-shot 2-way
    pretrain_n_shot_2_way_config_dict = copy.deepcopy(config_dict)
    pretrain_n_shot_2_way_config_dict["learner_config"]["dataset_config"][
        "dataset_kwargs"
    ]["task_name"] = "n_shot_k_way"
    pretrain_n_shot_2_way_config_dict["learner_config"]["dataset_config"][
        "dataset_kwargs"
    ]["p_bursty"] = 1.0
    pretrain_n_shot_2_way_config_dict["learner_config"]["dataset_config"][
        "dataset_kwargs"
    ]["k_way"] = 2
    pretrain_n_shot_2_way_config_dict["learner_config"]["dataset_config"][
        "dataset_kwargs"
    ]["num_sequences"] = num_test_tasks
    pretrain_n_shot_2_way_config = parse_dict(pretrain_n_shot_2_way_config_dict)

    # Complete OOD
    ood_config_dict = copy.deepcopy(config_dict)
    ood_config_dict["learner_config"]["dataset_config"]["dataset_kwargs"][
        "split"
    ] = "test"
    ood_config_dict["learner_config"]["dataset_config"]["dataset_kwargs"][
        "num_sequences"
    ] = num_test_tasks
    ood_config = parse_dict(ood_config_dict)

    # OOD N-shot 2-way
    test_n_shot_2_way_config_dict = copy.deepcopy(config_dict)
    test_n_shot_2_way_config_dict["learner_config"]["dataset_config"]["dataset_kwargs"][
        "split"
    ] = "test"
    test_n_shot_2_way_config_dict["learner_config"]["dataset_config"]["dataset_kwargs"][
        "task_name"
    ] = "n_shot_k_way"
    test_n_shot_2_way_config_dict["learner_config"]["dataset_config"]["dataset_kwargs"][
        "p_bursty"
    ] = 1.0
    test_n_shot_2_way_config_dict["learner_config"]["dataset_config"]["dataset_kwargs"][
        "k_way"
    ] = 2
    test_n_shot_2_way_config_dict["learner_config"]["dataset_config"]["dataset_kwargs"][
        "num_sequences"
    ] = num_test_tasks
    test_n_shot_2_way_config = parse_dict(test_n_shot_2_way_config_dict)

    configs = {
        "same_pretraining": same_pretraining_config,
        "in_weight": in_weight_config,
        "pretrain_n_shot_2_way": pretrain_n_shot_2_way_config,
        "ood": ood_config,
        "test_n_shot_2_way": test_n_shot_2_way_config,
    }

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
    num_workers = args.num_workers

    ablation_name = os.path.basename(runs_dir)

    all_results = {}
    save_path = os.path.join(args.save_path, ablation_name)
    os.makedirs(os.path.join(save_path, "agg_data"), exist_ok=True)
    for curr_run_path in tqdm(os.listdir(runs_dir)):
        learner_path = os.path.join(runs_dir, curr_run_path)
        exp_name = "-".join(curr_run_path.split("-")[:-8])
        # if not exp_name.endswith("-tf"):
        #     continue
        all_results.setdefault(exp_name, {})

        config_dict, config = load_config(learner_path)

        train_dataset = get_dataset(
            config.learner_config.dataset_config,
            config.learner_config.seeds.data_seed,
        )

        context_len = config.model_config.num_contexts
        num_samples_per_task = train_dataset._dataset.sequence_length - context_len
        sequence_length = train_dataset._dataset.sequence_length

        fixed_length = True
        if hasattr(config.learner_config.dataset_config, "dataset_wrapper"):
            fixed_length = (
                config.learner_config.dataset_config.dataset_wrapper.type
                in ["FixedLengthContextDataset"]
            )

        print(num_samples_per_task, num_train_tasks, sequence_length, context_len)

        datasets, dataset_configs = get_eval_datasets(
            config_dict,
            num_test_tasks,
            test_data_seed,
        )
        datasets["pretraining"] = (
            train_dataset,
            train_dataset.get_dataloader(config.learner_config),
        )
        dataset_configs["pretraining"] = config.learner_config.dataset_config

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
                    max_label=2 if eval_name.endswith("2_way") else None,
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
