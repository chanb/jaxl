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
    # Pretrain
    same_pretraining_config = copy.deepcopy(config_dict)

    # IWL evaluation
    iwl_config_dict = copy.deepcopy(config_dict)

    dataset_kwargs = {
        "num_examples": 8,
        "p_bursty": 0.0,
        "bursty_len": 3,
        "zipf_exp": 0.0,
        "input_noise_std": 0.5,
        "target_allowed_in_example": False,
        "empty_examples": False,
        "num_base_classes": 100000,
        "num_abstract_classes": 32,
        "num_dims": 128,
        "base_per_abstract_map": None,
        "novel_abstract_class": False,
        "tf_type": "prepare_seqs_for_transformer",
        "num_workers": -1,
    }

    iwl_config_dict["learner_config"]["dataset_config"][
        "dataset_kwargs"
    ] = dataset_kwargs
    iwl_config = parse_dict(iwl_config_dict)

    # IWL evaluation with empty context
    iwl_empty_examples_config_dict = copy.deepcopy(config_dict)

    dataset_kwargs = {
        "num_examples": 8,
        "p_bursty": 0.0,
        "bursty_len": 3,
        "zipf_exp": 0.0,
        "input_noise_std": 0.5,
        "target_allowed_in_example": False,
        "empty_examples": True,
        "num_base_classes": 100000,
        "num_abstract_classes": 32,
        "num_dims": 128,
        "base_per_abstract_map": None,
        "novel_abstract_class": False,
        "tf_type": "prepare_seqs_for_transformer",
        "num_workers": -1,
    }

    iwl_empty_examples_config_dict["learner_config"]["dataset_config"][
        "dataset_kwargs"
    ] = dataset_kwargs
    iwl_empty_examples_config = parse_dict(iwl_empty_examples_config_dict)

    # ICL with novel input
    icl_novel_inputs_config_dict = copy.deepcopy(config_dict)

    dataset_kwargs = {
        "num_examples": 8,
        "p_bursty": 1.0,
        "bursty_len": 4,
        "zipf_exp": 0.0,
        "input_noise_std": 0.5,
        "target_allowed_in_example": False,
        "empty_examples": False,
        "num_base_classes": 100000,
        "num_abstract_classes": 32,
        "num_dims": 128,
        "base_per_abstract_map": None,
        "novel_abstract_class": False,
        "tf_type": "prepare_seqs_for_transformer",
        "num_workers": -1,
    }

    icl_novel_inputs_config_dict["learner_config"]["dataset_config"][
        "dataset_kwargs"
    ] = dataset_kwargs
    icl_novel_inputs_config_dict["learner_config"]["seeds"][
        "data_seed"
    ] = test_data_seed
    icl_novel_inputs_config = parse_dict(icl_novel_inputs_config_dict)

    # ICL with permuted label
    icl_permuted_label_config_dict = copy.deepcopy(config_dict)

    dataset_kwargs = {
        "num_examples": 8,
        "p_bursty": 1.0,
        "bursty_len": 4,
        "zipf_exp": 0.0,
        "input_noise_std": 0.5,
        "target_allowed_in_example": False,
        "empty_examples": False,
        "num_base_classes": 100000,
        "num_abstract_classes": 32,
        "num_dims": 128,
        "base_per_abstract_map": None,
        "novel_abstract_class": True,
        "tf_type": "prepare_seqs_for_transformer",
        "num_workers": -1,
    }

    icl_permuted_label_config_dict["learner_config"]["dataset_config"][
        "dataset_kwargs"
    ] = dataset_kwargs
    icl_permuted_label_config = parse_dict(icl_permuted_label_config_dict)

    configs = {
        "same_pretraining": same_pretraining_config,
        "iwl": iwl_config,
        "iwl_empty_examples": iwl_empty_examples_config,
        "icl_novel_inputs": icl_novel_inputs_config,
        "icl_permuted_label": icl_permuted_label_config,
    }

    return {
        eval_name: get_data_loader(config, config.learner_config.seeds.data_seed)
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
        # if not "random" in exp_name:
        #     continue
        # if not "learn_embedder" in exp_name:
        #     continue
        all_results.setdefault(exp_name, {})

        config_dict, config = load_config(learner_path)

        train_dataset = get_dataset(
            config.learner_config.dataset_config,
            config.learner_config.seeds.data_seed,
        )

        context_len = config.model_config.num_contexts

        fixed_length = True
        if hasattr(config.learner_config.dataset_config, "dataset_wrapper"):
            fixed_length = (
                config.learner_config.dataset_config.dataset_wrapper.type
                in ["FixedLengthContextDataset"]
            )

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
                    max_label=(
                        2
                        if eval_name.endswith("2_way")
                        or eval_name.endswith("cos")
                        or eval_name.endswith(None)
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
