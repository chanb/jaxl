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
    test_data_seed: int,
    context_len: int,
):
    # ICL with novel input
    icl_novel_inputs_config_dict = copy.deepcopy(config_dict)
    icl_novel_inputs_config_dict["learner_config"]["seeds"][
        "data_seed"
    ] = test_data_seed
    icl_novel_inputs_config = parse_dict(icl_novel_inputs_config_dict)

    icl_iid_context_config_dict = copy.deepcopy(config_dict)
    dataset_kwargs = {"mode": "iid_context"}

    icl_iid_context_config_dict["learner_config"]["dataset_config"][
        "dataset_kwargs"
    ].update(dataset_kwargs)
    icl_iid_context = parse_dict(icl_iid_context_config_dict)

    configs = {
        "icl_novel_inputs": icl_novel_inputs_config,
        "icl_iid_context": icl_iid_context,
    }

    # Context length evaluations
    for prob_key in ["sample_high_prob_class_only", "sample_low_prob_class_only"]:
        for fixed_start_pos in range(context_len):
            start_pos_config_dict = copy.deepcopy(config_dict)

            dataset_kwargs = {
                prob_key: 1,
                "fixed_start_pos": fixed_start_pos,
                "mode": "default",
            }

            start_pos_config_dict["learner_config"]["dataset_config"][
                "dataset_kwargs"
            ].update(dataset_kwargs)

            start_pos_config = parse_dict(start_pos_config_dict)
            configs["{}-start_pos_{}".format(prob_key, fixed_start_pos)] = (
                start_pos_config
            )

    return {
        eval_name: get_data_loader(config, config.learner_config.seeds.data_seed)
        for eval_name, config in configs.items()
    }, configs


def main(args: SimpleNamespace):
    device = args.device
    get_device(device)

    runs_dir = args.runs_dir
    batch_size = args.batch_size
    num_eval_samples = args.num_eval_samples
    test_data_seed = args.test_data_seed

    ablation_name = os.path.basename(runs_dir)

    all_results = {}
    save_path = os.path.join(args.save_path, ablation_name)
    os.makedirs(os.path.join(save_path, "agg_data"), exist_ok=True)
    for curr_run_path in os.listdir(runs_dir):
        print("Evaluating {}".format(curr_run_path))
        learner_path = os.path.join(runs_dir, curr_run_path)
        exp_name = "-".join(curr_run_path.split("-")[:-8])
        all_results.setdefault(exp_name, {})

        config_dict, config = load_config(learner_path)
        config_dict["learner_config"]["batch_size"] = batch_size
        config = parse_dict(config_dict)

        train_dataset = get_dataset(
            config.learner_config.dataset_config,
            config.learner_config.seeds.data_seed,
        )

        context_len = config.model_config.num_contexts
        fixed_length = True

        datasets, dataset_configs = get_eval_datasets(
            config_dict,
            test_data_seed,
            context_len,
        )
        datasets["pretraining"] = (
            train_dataset,
            train_dataset.get_dataloader(config.learner_config),
        )
        dataset_configs["pretraining"] = config.learner_config.dataset_config

        prefetched_data = {}
        for eval_name in tqdm(datasets, postfix="Prefetching data"):
            dataset, data_loader = datasets[eval_name]
            data_iter = iter(data_loader)
            prefetched_data[eval_name] = dict(
                samples=[
                    next(data_iter)
                    for _ in range(num_eval_samples // batch_size)
                ],
                dataset_output_dim=dataset.output_dim[0]
            )

        accuracies = {eval_name: [] for eval_name in datasets}
        auxes = {eval_name: [] for eval_name in datasets}
        checkpoint_steps = []
        for params, model, checkpoint_step in tqdm(
            iterate_models(
                train_dataset.input_dim, train_dataset.output_dim, learner_path
            )
        ):
            checkpoint_steps.append(checkpoint_step)
            for eval_name in datasets:
                dataset, data_loader = datasets[eval_name]
                acc, aux = evaluate(
                    model=model,
                    params=params,
                    prefetched_data=prefetched_data[eval_name],
                    max_label=None,
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
        "--num_eval_samples",
        type=int,
        default=1000,
        help="The number of evaluation tasks",
    )
    parser.add_argument("--batch_size", type=int, default=100, help="The batch size")
    parser.add_argument(
        "--test_data_seed",
        type=int,
        default=1000,
        help="The seed for generating the test data",
    )
    args = parser.parse_args()

    main(args)
