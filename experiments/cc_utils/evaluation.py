import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm
from types import SimpleNamespace
from typing import Dict, Any

import _pickle as pickle
import argparse
import copy
import timeit

from jaxl.constants import *
from jaxl.datasets import get_dataset
from jaxl.models import load_config, iterate_models
from jaxl.utils import parse_dict

from cc_utils.utils import *
from cc_utils.constants import (
    CONFIG_DIR,
    LOG_DIR,
    RUN_REPORT_DIR,
    REPO_PATH,
    CC_ACCOUNT,
)


def get_eval_datasets(
    config_dict: Dict[str, Any],
    test_data_seed: int,
    context_len: int,
    skip_test: bool = False,
):
    configs = dict()
    for split in ["pretrain", "test"]:
        if skip_test and split == "test":
            continue

        if split == "test":

            def modify_seed(config_dict):
                config_dict["learner_config"]["seeds"]["data_seed"] = test_data_seed

        else:

            def modify_seed(config_dict):
                pass

        icl_iid_context_config_dict = copy.deepcopy(config_dict)
        modify_seed(icl_iid_context_config_dict)
        dataset_kwargs = {"mode": "iid_context"}

        icl_iid_context_config_dict["learner_config"]["dataset_config"][
            "dataset_kwargs"
        ].update(dataset_kwargs)
        icl_iid_context = parse_dict(icl_iid_context_config_dict)

        configs[f"{split}-icl_iid_context"] = icl_iid_context

        # Context length evaluations
        for prob_key in ["sample_high_prob_class_only", "sample_low_prob_class_only"]:
            for fixed_start_pos in range(context_len):
                for flip_label in [False, True]:
                    start_pos_config_dict = copy.deepcopy(config_dict)
                    modify_seed(start_pos_config_dict)

                    dataset_kwargs = {
                        prob_key: 1,
                        "fixed_start_pos": fixed_start_pos,
                        "mode": "default",
                        "flip_label": flip_label,
                    }

                    start_pos_config_dict["learner_config"]["dataset_config"][
                        "dataset_kwargs"
                    ].update(dataset_kwargs)

                    start_pos_config = parse_dict(start_pos_config_dict)
                    configs[
                        "{}-{}-start_pos_{}{}".format(
                            split,
                            prob_key,
                            fixed_start_pos,
                            "-flip_label" if flip_label else "",
                        )
                    ] = start_pos_config

    return {
        eval_name: get_data_loader(config, config.learner_config.seeds.data_seed)
        for eval_name, config in configs.items()
    }, configs


def main(args: SimpleNamespace):
    learner_path = args.learner_path
    batch_size = args.batch_size
    num_eval_samples = args.num_eval_samples
    test_data_seed = args.test_data_seed

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
            samples=[next(data_iter) for _ in range(num_eval_samples // batch_size)],
            dataset_output_dim=dataset.output_dim[0],
        )

    accuracies = {eval_name: [] for eval_name in datasets}
    losses = {eval_name: [] for eval_name in datasets}
    auxes = {eval_name: [] for eval_name in datasets}
    checkpoint_steps = []
    for params, model, checkpoint_step in tqdm(
        iterate_models(train_dataset.input_dim, train_dataset.output_dim, learner_path)
    ):
        checkpoint_steps.append(checkpoint_step)
        for eval_name in datasets:
            dataset, data_loader = datasets[eval_name]
            acc, loss, aux = evaluate(
                model=model,
                params=params,
                prefetched_data=prefetched_data[eval_name],
                max_label=None,
                context_len=context_len,
                fixed_length=fixed_length,
            )
            accuracies[eval_name].append(acc)
            losses[eval_name].append(loss)
            auxes[eval_name].append(aux)

    pickle.dump(
        {
            "checkpoint_steps": checkpoint_steps,
            "accuracies": accuracies,
            "losses": losses,
            "auxes": auxes,
        },
        open(
            os.path.join(learner_path, "evaluation.pkl"),
            "wb",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learner_path",
        type=str,
        required=True,
        help="The experiment run to load from",
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
