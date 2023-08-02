from itertools import chain
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager
from typing import Iterable

import _pickle as pickle
import gymnasium as gym
import gzip
import jax
import jax.numpy as jnp
import json
import numpy as np
import os

from jaxl.losses import get_loss_function
from jaxl.models import get_model
from jaxl.utils import flatten_dict, parse_dict

import jaxl.envs

env_seed = 259
base_dir = "/Users/chanb/research/personal/mtil_results/final_results/data/"
finetune_dir = "finetune_mtbc_main"
pretrain_dir = "pretrain_mtbc_main"
task = "cheetah"
control_mode = "discrete"
target_env_dir = f"DMCCheetah-v0.control_mode_discrete.env_seed_{env_seed}.model_seed_927.num_samples_100000.subsampling_length_1000"
num_taskss_int = np.array([1, 2, 4, 8, 16])
num_taskss = ["num_tasks_{}".format(num_task) for num_task in num_taskss_int]
pretrain_model_seed = "pretrained_model_seed_27"
eps = 1e-3

# env_seed = 127
# base_dir = "/Users/chanb/research/personal/mtil_results/final_results/data/"
# finetune_dir = "finetune_mtbc_main"
# pretrain_dir = "pretrain_mtbc_main"
# task = "pendulum"
# control_mode = "discrete"
# target_env_dir = f"ParameterizedPendulum-v0.control_mode_discrete.env_seed_{env_seed}.model_seed_927.num_samples_100000.subsampling_length_200"
# num_taskss_int = np.array([1, 2, 4, 8, 16])
# num_taskss = ["num_tasks_{}".format(num_task) for num_task in num_taskss_int]
# pretrain_model_seed = "pretrained_model_seed_27"
# eps = 1e-3


def l2_distance(finetune_config, pretrain_config):
    target_env_config = finetune_config["learner_config"]["env_configs"][0]
    target_env_config["env_name"] = target_env_config["env_name"].split("/")[-1]
    target_env = gym.make(
        target_env_config["env_name"], **target_env_config["env_kwargs"]
    )
    target_env_params = target_env.get_config()["modified_attributes"]
    target_env.close()

    source_env_configs = pretrain_config["learner_config"]["env_configs"]
    source_env_paramss = []
    for source_env_config in source_env_configs:
        source_env_config["env_name"] = source_env_config["env_name"].split("/")[-1]
        source_env = gym.make(
            source_env_config["env_name"], **source_env_config["env_kwargs"]
        )
        source_env_paramss.append(source_env.get_config()["modified_attributes"])
        source_env.close()

    target_env_vec = np.array(
        list(
            chain(
                *[
                    val if isinstance(val, Iterable) else [val]
                    for val in jax.tree_util.tree_leaves(target_env_params)
                ]
            )
        )
    )
    source_env_vecs = np.array(
        [
            list(
                chain(
                    *[
                        val if isinstance(val, Iterable) else [val]
                        for val in jax.tree_util.tree_leaves(source_env_params)
                    ]
                )
            )
            for source_env_params in source_env_paramss
        ]
    )

    pairwise_distance = np.sum(
        (source_env_vecs - target_env_vec[None, :]) ** 2, axis=-1
    )
    avg_distance = np.mean(pairwise_distance)
    std_distance = np.std(pairwise_distance)
    min_distance = np.min(pairwise_distance)
    return avg_distance, std_distance, min_distance


def approx_kl(
    finetune_run_dir,
    pretrain_run_dir,
    finetune_config,
    pretrain_config,
    finetune_dataset_path,
    pretrain_dataset_paths,
):
    target_env_buffer = pickle.load(gzip.open(finetune_dataset_path, "rb"))
    # target_buffer_size = finetune_config["learner_config"]["buffer_configs"][0][
    #     "set_size"
    # ]
    target_buffer_size = 10000

    source_statess = []
    source_h_statess = []
    source_actionss = []
    num_tasks = 0
    for buffer_config, pretrain_dataset_path in zip(
        pretrain_config["learner_config"]["buffer_configs"], pretrain_dataset_paths
    ):
        source_env_buffer = pickle.load(gzip.open(pretrain_dataset_path, "rb"))
        # source_buffer_size = buffer_config["set_size"]
        source_buffer_size = 10000
        source_statess.append(source_env_buffer["observations"][:source_buffer_size])
        source_h_statess.append(source_env_buffer["hidden_states"][:source_buffer_size])
        source_actionss.append(source_env_buffer["actions"][:source_buffer_size])
        num_tasks += 1

    source_statess = np.stack(source_statess)
    source_h_statess = np.stack(source_h_statess)
    source_actionss = np.stack(source_actionss)

    target_states = np.tile(
        target_env_buffer["observations"][:target_buffer_size][None, ...],
        (num_tasks, 1, 1),
    )
    target_h_states = np.tile(
        target_env_buffer["hidden_states"][:target_buffer_size][None, ...],
        (num_tasks, 1, 1),
    )
    target_actions = np.tile(
        target_env_buffer["actions"][:target_buffer_size][None, ...], (num_tasks, 1, 1)
    )

    model = get_model(
        target_states.shape[1:],
        target_env_buffer["act_dim"],
        parse_dict(pretrain_config["model_config"]),
    )

    checkpoint_manager = CheckpointManager(
        os.path.join(pretrain_run_dir, "models"),
        PyTreeCheckpointer(),
    )
    all_params = checkpoint_manager.restore(checkpoint_manager.latest_step())
    model_dict = all_params["model_dict"]

    loss_fn = get_loss_function(
        model.predictor.model,
        pretrain_config["learner_config"]["losses"][0],
        parse_dict(pretrain_config["learner_config"]["loss_settings"][0]),
        num_classes=target_env_buffer["act_dim"][-1],
    )

    def loss(
        model_dicts,
        obss,
        h_states,
        acts,
        *args,
        **kwargs,
    ):
        reps, _ = jax.vmap(model.encode, in_axes=[None, 0, 0])(
            model_dicts["encoder"], obss, h_states
        )

        bc_loss, bc_aux = jax.vmap(loss_fn)(
            model_dicts["predictor"],
            reps,
            h_states,
            acts,
        )

        return bc_loss, bc_aux

    compute_source_loss = jax.jit(loss)
    compute_target_loss = jax.jit(loss)

    source_loss, _ = compute_source_loss(
        model_dict["model"]["policy"],
        source_statess,
        source_h_statess,
        source_actionss,
    )

    target_loss, _ = compute_target_loss(
        model_dict["model"]["policy"],
        target_states,
        target_h_states,
        target_actions,
    )
    best_target_loss = jnp.min(target_loss)

    print(source_loss)
    print(target_loss)

    diversity = jnp.mean(source_loss) / (best_target_loss + eps)
    return diversity


def expert_data_performance(
    finetune_config, pretrain_config, finetune_dataset_path, pretrain_dataset_paths
):
    target_env_buffer = pickle.load(gzip.open(finetune_dataset_path, "rb"))
    # target_buffer_size = finetune_config["learner_config"]["buffer_configs"][0][
    #     "set_size"
    # ]
    target_buffer_size = 10000
    target_data_performance = np.mean(target_env_buffer["rewards"][:target_buffer_size])

    source_data_performances = []
    for buffer_config, pretrain_dataset_path in zip(
        pretrain_config["learner_config"]["buffer_configs"], pretrain_dataset_paths
    ):
        source_env_buffer = pickle.load(gzip.open(pretrain_dataset_path, "rb"))
        # source_buffer_size = buffer_config["set_size"]
        source_buffer_size = 10000
        source_data_performances.append(
            np.mean(source_env_buffer["rewards"][:source_buffer_size])
        )

    source_data_performances = np.array(source_data_performances)

    pairwise_distance = source_data_performances - target_data_performance

    avg_distance = np.mean(pairwise_distance)
    std_distance = np.std(pairwise_distance)
    min_distance = np.min(pairwise_distance)
    return avg_distance, std_distance, min_distance


print("TARGET ENV SEED: {}".format(target_env_dir.split(".")[2]))

l2_diversities = []
data_performance_diversities = []
approx_kl_diversities = []

for num_tasks in num_taskss:
    print(f"NUM TASKS: {num_tasks}")
    finetune_run_dir = os.path.join(
        base_dir,
        finetune_dir,
        task,
        control_mode,
        "runs",
        target_env_dir,
        num_tasks,
        pretrain_model_seed,
    )

    finetune_run_dir = os.path.join(finetune_run_dir, os.listdir(finetune_run_dir)[-1])

    dataset_dir = (
        "/Users/chanb/research/personal/mtil_results/final_results/data/expert_data"
    )
    dataset_task_name = "{}_{}".format(task, control_mode[:4])

    assert os.path.isdir(finetune_run_dir)

    with open(os.path.join(finetune_run_dir, "config.json"), "r") as f:
        finetune_config = json.load(f)

    pretrain_run_dir = os.path.join(
        base_dir,
        pretrain_dir,
        task,
        control_mode,
        "runs",
        num_tasks,
        os.path.basename(
            os.path.dirname(finetune_config["learner_config"]["load_encoder"])
        ),
    )
    with open(os.path.join(pretrain_run_dir, "config.json"), "r") as f:
        pretrain_config = json.load(f)

    finetune_dataset_path = os.path.join(
        dataset_dir,
        dataset_task_name,
        os.path.basename(
            finetune_config["learner_config"]["buffer_configs"][0]["load_buffer"]
        ),
    )

    pretrain_dataset_paths = [
        os.path.join(
            dataset_dir,
            dataset_task_name,
            os.path.basename(buffer_config["load_buffer"]),
        )
        for buffer_config in pretrain_config["learner_config"]["buffer_configs"]
    ]

    avg_distance, std_distance, min_distance = l2_distance(
        finetune_config, pretrain_config
    )
    l2_diversity = 1 - jax.nn.sigmoid(avg_distance)
    print(l2_diversity)

    avg_distance, std_distance, min_distance = expert_data_performance(
        finetune_config, pretrain_config, finetune_dataset_path, pretrain_dataset_paths
    )
    data_performance_diversity = 1 - jax.nn.sigmoid(avg_distance)
    print(data_performance_diversity)

    kl_diversity = approx_kl(
        finetune_run_dir,
        pretrain_run_dir,
        finetune_config,
        pretrain_config,
        finetune_dataset_path,
        pretrain_dataset_paths,
    )
    print(kl_diversity)

    l2_diversities.append(l2_diversity)
    data_performance_diversities.append(data_performance_diversity)
    approx_kl_diversities.append(kl_diversity)

print(num_taskss_int[np.argsort(l2_diversities)])
print(num_taskss_int[np.argsort(data_performance_diversities)])
print(num_taskss_int[np.argsort(approx_kl_diversities)])
