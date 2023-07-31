import _pickle as pickle
import gymnasium as gym
import gzip
import jax
import json
import numpy as np
import os

from jaxl.utils import flatten_dict

import jaxl.envs

env_seed = 127

base_dir = "/Users/chanb/research/personal/mtil_results/final_results/data/"
finetune_dir = "finetune_mtbc_main"
pretrain_dir = "pretrain_mtbc_main"
task = "pendulum"
control_mode = "discrete"
target_env_dir = f"ParameterizedPendulum-v0.control_mode_discrete.env_seed_{env_seed}.model_seed_927.num_samples_100000.subsampling_length_200"
num_taskss = ["num_tasks_{}".format(num_task) for num_task in [1, 2, 4, 8, 16]]
pretrain_model_seed = "pretrained_model_seed_27"
eps = 1e-3


def l2_distance(finetune_config, pretrain_config):
    target_env_config = finetune_config["learner_config"]["env_configs"][0]
    target_env_config["env_name"] = target_env_config["env_name"].split("/")[-1]
    target_env = gym.make(target_env_config["env_name"], **target_env_config["env_kwargs"])
    target_env_params = {k: v for k, v in flatten_dict(target_env.get_config())}
    target_env.close()

    source_env_configs = pretrain_config["learner_config"]["env_configs"]
    source_env_paramss = []
    for source_env_config in source_env_configs:
        source_env_config["env_name"] = source_env_config["env_name"].split("/")[-1]
        source_env = gym.make(source_env_config["env_name"], **source_env_config["env_kwargs"])
        source_env_paramss.append({k: v for k, v in flatten_dict(source_env.get_config())})
        source_env.close()

    target_env_vec = np.array(list(target_env_params.values()))
    source_env_vecs = np.array([
        list(source_env_params.values()) for source_env_params in source_env_paramss
    ])
    
    pairwise_distance = np.sum((source_env_vecs - target_env_vec[:, None]) ** 2, axis=-1)
    avg_distance = np.mean(pairwise_distance)
    std_distance = np.std(pairwise_distance)
    min_distance = np.min(pairwise_distance)
    return avg_distance, std_distance, min_distance


def approx_kl():
    pass


def expert_data_performance(finetune_config, pretrain_config, finetune_dataset_path, pretrain_dataset_paths):
    target_env_buffer = pickle.load(gzip.open(finetune_dataset_path, "rb"))
    target_buffer_size = finetune_config["learner_config"]["buffer_configs"][0]["set_size"]
    target_data_performance = np.mean(target_env_buffer["rewards"][:target_buffer_size])

    source_data_performances = []
    for buffer_config, pretrain_dataset_path in zip(pretrain_config["learner_config"]["buffer_configs"], pretrain_dataset_paths):
        source_env_buffer = pickle.load(gzip.open(pretrain_dataset_path, "rb"))
        source_buffer_size = buffer_config["set_size"]
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
        pretrain_model_seed
    )

    finetune_run_dir = os.path.join(finetune_run_dir, os.listdir(finetune_run_dir)[-1])

    dataset_dir = "/Users/chanb/research/personal/mtil_results/final_results/data/expert_data"
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
        os.path.basename(os.path.dirname(finetune_config["learner_config"]["load_encoder"])),
    )
    with open(os.path.join(pretrain_run_dir, "config.json"), "r") as f:
        pretrain_config = json.load(f)

    finetune_dataset_path = os.path.join(
        dataset_dir,
        dataset_task_name,
        os.path.basename(finetune_config["learner_config"]["buffer_configs"][0]["load_buffer"])
    )

    pretrain_dataset_paths = [
        os.path.join(
            dataset_dir,
            dataset_task_name,
            os.path.basename(buffer_config["load_buffer"])
        ) for buffer_config in pretrain_config["learner_config"]["buffer_configs"]
    ]

    avg_distance, std_distance, min_distance = l2_distance(finetune_config, pretrain_config)
    l2_diversity = 1 - jax.nn.sigmoid(avg_distance)
    print(l2_diversity)


    avg_distance, std_distance, min_distance = expert_data_performance(
        finetune_config,
        pretrain_config,
        finetune_dataset_path,
        pretrain_dataset_paths
    )
    data_performance_diversity = 1 - jax.nn.sigmoid(avg_distance)
    print(data_performance_diversity)

