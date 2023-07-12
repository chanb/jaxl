"""
This script performs hyperparameter search on an environment.

Example command:
# ParameterizedPendulum-v0
python search_expert.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/jaxl/configs/classic_control/pendulum/local-ppo.json \
    --exp_name=search_expert-pendulum_cont \
    --out_dir=${HOME}/scratch/data/pendulum_cont/search_expert \
    --run_seed=0 \
    --num_epochs=500 \
    --run_time=02:00:00

python search_expert.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/jaxl/configs/classic_control/pendulum/local-ppo.json \
    --exp_name=search_expert-pendulum_disc \
    --out_dir=${HOME}/scratch/data/pendulum_disc/search_expert \
    --run_seed=0 \
    --num_epochs=500 \
    --run_time=02:00:00 \
    --discrete_control


Then, to generate the data, run the generated script run_all-*.sh ${run_seed}
"""

from absl import app, flags
from functools import partial

import itertools
import jax
import json
import numpy as np
import os


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "main_path",
    default="../../../../jaxl/main.py",
    help="Path to main.py",
    required=False,
)
flags.DEFINE_string(
    "config_template",
    default=None,
    help="Training configuration template",
    required=True,
)
flags.DEFINE_string(
    "exp_name",
    default=None,
    help="Experiment name",
    required=True,
)
flags.DEFINE_string(
    "out_dir",
    default=None,
    help="Directory for storing the experiment files",
    required=True,
)
flags.DEFINE_integer(
    "run_seed",
    default=0,
    help="The run seed",
    required=False,
)
flags.DEFINE_boolean(
    "discrete_control", default=False, help="Whether or not to use discrete control"
)
flags.DEFINE_integer(
    "num_epochs", default=300, help="The number of epochs to run the algorithm"
)
flags.DEFINE_string("run_time", default="03:00:00", help="The run time per variant")


NUM_FILES_PER_DIRECTORY = 100

HYPERPARAMETERS_CONFIG = {
    "ParameterizedPendulum-v0": {
        "model": [[64, 64], [128, 128]],
        "lr": [3e-4, 1e-3],
        "max_grad_norm": [False, 0.5],
        "obs_rms": [False, True],
        "opt_batch_size": [64, 128, 256],
        "opt_epochs": [4, 10],
        "vf_clip_param": [False, 0.2],
        "ent_coef": [
            {
                "scheduler": "constant_schedule",
                "scheduler_kwargs": {"value": 0.0},
            },
            {
                "scheduler": "linear_schedule",
                "scheduler_kwargs": {
                    "init_value": 0.002,
                    "end_value": 0.0,
                    "transition_begin": 0,
                    "transition_steps": 100,
                },
            },
        ],
    },
    "ParameterizedHopper-v0": {
        "model": [[128, 128]],
        "lr": [3e-4, 1e-3],
        "max_grad_norm": [False, 0.5],
        "obs_rms": [True],
        "opt_batch_size": [64, 128, 256, 512],
        "opt_epochs": [4, 10],
        "vf_clip_param": [False, 0.2],
        "value_rms": [False, True],
        "ent_coef": [
            {
                "scheduler": "constant_schedule",
                "scheduler_kwargs": {"value": 0.0},
            },
            {
                "scheduler": "linear_schedule",
                "scheduler_kwargs": {
                    "init_value": 0.002,
                    "end_value": 0.0,
                    "transition_begin": 0,
                    "transition_steps": 100,
                },
            },
        ],
    },
}


def main(config):
    assert os.path.isfile(config.main_path), f"{config.main_path} is not a file"
    assert os.path.isfile(
        config.config_template
    ), f"{config.config_template} is not a file"
    with open(config.config_template, "r") as f:
        template = json.load(f)

    assert template["learner_config"]["env_config"]["env_name"] in HYPERPARAMETERS_CONFIG, "{} has no hyperparameters config".format(template["learner_config"]["env_config"]["env_name"])

    assert (
        config.num_epochs > 0
    ), f"num_epochs needs to be at least 1, got {config.num_epochs}"
    assert (
        len(config.run_time.split(":")) == 3
    ), f"run_time needs to be in format hh:mm:ss, got {config.run_time}"

    os.makedirs(config.out_dir, exist_ok=True)

    # Standard template
    template["logging_config"]["experiment_name"] = ""

    hyperparamss = list(HYPERPARAMETERS_CONFIG[template["learner_config"]["env_config"]["env_name"]].values())
    hyperparam_keys = list(HYPERPARAMETERS_CONFIG[template["learner_config"]["env_config"]["env_name"]].keys())

    def map_key_to_hyperparameter(hyperparams, key):
        hyperparam_idx = hyperparam_keys.index(key)
        return hyperparam_keys[hyperparam_idx]

    base_script_dir = os.path.join(config.out_dir, "scripts")
    base_log_dir = os.path.join(config.out_dir, "logs")
    base_run_dir = os.path.join(config.out_dir, "runs")

    dat_content = ""
    num_runs = 0

    template["train_config"]["num_epochs"] = config.num_epochs
    template["learner_config"]["env_config"]["env_kwargs"]["use_default"] = True
    if config.discrete_control:
        template["learner_config"]["env_config"]["env_kwargs"][
            "discrete_control"
        ] = True
        template["learner_config"]["env_config"]["env_kwargs"][
            "control_mode"
        ] = "discrete"
        template["learner_config"]["policy_distribution"] = "softmax"
    else:
        template["learner_config"]["env_config"]["env_kwargs"][
            "discrete_control"
        ] = False
        template["learner_config"]["env_config"]["env_kwargs"][
            "control_mode"
        ] = "default"
        template["learner_config"]["policy_distribution"] = "gaussian"

    for idx, hyperparams in enumerate(itertools.product(*hyperparamss)):
        get_hyperparam = partial(map_key_to_hyperparameter, hyperparams)
        dir_i = str(idx // NUM_FILES_PER_DIRECTORY)
        curr_script_dir = os.path.join(base_script_dir, dir_i)
        curr_run_dir = os.path.join(base_run_dir, dir_i)
        if idx % NUM_FILES_PER_DIRECTORY == 0:
            os.makedirs(curr_run_dir, exist_ok=True)
            os.makedirs(curr_script_dir, exist_ok=True)

        if "model" in hyperparam_keys:
            template["model_config"]["policy"]["layers"] = get_hyperparam("model")
            template["model_config"]["vf"]["layers"] = get_hyperparam("model")
        
        if "lr" in hyperparam_keys:
            template["optimizer_config"]["policy"]["lr"]["scheduler_kwargs"][
                "value"
            ] = get_hyperparam("lr")
            template["optimizer_config"]["vf"]["lr"]["scheduler_kwargs"][
                "value"
            ] = get_hyperparam("lr")

        if "max_grad_norm" in hyperparam_keys:
            template["optimizer_config"]["policy"]["max_grad_norm"] = get_hyperparam("max_grad_norm")
            template["optimizer_config"]["vf"]["max_grad_norm"] = get_hyperparam("max_grad_norm")

        if "obs_rms" in hyperparam_keys:
            template["learner_config"]["obs_rms"] = get_hyperparam("obs_rms")

        if "value_rms" in hyperparam_keys:
            template["learner_config"]["value_rms"] = get_hyperparam("value_rms")

        if "opt_batch_size" in hyperparam_keys:
            template["learner_config"]["opt_batch_size"] = get_hyperparam("opt_batch_size")

        if "opt_epochs" in hyperparam_keys:
            template["learner_config"]["opt_epochs"] = get_hyperparam("opt_epochs")

        if "vf_clip_param" in hyperparam_keys:
            template["learner_config"]["vf_loss_setting"]["clip_param"] = get_hyperparam("vf_clip_param")

        if "ent_coef" in hyperparam_keys:
            template["learner_config"]["ent_loss_setting"] = get_hyperparam("ent_coef")

        variant = f"variant-{idx}"
        template["logging_config"]["experiment_name"] = variant
        template["logging_config"]["save_path"] = curr_run_dir

        out_path = os.path.join(curr_script_dir, variant)
        with open(f"{out_path}.json", "w+") as f:
            json.dump(template, f)

        num_runs += 1
        dat_content += "export run_seed={} ".format(config.run_seed)
        dat_content += "config_path={}.json \n".format(out_path)

    dat_path = os.path.join(f"./export-search_expert-{config.exp_name}.dat")
    with open(dat_path, "w+") as f:
        f.writelines(dat_content)

    os.makedirs(
        "/home/chanb/scratch/run_reports/search_expert-{}".format(config.exp_name),
        exist_ok=True,
    )
    sbatch_content = ""
    sbatch_content += "#!/bin/bash\n"
    sbatch_content += "#SBATCH --account=def-schuurma\n"
    sbatch_content += "#SBATCH --time={}\n".format(config.run_time)
    sbatch_content += "#SBATCH --cpus-per-task=1\n"
    sbatch_content += "#SBATCH --mem=3G\n"
    sbatch_content += "#SBATCH --array=1-{}\n".format(num_runs)
    sbatch_content += "#SBATCH --output=/home/chanb/scratch/run_reports/search_expert-{}/%j.out\n".format(
        config.exp_name
    )
    sbatch_content += "module load python/3.9\n"
    sbatch_content += "module load mujoco\n"
    sbatch_content += "source ~/jaxl_env/bin/activate\n"
    sbatch_content += '`sed -n "${SLURM_ARRAY_TASK_ID}p"'
    sbatch_content += " < {}`\n".format(dat_path)
    sbatch_content += "echo ${SLURM_ARRAY_TASK_ID}\n"
    sbatch_content += 'echo "Current working directory is `pwd`"\n'
    sbatch_content += 'echo "Running on hostname `hostname`"\n'
    sbatch_content += 'echo "Starting run at: `date`"\n'
    sbatch_content += "python3 {} \\\n".format(config.main_path)
    sbatch_content += "  --config_path=${config_path} \\\n"
    sbatch_content += "  --run_seed=${run_seed}\n"
    sbatch_content += 'echo "Program test finished with exit code $? at: `date`"\n'

    with open(os.path.join(f"./run_all-{config.exp_name}.sh"), "w+") as f:
        f.writelines(sbatch_content)


if __name__ == "__main__":
    jax.config.config_with_absl()

    def _main(argv):
        del argv
        main(FLAGS)

    app.run(_main)
