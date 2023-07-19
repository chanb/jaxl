"""
This script performs hyperparameter search on multiple environment variations.

Example command:
# DMCCheetah-v0
python single_hyperparam_robustness.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/single_hyperparam_robustness \
    --run_seed=0 \
    --env_name=DMCCheetah-v0 \
    --exp_name=cheetah \
    --discrete_control \
    --num_envs=5 \
    --num_runs=5


Then, to generate the data, run the generated script run_all-*.sh ${run_seed}
"""

from absl import app, flags
from functools import partial

import itertools
import jax
import json
import numpy as np
import os

from search_config import HYPERPARAMETERS_CONFIG, POLICY_CONFIG


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
    "num_epochs", default=1000, help="The number of epochs to run the algorithm"
)
flags.DEFINE_string("run_time", default="05:00:00", help="The run time per variant")
flags.DEFINE_string(
    "env_name", default=None, required=True, help="The environment name"
)
flags.DEFINE_integer("num_envs", default=5, help="The number of environment variations")
flags.DEFINE_integer("num_runs", default=5, help="The number of runs per variation")


NUM_FILES_PER_DIRECTORY = 100


def set_ppo(template, key=None, val=None, hyperparam_keys=None, hyperparam_map=None):
    assert (key is not None) != (
        hyperparam_keys is not None and hyperparam_map is not None
    )
    if key is not None:
        if key == "objective":
            template["learner_config"]["pi_loss_setting"]["objective"] = val
    elif hyperparam_keys is not None:
        if "model" in hyperparam_keys:
            template["model_config"]["policy"]["layers"] = hyperparam_map("model")
            template["model_config"]["vf"]["layers"] = hyperparam_map("model")

        if "lr" in hyperparam_keys:
            template["optimizer_config"]["policy"]["lr"]["scheduler_kwargs"][
                "value"
            ] = hyperparam_map("lr")
            template["optimizer_config"]["vf"]["lr"]["scheduler_kwargs"][
                "value"
            ] = hyperparam_map("lr")

        if "max_grad_norm" in hyperparam_keys:
            template["optimizer_config"]["policy"]["max_grad_norm"] = hyperparam_map(
                "max_grad_norm"
            )
            template["optimizer_config"]["vf"]["max_grad_norm"] = hyperparam_map(
                "max_grad_norm"
            )

        if "obs_rms" in hyperparam_keys:
            template["learner_config"]["obs_rms"] = hyperparam_map("obs_rms")

        if "value_rms" in hyperparam_keys:
            template["learner_config"]["value_rms"] = hyperparam_map("value_rms")

        if "opt_batch_size" in hyperparam_keys:
            template["learner_config"]["opt_batch_size"] = hyperparam_map(
                "opt_batch_size"
            )

        if "opt_epochs" in hyperparam_keys:
            template["learner_config"]["opt_epochs"] = hyperparam_map("opt_epochs")

        if "vf_clip_param" in hyperparam_keys:
            template["learner_config"]["vf_loss_setting"][
                "clip_param"
            ] = hyperparam_map("vf_clip_param")

        if "ent_coef" in hyperparam_keys:
            template["learner_config"]["ent_loss_setting"] = hyperparam_map("ent_coef")

        if "beta" in hyperparam_keys:
            template["learner_config"]["pi_loss_setting"]["beta"] = hyperparam_map(
                "beta"
            )

        if "clip_param" in hyperparam_keys:
            template["learner_config"]["pi_loss_setting"][
                "clip_param"
            ] = hyperparam_map("clip_param")


def main(config):
    assert os.path.isfile(config.main_path), f"{config.main_path} is not a file"
    assert os.path.isfile(
        config.config_template
    ), f"{config.config_template} is not a file"
    with open(config.config_template, "r") as f:
        template = json.load(f)

    assert (
        config.num_epochs > 0
    ), f"num_epochs needs to be at least 1, got {config.num_epochs}"
    assert (
        config.num_envs > 0
    ), f"num_envs needs to be at least 1, got {config.num_envs}"
    assert (
        config.num_runs > 0
    ), f"num_runs needs to be at least 1, got {config.num_runs}"
    assert (
        len(config.run_time.split(":")) == 3
    ), f"run_time needs to be in format hh:mm:ss, got {config.run_time}"

    os.makedirs(config.out_dir, exist_ok=True)

    # Only use the default environmental parameters for hyperparameter search.
    # Should generalize well anyway
    template["train_config"]["num_epochs"] = config.num_epochs

    algo = template["learner_config"]["learner"]
    if algo == "ppo":
        template_setter = set_ppo
    else:
        raise ValueError(f"{algo} not supported")

    # Set action-space specific hyperparameters
    control_mode = "discrete" if config.discrete_control else "continuous"
    template["learner_config"]["policy_distribution"] = POLICY_CONFIG[algo][
        control_mode
    ]["policy_distribution"]

    for key, val in POLICY_CONFIG[algo][control_mode].items():
        if key == "hyperparameters":
            continue

        template_setter(template=template, key=key, val=val)

    # Set environment configuration
    template["learner_config"]["env_config"]["env_name"] = config.env_name
    template["learner_config"]["env_config"]["env_kwargs"] = {
        "use_default": False,
        "control_mode": control_mode,
    }

    template["logging_config"]["checkpoint_interval"] = False

    rng = np.random.RandomState(config.run_seed)
    model_seeds = rng.permutation(2**10)[: config.num_runs]
    env_seeds = rng.permutation(2**10)[: config.num_envs]
    # Hyperparameter list
    hyperparamss = (
        list(HYPERPARAMETERS_CONFIG[algo].values())
        + list(POLICY_CONFIG[algo][control_mode]["hyperparameters"].values())
        + [env_seeds, model_seeds]
    )
    hyperparam_keys = (
        list(HYPERPARAMETERS_CONFIG[algo].keys())
        + list(POLICY_CONFIG[algo][control_mode]["hyperparameters"].keys())
        + ["env_seed", "model_seed"]
    )

    def map_key_to_hyperparameter(hyperparams, key):
        hyperparam_idx = hyperparam_keys.index(key)
        return hyperparams[hyperparam_idx]

    # Create config per setting
    base_script_dir = os.path.join(
        config.out_dir, config.exp_name, control_mode, "scripts"
    )
    base_run_dir = os.path.join(config.out_dir, config.exp_name, control_mode, "runs")
    dat_content = ""
    num_runs = 0
    for idx, hyperparams in enumerate(itertools.product(*hyperparamss)):
        hyperparam_map = partial(map_key_to_hyperparameter, hyperparams)
        dir_i = str(idx // NUM_FILES_PER_DIRECTORY)
        curr_script_dir = os.path.join(base_script_dir, dir_i)
        curr_run_dir = os.path.join(base_run_dir, dir_i)
        if idx % NUM_FILES_PER_DIRECTORY == 0:
            os.makedirs(curr_run_dir, exist_ok=True)
            os.makedirs(curr_script_dir, exist_ok=True)

        template_setter(
            template=template,
            hyperparam_keys=hyperparam_keys,
            hyperparam_map=hyperparam_map,
        )

        if "buffer_size" in hyperparam_keys:
            template["learner_config"]["buffer_config"]["buffer_size"] = hyperparam_map(
                "buffer_size"
            )

        template["learner_config"]["env_config"]["env_kwargs"][
            "env_seed"
        ] = hyperparam_map("env_seed")
        template["learner_config"]["seeds"]["env_seed"] = hyperparam_map("env_seed")
        template["learner_config"]["seeds"]["model_seed"] = hyperparam_map("model_seed")

        variant = f"variant-{idx}"
        template["logging_config"]["experiment_name"] = variant
        template["logging_config"]["save_path"] = curr_run_dir

        out_path = os.path.join(curr_script_dir, variant)
        with open(f"{out_path}.json", "w+") as f:
            json.dump(template, f)

        num_runs += 1
        dat_content += "export run_seed={} ".format(config.run_seed)
        dat_content += "config_path={}.json \n".format(out_path)

    dat_path = os.path.join(
        f"./export-single_hyperparam_robustness-{config.exp_name}_{control_mode}.dat"
    )
    with open(dat_path, "w+") as f:
        f.writelines(dat_content)

    os.makedirs(
        "/home/chanb/scratch/run_reports/single_hyperparam_robustness-{}_{}".format(
            config.exp_name, control_mode
        ),
        exist_ok=True,
    )
    sbatch_content = ""
    sbatch_content += "#!/bin/bash\n"
    sbatch_content += "#SBATCH --account=def-schuurma\n"
    sbatch_content += "#SBATCH --time={}\n".format(config.run_time)
    sbatch_content += "#SBATCH --cpus-per-task=1\n"
    sbatch_content += "#SBATCH --mem=3G\n"
    sbatch_content += "#SBATCH --array=1-{}\n".format(num_runs)
    sbatch_content += "#SBATCH --output=/home/chanb/scratch/run_reports/single_hyperparam_robustness-{}_{}/%j.out\n".format(
        config.exp_name, control_mode
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

    with open(
        os.path.join(f"./run_all-{config.exp_name}_{control_mode}.sh"), "w+"
    ) as f:
        f.writelines(sbatch_content)


if __name__ == "__main__":
    jax.config.config_with_absl()

    def _main(argv):
        del argv
        main(FLAGS)

    app.run(_main)
