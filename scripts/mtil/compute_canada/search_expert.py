""" Script for generating experiment for multitask imitation learning

Example command:
python search_expert.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/jaxl/configs/classic_control/pendulum/local-ppo.json \
    --exp_name=search_expert-pendulum_cont \
    --out_dir=${JAXL_PATH}/data/pendulum_cont/search_expert \
    --run_seed=0

python search_expert.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/jaxl/configs/classic_control/pendulum/local-ppo.json \
    --exp_name=search_expert-pendulum_disc \
    --out_dir=${JAXL_PATH}/data/pendulum_disc/search_expert \
    --run_seed=0 \
    --discrete_control


Then, to generate the data, run the generated script run_all-*.sh ${run_seed}
"""

from absl import app, flags

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
    "discrete_control",
    default=False,
    help="Whether or not to use discrete control"
)

NUM_FILES_PER_DIRECTORY = 100


def main(config):
    assert os.path.isfile(
        config.config_template
    ), f"{config.config_template} is not a file"
    with open(config.config_template, "r") as f:
        template = json.load(f)

    os.makedirs(config.out_dir, exist_ok=True)

    # Standard template
    template["logging_config"]["experiment_name"] = ""

    models = [[64, 64], [128, 128]]
    lrs = [3e-4, 1e-3]
    max_grad_norms = [False, 0.5, 10.0]
    obs_rmss = [False, True]
    opt_batch_sizes = [64, 128, 256]
    opt_epochss = [4, 10]
    vf_clip_params = [False, 0.2]
    ent_coefs = [
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
    ]
    hyperparamss = [
        models,
        lrs,
        max_grad_norms,
        obs_rmss,
        opt_batch_sizes,
        opt_epochss,
        vf_clip_params,
        ent_coefs,
    ]

    base_script_dir = os.path.join(config.out_dir, "scripts")
    base_log_dir = os.path.join(config.out_dir, "logs")
    base_run_dir = os.path.join(config.out_dir, "runs")

    dat_content = ""
    num_runs = 0
    for idx, hyperparams in enumerate(itertools.product(*hyperparamss)):
        dir_i = str(idx // NUM_FILES_PER_DIRECTORY)
        curr_script_dir = os.path.join(base_script_dir, dir_i)
        curr_log_dir = os.path.join(base_log_dir, dir_i)
        curr_run_dir = os.path.join(base_run_dir, dir_i)
        if idx % NUM_FILES_PER_DIRECTORY == 0:
            os.makedirs(curr_log_dir, exist_ok=True)
            os.makedirs(curr_run_dir, exist_ok=True)
            os.makedirs(curr_script_dir, exist_ok=True)

        template["model_config"]["policy"]["layers"] = hyperparams[0]
        template["model_config"]["vf"]["layers"] = hyperparams[0]
        template["optimizer_config"]["policy"]["lr"]["scheduler_kwargs"][
            "value"
        ] = hyperparams[1]
        template["optimizer_config"]["vf"]["lr"]["scheduler_kwargs"][
            "value"
        ] = hyperparams[1]
        template["optimizer_config"]["policy"]["max_grad_norm"] = hyperparams[2]
        template["optimizer_config"]["vf"]["max_grad_norm"] = hyperparams[2]
        template["learner_config"]["obs_rms"] = hyperparams[3]
        template["learner_config"]["opt_batch_size"] = hyperparams[4]
        template["learner_config"]["opt_epochs"] = hyperparams[5]
        template["learner_config"]["vf_loss_setting"]["clip_param"] = hyperparams[6]
        template["learner_config"]["ent_loss_setting"] = hyperparams[7]

        variant = f"variant-{idx}"
        template["logging_config"]["experiment_name"] = f"variant-{idx}"
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
        template["logging_config"]["save_path"] = curr_run_dir

        out_path = os.path.join(curr_script_dir, variant)
        with open(f"{out_path}.json", "w+") as f:
            json.dump(template, f)

        num_runs += 1
        dat_content += "export run_seed={} ".format(config.run_seed)
        dat_content += "config_path={}.json \n".format(out_path)

    with open(
        os.path.join(f"./export-search_expert-{config.exp_name}.dat"), "w+"
    ) as f:
        f.writelines(dat_content)

    os.makedirs()
    sbatch_content = "/home/chanb/scratch/run_reports/search_expert-{}".format(config.exp_name)
    sbatch_content += "#!/bin/bash \n"
    sbatch_content += "#SBATCH --account=def-schuurma \n"
    sbatch_content += "#SBATCH --time=06:00:00 \n"
    sbatch_content += "#SBATCH --cpus-per-task=1 \n"
    sbatch_content += "#SBATCH --mem=3G \n"
    sbatch_content += "#SBATCH --array=1:{} \n".format(num_runs)
    sbatch_content += "#SBATCH --output=/home/chanb/scratch/run_reports/search_expert-{}/%j.out \n".format(config.exp_name)
    sbatch_content += "module load python/3.9 \n"
    sbatch_content += 'module load mujoco \n'
    sbatch_content += 'source ~/jaxl_env/bin/activate \n'
    sbatch_content += '`sed -n "${SLURM_ARRAY_TASK_ID}p" < export-generate_expert_data-gravity-num_data_analysis.dat` \n'
    sbatch_content += 'echo ${SLURM_ARRAY_TASK_ID} \n'
    sbatch_content += 'echo "Current working directory is `pwd`" \n'
    sbatch_content += 'echo "Running on hostname `hostname`" \n'
    sbatch_content += 'echo "Starting run at: `date`" \n'
    sbatch_content += 'python3 /home/chanb/scratch/jaxl/jaxl/main.py \\ \n'
    sbatch_content += '--config_path=${config_path} \\ \n'
    sbatch_content += '--run_seed=${run_seed} \n'
    sbatch_content += 'echo "Program test finished with exit code $? at: `date`" \n'

    with open(os.path.join(f"./run_all-{config.exp_name}.sh"), "w+") as f:
        f.writelines(sbatch_content)


if __name__ == "__main__":
    jax.config.config_with_absl()

    def _main(argv):
        del argv
        main(FLAGS)

    app.run(_main)
