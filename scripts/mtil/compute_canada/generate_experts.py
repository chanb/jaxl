"""
This script generates an expert for each environment variant.

Example command:
python generate_experts.py \
    --config_template=${JAXL_PATH}/jaxl/configs/parameterized_envs/inverted_double_pendulum/template-generate_expert-ppo.json \
    --exp_name=gravity \
    --run_seed=0 \
    --out_dir=${HOME}/scratch/jaxl/data/inverted_double_pendulum/expert_models \
    --num_model_seeds=1 \
    --num_envs=100 \
    --run_time=02:00:00

python generate_experts.py \
    --config_template=${JAXL_PATH}/jaxl/configs/parameterized_envs/inverted_double_pendulum/template-generate_expert-ppo.json \
    --exp_name=gravity \
    --run_seed=0 \
    --out_dir=${HOME}/scratch/jaxl/data/inverted_double_pendulum/expert_models \
    --num_model_seeds=1 \
    --num_envs=100 \
    --run_time=02:00:00 \
    --discrete_control


This will generate a dat file that consists of various runs.
"""

from absl import app, flags
from absl.flags import FlagValues

import itertools
import jax
import json
import numpy as np
import os


FLAGS = flags.FLAGS
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
flags.DEFINE_integer("run_seed", default=None, help="Seed for the run", required=True)
flags.DEFINE_string(
    "out_dir",
    default=None,
    help="Directory for storing the experiment files",
    required=True,
)
flags.DEFINE_integer(
    "num_model_seeds",
    default=None,
    help="The number of seeds for initializing model parameters",
    required=True,
)
flags.DEFINE_integer(
    "num_envs",
    default=None,
    help="The number of environment variations",
    required=True,
)
flags.DEFINE_boolean(
    "discrete_control", default=False, help="Whether or not to use discrete control"
)

NUM_FILES_PER_DIRECTORY = 100


def main(config: FlagValues):
    assert os.path.isfile(
        config.config_template
    ), f"{config.config_template} is not a file"
    with open(config.config_template, "r") as f:
        template = json.load(f)

    out_dir = os.path.join(config.out_dir, config.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    assert (
        config.num_model_seeds > 0
    ), f"need at least one model_seed, got {config.num_model_seeds}"
    while True:
        model_seeds = np.random.randint(0, 2**32 - 1, config.num_model_seeds)
        if len(model_seeds) == len(np.unique(model_seeds)):
            break

    assert (
        config.num_envs > 0
    ), f"need at least one environment variant, got {config.num_envs}"
    env_seeds = np.arange(config.num_envs)

    # Standard template
    template["logging_config"]["experiment_name"] = ""

    base_script_dir = os.path.join(out_dir, "scripts")
    base_run_dir = os.path.join(out_dir, "runs")
    dat_content = ""
    for idx, (env_seed, model_seed) in enumerate(
        itertools.product(env_seeds, model_seeds)
    ):
        dir_i = str(idx // NUM_FILES_PER_DIRECTORY)
        curr_script_dir = os.path.join(base_script_dir, dir_i)
        curr_run_dir = os.path.join(base_run_dir, dir_i)
        if idx % NUM_FILES_PER_DIRECTORY == 0:
            os.makedirs(curr_run_dir, exist_ok=True)
            os.makedirs(curr_script_dir, exist_ok=True)

        variant = f"variant-{idx}-env_seed_{env_seed}-model_seed_{model_seed}"
        template["learner_config"]["seeds"]["model_seed"] = int(model_seed)
        template["logging_config"]["save_path"] = curr_run_dir
        template["logging_config"]["experiment_name"] = variant

        template["learner_config"]["env_config"]["env_kwargs"]["use_default"] = False
        template["learner_config"]["env_config"]["env_kwargs"]["seed"] = int(env_seed)
        if config.discrete_control:
            template["learner_config"]["env_config"]["env_kwargs"][
                "discrete_control"
            ] = True
            template["learner_config"]["policy_distribution"] = "softmax"
        else:
            template["learner_config"]["env_config"]["env_kwargs"][
                "discrete_control"
            ] = False
            template["learner_config"]["policy_distribution"] = "gaussian"

        out_path = os.path.join(curr_script_dir, variant)
        with open(f"{out_path}.json", "w+") as f:
            json.dump(template, f)

        dat_content += "export run_seed={} ".format(config.run_seed)
        dat_content += "config_path={}.json \n".format(out_path)

    dat_path = os.path.join(f"./export-generate_experts-{config.exp_name}.dat")
    with open(dat_path, "w+") as f:
        f.writelines(dat_content)

    os.makedirs(
        "/home/chanb/scratch/run_reports/generate_experts-{}".format(config.exp_name),
        exist_ok=True,
    )
    sbatch_content = ""
    sbatch_content += "#!/bin/bash\n"
    sbatch_content += "#SBATCH --account=def-schuurma\n"
    sbatch_content += "#SBATCH --time={}\n".format(config.run_time)
    sbatch_content += "#SBATCH --cpus-per-task=1\n"
    sbatch_content += "#SBATCH --mem=3G\n"
    sbatch_content += "#SBATCH --array=1-{}\n".format(num_runs)
    sbatch_content += "#SBATCH --output=/home/chanb/scratch/run_reports/generate_experts-{}/%j.out\n".format(
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
        """
        Generates experimental scripts for creating experts with different hyperparameters.

        :param argv: the arguments provided by the user

        """
        del argv
        main(FLAGS)

    app.run(_main)
