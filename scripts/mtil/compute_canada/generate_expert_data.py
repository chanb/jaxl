"""
This script generates the expert data for each provided model.

Example command:
python generate_expert_data.py \
    --main_path=${JAXL_PATH}/jaxl/evaluate_rl_agents.py \
    --runs_dir=${HOME}/scratch/data/expert_models/pendulum_disc \
    --exp_name=pendulum_disc \
    --run_seed=0 \
    --env_seed=0 \
    --out_dir=${HOME}/scratch/jaxl/data/expert_data \
    --num_episodes=1000 \
    --max_episode_length=1000 \
    --run_time=01:00:00


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
    "main_path",
    default="../../../../jaxl/evaluate_rl_agents.py",
    help="Path to evaluate_rl_agents.py",
    required=False,
)
flags.DEFINE_string(
    "runs_dir",
    default=None,
    help="The directory storing the runs",
    required=True,
)
flags.DEFINE_string(
    "exp_name",
    default=None,
    help="Experiment name",
    required=True,
)
flags.DEFINE_integer("run_seed", default=None, help="Seed for the run", required=True)
flags.DEFINE_integer(
    "env_seed", default=None, help="Seed for the environment", required=True
)
flags.DEFINE_string(
    "out_dir",
    default=None,
    help="Directory for storing the experiment files",
    required=True,
)
flags.DEFINE_integer(
    "num_episodes",
    default=None,
    help="The number of demonstration episodes to store",
    required=True,
)
flags.DEFINE_integer(
    "max_episode_length",
    default=None,
    help="The maximum length of an episode of the MDP",
    required=True,
)
flags.DEFINE_string("run_time", default="03:00:00", help="The run time per variant")

NUM_FILES_PER_DIRECTORY = 100


def main(config: FlagValues):
    assert os.path.isfile(config.main_path), f"{config.main_path} is not a file"
    assert os.path.isdir(config.runs_dir), f"{config.runs_dir} is not a directory"
    assert (
        len(config.run_time.split(":")) == 3
    ), f"run_time needs to be in format hh:mm:ss, got {config.run_time}"
    assert os.path.isfile(
        config.config_template
    ), f"{config.config_template} is not a file"
    with open(config.config_template, "r") as f:
        template = json.load(f)

    assert (
        config.max_episode_length > 0
    ), f"max_episode_length should be at least 0, got {config.max_episode_length}"
    assert (
        config.num_episodes > 0
    ), f"num_episodes shoudl be at least 0, got {config.num_episodes}"

    out_dir = os.path.join(config.out_dir, config.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    buffer_size = config.num_episodes * config.max_episode_length

    dat_content = ""
    num_runs = 0
    for run_path, _, filenames in os.walk(config.runs_dir):
        for filename in filenames:
            if filename != "config.json":
                continue

            save_id = os.path.basename(
                os.path.abspath(os.path.join(run_path, os.pardir))
            )
            save_buffer = os.path.join(
                out_dir,
                f"{save_id}-{os.path.basename(run_path)}-num_episodes_{config.num_episodes}.gzip",
            )
            dat_content += (
                "export buffer_size={} num_episodes={} env_seed={} run_seed={} ".format(
                    buffer_size,
                    config.num_episodes,
                    config.env_seed,
                    config.run_seed,
                )
            )
            dat_content += "save_buffer={} run_path={}\n".format(
                save_buffer, config.run_path
            )
            num_runs += 1

    dat_path = os.path.join(f"./export-generate_expert_data-{config.exp_name}.dat")
    with open(dat_path, "w+") as f:
        f.writelines(dat_content)

    os.makedirs(
        "/home/chanb/scratch/run_reports/generate_expert_data-{}".format(
            config.exp_name
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
    sbatch_content += "#SBATCH --output=/home/chanb/scratch/run_reports/generate_expert_data-{}/%j.out\n".format(
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

    with open(
        os.path.join(f"./run_all-generate_expert_data-{config.exp_name}.sh"), "w+"
    ) as f:
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
