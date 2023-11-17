"""
This script generates the expert data for each provided model.

This will generate a dat file that consists of various runs.
"""

from absl import app, flags
from absl.flags import FlagValues

import jax
import json
import numpy as np
import os


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "main_path",
    default="./gather_metaworld_data.py",
    help="Path to gather_metaworld_data.py",
    required=False,
)
flags.DEFINE_string(
    "exp_name",
    default=None,
    help="Experiment name",
    required=True,
)
flags.DEFINE_integer("run_seed", default=None, help="Seed for the run", required=True)
flags.DEFINE_integer(
    "num_envs", default=None, help="Amount of environments to generate data from", required=True
)
flags.DEFINE_integer(
    "scrambling_step", default=None, help="The number of random initialization steps", required=True
)
flags.DEFINE_string(
    "out_dir",
    default=None,
    help="Directory for storing the experiment files",
    required=True,
)
flags.DEFINE_integer(
    "num_samples",
    default=None,
    help="The number of demonstration samples to store",
    required=True,
)
flags.DEFINE_integer(
    "max_episode_length",
    default=None,
    help="The maximum episode length",
    required=False,
)
flags.DEFINE_multi_integer(
    "subsampling_lengths",
    default=None,
    help="The lengths of subtrajectories to gather per episode",
    required=True,
)
flags.DEFINE_string(
    "run_time",
    default="03:00:00",
    help="The run time per variant",
    required=False,
)

NUM_FILES_PER_DIRECTORY = 100


def main(config: FlagValues):
    assert os.path.isfile(config.main_path), f"{config.main_path} is not a file"
    assert (
        len(config.run_time.split(":")) == 3
    ), f"run_time needs to be in format hh:mm:ss, got {config.run_time}"

    assert all(
        length > 0 for length in config.subsampling_lengths
    ), f"all of subsampling_lengths should be at least 1, got {config.subsampling_lengths}"
    assert (
        config.num_samples > 0
    ), f"num_samples should be at least 1, got {config.num_samples}"
    assert (
        config.scrambling_step >= 0
    ), f"scrambling_step should be at least 0, got {config.scrambling_step}"
    assert (
        config.num_envs > 0
    ), f"num_envs needs to be at least 1, got {config.num_envs}"
    assert (
        config.max_episode_length > 0
    ), f"max_episode_length needs to be at least 1, got {config.max_episode_length}"

    rng = np.random.RandomState(config.run_seed)
    env_seeds = rng.permutation(2**10)[: config.num_envs]

    out_dir = os.path.join(config.out_dir, config.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    dat_content = ""
    num_runs = 0
    for env_seed in env_seeds:
        for subsampling_length in config.subsampling_lengths:
            save_buffer = os.path.join(
                out_dir,
                (
                    f"metaworld-{config.exp_name}.env_seed_{env_seed}"
                    f".num_samples_{config.num_samples}.subsampling_length_{subsampling_length}.gzip"
                )
            )
            
            dat_content += (
                f"export env_seed={env_seed} scrambling_step={config.scrambling_step}"
                f" run_seed={config.run_seed} num_samples={config.num_samples}"
                f" subsampling_length={subsampling_length} save_buffer={save_buffer}"
                f" max_episode_length={config.max_episode_length}\n"
            )
            num_runs += 1

    dat_path = os.path.join(f"./export-generate_metaworld_data-{config.exp_name}.dat")
    with open(dat_path, "w+") as f:
        f.writelines(dat_content)

    os.makedirs(
        "/home/chanb/scratch/run_reports/generate_metaworld_data-{}".format(
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
    sbatch_content += "#SBATCH --output=/home/chanb/scratch/run_reports/generate_metaworld_data-{}/%j.out\n".format(
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
    sbatch_content += "  --env_seed=${env_seed} \\\n"
    sbatch_content += "  --run_seed=${run_seed} \\\n"
    sbatch_content += "  --max_episode_length=${max_episode_length} \\\n"
    sbatch_content += "  --num_samples=${num_samples} \\\n"
    sbatch_content += "  --subsampling_length=${subsampling_length} \\\n"
    sbatch_content += "  --scrambling_step=${scrambling_step} \\\n"
    sbatch_content += "  --save_buffer=${save_buffer}\n"
    sbatch_content += 'echo "Program test finished with exit code $? at: `date`"\n'

    with open(
        os.path.join(f"./run_all-generate_metaworld_data-{config.exp_name}.sh"), "w+"
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
