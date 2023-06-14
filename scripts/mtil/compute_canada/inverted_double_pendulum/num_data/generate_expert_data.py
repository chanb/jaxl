# TODO: Update example command
""" Script for generating experiment for multitask imitation learning

Example command:
python generate_expert_data.py \
    --run_path=/home/chanb/scratch/jaxl/data/inverted_double_pendulum/expert_models/gravity/runs/0/gravity_-8.249612491943623/06-09-23_15_21_56-296b3f54-5c33-43f3-97dd-3b7eb184bc99 \
    --out_dir=/home/chanb/scratch/jaxl/data/inverted_double_pendulum/expert_data \
    --exp_name=gravity-num_data_analysis \
    --run_seed=0 \
    --env_seed=42 \
    --num_episodes_variants=200,400,800 \
    --max_episode_length=1000


This will generate a dat file that consists of various runs.
"""

from absl import app, flags
from absl.flags import FlagValues

import jax
import os


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_name",
    default=None,
    help="Experiment name",
    required=True,
)
flags.DEFINE_integer("run_seed", default=None, help="Seed for the run", required=True)
flags.DEFINE_string(
    "run_path",
    default=None,
    help="The run directory",
    required=True,
)
flags.DEFINE_string(
    "out_dir",
    default=None,
    help="Directory for storing the experiment files",
    required=True,
)
flags.DEFINE_integer(
    "env_seed",
    default=None,
    help="Environment seed for resetting episodes",
    required=False,
)
flags.DEFINE_list(
    "num_episodes_variants",
    default=None,
    help="Number of evaluation episodes per variant",
    required=True,
)
flags.DEFINE_integer(
    "max_episode_length", default=1000, help="Maximum episode length", required=False
)


def main(config: FlagValues):
    out_dir = os.path.join(config.out_dir, config.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    num_episodes_variants = [
        int(num_episodes) for num_episodes in config.num_episodes_variants
    ]

    dat_content = ""
    for idx, num_episodes in enumerate(num_episodes_variants):
        save_id = os.path.basename(
            os.path.abspath(os.path.join(config.run_path, os.pardir))
        )
        save_buffer = os.path.join(
            out_dir,
            f"{save_id}-{os.path.basename(config.run_path)}-num_episodes_{num_episodes}.gzip",
        )
        dat_content += (
            "export buffer_size={} num_episodes={} env_seed={} run_seed={} ".format(
                config.max_episode_length * num_episodes,
                num_episodes,
                config.env_seed,
                config.run_seed,
            )
        )
        dat_content += "save_buffer={} run_path={}\n".format(
            save_buffer, config.run_path
        )
    with open(
        os.path.join(f"./export-generate_expert_data-{config.exp_name}.dat"), "w+"
    ) as f:
        f.writelines(dat_content)


if __name__ == "__main__":
    jax.config.config_with_absl()

    def _main(argv):
        """
        Generates experimental scripts for creating experts' data located in a directory.

        :param argv: the arguments provided by the user

        """
        del argv
        main(FLAGS)

    app.run(_main)
