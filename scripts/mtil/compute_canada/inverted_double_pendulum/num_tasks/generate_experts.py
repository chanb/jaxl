""" Script for generating experiment for multitask imitation learning

Example command:
python generate_experts.py \
    --config_template=/home/chanb/scratch/jaxl/jaxl/configs/parameterized_envs/inverted_double_pendulum/template-generate_expert-ppo.json \
    --exp_name=gravity \
    --run_seed=0 \
    --out_dir=/home/chanb/scratch/jaxl/data/inverted_double_pendulum/expert_models \
    --num_model_seeds=1 \
    --num_env_seeds=1 \
    --num_envs=100 \
    --min_gravity=-15.0 \
    --max_gravity=-5.0


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
    "num_env_seeds",
    default=None,
    help="The number of seeds for initializing environments",
    required=True,
)
flags.DEFINE_integer(
    "num_envs",
    default=None,
    help="The number of environment variations",
    required=True,
)
flags.DEFINE_float(
    "min_gravity",
    default=-9.81,
    help="the minimum gravity to sample from",
)
flags.DEFINE_float(
    "max_gravity",
    default=-9.81,
    help="the maximum gravity to sample from",
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
        config.num_env_seeds > 0
    ), f"need at least one env_seed, got {config.num_env_seeds}"
    while True:
        env_seeds = np.random.randint(0, 2**32 - 1, config.num_env_seeds)
        if len(env_seeds) == len(np.unique(env_seeds)):
            break

    assert (
        config.num_envs > 0
    ), f"need at least one environment variant, got {config.num_envs}"
    assert (
        config.min_gravity <= config.max_gravity
    ), f"min_gravity {config.min_gravity} should be less than max_gravity {config.max_gravity}"
    gravities = np.ones(config.num_envs) * config.min_gravity
    if config.min_gravity < config.max_gravity:
        gravities = np.random.uniform(
            config.min_gravity, config.max_gravity, size=config.num_envs
        )

    # Standard template
    template["logging_config"]["experiment_name"] = ""

    base_script_dir = os.path.join(out_dir, "scripts")
    base_run_dir = os.path.join(out_dir, "runs")
    dat_content = ""
    for idx, (env_seed, model_seed, gravity) in enumerate(
        itertools.product(env_seeds, model_seeds, gravities)
    ):
        dir_i = str(idx // NUM_FILES_PER_DIRECTORY)
        curr_script_dir = os.path.join(base_script_dir, dir_i)
        curr_run_dir = os.path.join(base_run_dir, dir_i)
        if idx % NUM_FILES_PER_DIRECTORY == 0:
            os.makedirs(curr_run_dir, exist_ok=True)
            os.makedirs(curr_script_dir, exist_ok=True)

        variant = (
            f"variant-env_seed_{env_seed}-model_seed_{model_seed}-gravity_{gravity}"
        )
        template["learner_config"]["seeds"]["model_seed"] = int(model_seed)
        template["learner_config"]["seeds"]["env_seed"] = int(env_seed)
        template["learner_config"]["env_config"]["env_kwargs"]["gravity"] = gravity
        template["logging_config"]["save_path"] = curr_run_dir
        template["logging_config"]["experiment_name"] = f"gravity_{gravity}"

        out_path = os.path.join(curr_script_dir, variant)
        with open(f"{out_path}.json", "w+") as f:
            json.dump(template, f)

        dat_content += "export run_seed={} ".format(config.run_seed)
        dat_content += "config_path={}.json \n".format(out_path)
    with open(
        os.path.join(f"./export-generate_experts-{config.exp_name}.dat"), "w+"
    ) as f:
        f.writelines(dat_content)


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
