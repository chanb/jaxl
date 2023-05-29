""" Script for generating experiment for multitask imitation learning

Example command:
python generate_experts.py \
    --main_path=/Users/chanb/research/personal/jaxl/jaxl/main.py \
    --config_template=/Users/chanb/research/personal/jaxl/jaxl/configs/parameterized_envs/inverted_pendulum/template-generate_expert-reinforce.json \
    --exp_name=inverted_pendulum \
    --out_dir=/Users/chanb/research/personal/jaxl/data/inverted_pendulum \
    --num_model_seeds=1 \
    --num_env_seeds=1 \
    --num_envs=1000 \
    --min_gravity=-9.0 \
    --max_gravity=-8.0


Then, to generate the data, run the generated script run_all-*.sh
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
    default="../jaxl/main.py",
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


def main(config):
    assert os.path.isfile(
        config.config_template
    ), f"{config.config_template} is not a file"
    with open(config.config_template, "r") as f:
        template = json.load(f)

    os.makedirs(config.out_dir, exist_ok=True)

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

    assert config.num_envs > 0, f"need at least one environment variant, got {config.num_envs}"
    assert config.min_gravity <= config.max_gravity, f"min_gravity {config.min_gravity} should be less than max_gravity {config.max_gravity}"
    gravities = np.ones(config.num_envs) * config.min_gravity
    if config.min_gravity < config.max_gravity:
        gravities = np.random.uniform(config.min_gravity, config.max_gravity, size=config.num_envs)

    # Standard template
    template["logging_config"]["experiment_name"] = ""

    base_script_dir = os.path.join(config.out_dir, "scripts")
    base_log_dir = os.path.join(config.out_dir, "logs")
    base_run_dir = os.path.join(config.out_dir, "runs")
    shell_script = "run_seed=$1\n"
    for idx, (env_seed, model_seed, gravity) in enumerate(
        itertools.product(env_seeds, model_seeds, gravities)
    ):
        dir_i = str(idx // NUM_FILES_PER_DIRECTORY)
        curr_script_dir = os.path.join(base_script_dir, dir_i)
        curr_log_dir = os.path.join(base_log_dir, dir_i)
        curr_run_dir = os.path.join(base_run_dir, dir_i)
        if idx % NUM_FILES_PER_DIRECTORY == 0:
            os.makedirs(curr_log_dir, exist_ok=True)
            os.makedirs(curr_run_dir, exist_ok=True)
            os.makedirs(curr_script_dir, exist_ok=True)

        variant = f"variant-env_seed_{env_seed}-model_seed_{model_seed}-gravity_{gravity}"
        template["learner_config"]["seeds"]["model_seed"] = int(model_seed)
        template["learner_config"]["seeds"]["env_seed"] = int(env_seed)
        template["learner_config"]["env_config"]["env_kwargs"]["gravity"] = gravity
        template["logging_config"]["save_path"] = curr_run_dir

        out_path = os.path.join(curr_script_dir, variant)
        with open(f"{out_path}.json", "w+") as f:
            json.dump(template, f)

        shell_script += "python {} --config_path={}.json --run_seed=${{run_seed}} > {}.logs 2>&1 \n".format(
            config.main_path, out_path, os.path.join(curr_log_dir, variant)
        )
    with open(os.path.join(f"./run_all-{config.exp_name}.sh"), "w+") as f:
        f.writelines(shell_script)


if __name__ == "__main__":
    jax.config.config_with_absl()

    def _main(argv):
        del argv
        main(FLAGS)

    app.run(_main)
