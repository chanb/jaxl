""" Script for generating (number of tasks) experiment for multitask BC

Example command:
python generate_test_mtbc.py \
    --config_template=/home/chanb/scratch/jaxl/jaxl/configs/parameterized_envs/inverted_double_pendulum/template-test_mtbc.json \
    --exp_name=gravity-representation_analysis \
    --run_seed=0 \
    --test_data_path=/home/chanb/scratch/jaxl/data/inverted_double_pendulum/expert_data/gravity/gravity_-8.249612491943623-06-09-23_15_21_56-296b3f54-5c33-43f3-97dd-3b7eb184bc99.gzip \
    --runs_dir=/home/chanb/scratch/jaxl/data/inverted_double_pendulum/train_mtbc/gravity-representation_analysis/runs \
    --out_dir=/home/chanb/scratch/jaxl/data/inverted_double_pendulum/test_mtbc/ \
    --num_model_seeds=1


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
    "test_data_path",
    default=None,
    help="Path to load the expert data from",
    required=True,
)
flags.DEFINE_string(
    "runs_dir",
    default=None,
    help="Directory to load the saved representations from (recursively)",
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

NUM_FILES_PER_DIRECTORY = 100


def main(config: FlagValues):
    assert os.path.isfile(
        config.config_template
    ), f"{config.config_template} is not a file"
    with open(config.config_template, "r") as f:
        template = json.load(f)

    out_dir = os.path.join(config.out_dir, config.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    assert os.path.isfile(
        config.test_data_path
    ), f"expert dataset path {config.test_data_path} not found"

    assert (
        config.num_model_seeds > 0
    ), f"need at least one model_seed, got {config.num_model_seeds}"
    while True:
        model_seeds = np.random.randint(0, 2**32 - 1, config.num_model_seeds)
        if len(model_seeds) == len(np.unique(model_seeds)):
            break

    runs_info = []
    for root, _, filenames in os.walk(config.runs_dir):
        for filename in filenames:
            if filename != "config.json":
                continue

            run_path = root
            with open(os.path.join(root, filename), "r") as f:
                curr_run_config = json.load(f)
                num_tasks = len(curr_run_config["learner_config"]["buffer_configs"])
                encoder_architecture = curr_run_config["model_config"]["encoder"]
                experiment_name = curr_run_config["logging_config"]["experiment_name"]
            runs_info.append((run_path, num_tasks, encoder_architecture, experiment_name))

    # Standard template
    template["logging_config"]["experiment_name"] = ""

    base_script_dir = os.path.join(out_dir, "scripts")
    base_run_dir = os.path.join(out_dir, "runs")
    dat_content = ""
    for idx, (model_seed, run_info) in enumerate(
        itertools.product(model_seeds, runs_info)
    ):
        (run_path, num_tasks, encoder_architecture, experiment_name) = run_info
        dir_i = str(idx // NUM_FILES_PER_DIRECTORY)
        curr_script_dir = os.path.join(base_script_dir, dir_i)
        curr_run_dir = os.path.join(base_run_dir, dir_i)
        if idx % NUM_FILES_PER_DIRECTORY == 0:
            os.makedirs(curr_run_dir, exist_ok=True)
            os.makedirs(curr_script_dir, exist_ok=True)

        num_layers = int(experiment_name.split("num_layers_")[1].split("-")[0])
        num_hidden_units = int(experiment_name.split("num_hidden_units_")[1].split("-")[0])
        variant = f"variant-model_seed_{model_seed}-num_tasks_{num_tasks}-num_layers_{num_layers}-num_hidden_units_{num_hidden_units}"
        template["learner_config"]["seeds"]["model_seed"] = int(model_seed)
        template["learner_config"]["load_encoder"] = os.path.join(
            run_path, "termination_model"
        )

        template["model_config"]["encoder"] = encoder_architecture

        template["learner_config"]["buffer_configs"][0][
            "load_buffer"
        ] = config.test_data_path
        template["logging_config"]["save_path"] = curr_run_dir
        template["logging_config"]["experiment_name"] = experiment_name

        out_path = os.path.join(curr_script_dir, variant)
        with open(f"{out_path}.json", "w+") as f:
            json.dump(template, f)

        dat_content += "export run_seed={} ".format(config.run_seed)
        dat_content += "config_path={}.json \n".format(out_path)
    with open(
        os.path.join(f"./export-generate_test_mtbc-{config.exp_name}.dat"),
        "w+",
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
