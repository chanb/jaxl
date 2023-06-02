""" Script for generating (number of tasks) experiment for multitask BC

Example command:
python generate_mtbc.py \
    --config_template=/home/chanb/scratch/jaxl/jaxl/configs/parameterized_envs/inverted_pendulum/template-mtbc.json \
    --exp_name=gravity-num_tasks_analysis \
    --run_seed=0 \
    --datasets_dir=/home/chanb/scratch/jaxl/data/inverted_pendulum/expert_data/gravity \
    --out_dir=/home/chanb/scratch/jaxl/data/inverted_pendulum/train_mtbc/ \
    --num_model_seeds=1 \
    --num_tasks_variants=1,2,4,8,16,32,64


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
    "datasets_dir",
    default=None,
    help="The directory to load the expert data from (not recursively)",
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
flags.DEFINE_list(
    "num_tasks_variants",
    default=None,
    help="A list of number of tasks",
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

    datasets_path = [
        os.path.join(config.datasets_dir, dataset_path)
        for dataset_path in os.listdir(config.datasets_dir)
        if dataset_path.endswith(".gzip")
    ]

    assert (
        len(datasets_path) > 0
    ), f"need at least one dataset, got {len(datasets_path)}"

    assert (
        config.num_model_seeds > 0
    ), f"need at least one model_seed, got {config.num_model_seeds}"
    while True:
        model_seeds = np.random.randint(0, 2**32 - 1, config.num_model_seeds)
        if len(model_seeds) == len(np.unique(model_seeds)):
            break

    num_tasks_variants = np.array(
        [int(num_tasks) for num_tasks in config.num_tasks_variants]
    )
    assert np.all(num_tasks_variants > 0), f"need at least one task for MTBC"

    # Standard template
    template["logging_config"]["experiment_name"] = ""

    base_script_dir = os.path.join(out_dir, "scripts")
    base_run_dir = os.path.join(out_dir, "runs")
    dat_content = ""
    for idx, (model_seed, num_tasks) in enumerate(
        itertools.product(model_seeds, num_tasks_variants)
    ):
        dir_i = str(idx // NUM_FILES_PER_DIRECTORY)
        curr_script_dir = os.path.join(base_script_dir, dir_i)
        curr_run_dir = os.path.join(base_run_dir, dir_i)
        if idx % NUM_FILES_PER_DIRECTORY == 0:
            os.makedirs(curr_run_dir, exist_ok=True)
            os.makedirs(curr_script_dir, exist_ok=True)

        variant = f"variant-model_seed_{model_seed}-num_tasks_{num_tasks}"
        template["learner_config"]["seeds"]["model_seed"] = int(model_seed)
        template["learner_config"]["buffer_configs"] = [
            {
                "load_buffer": datasets_path[task_i],
                "buffer_type": "default",
            }
            for task_i in range(num_tasks)
        ]
        template["model_config"]["predictor"]["num_models"] = int(num_tasks)
        template["logging_config"]["save_path"] = curr_run_dir
        template["logging_config"]["experiment_name"] = f"num_tasks_{num_tasks}"

        out_path = os.path.join(curr_script_dir, variant)
        with open(f"{out_path}.json", "w+") as f:
            json.dump(template, f)

        dat_content += "export run_seed={} ".format(config.run_seed)
        dat_content += "config_path={}.json \n".format(out_path)
    with open(
        os.path.join(f"./export-generate_mtbc-{config.exp_name}.dat"),
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
