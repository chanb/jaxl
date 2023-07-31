"""
This script performs hyperparameter search on multiple environment variations.

Then, to generate the data, run the generated script run_all-*.sh ${run_seed}
"""

from absl import app, flags
from functools import partial

import _pickle as pickle
import itertools
import jax
import json
import numpy as np
import os

from search_config import HYPERPARAM_SETS
from utils import set_mtbc


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
    "num_variations",
    default=0,
    help="The number of experiments of varying source tasks",
    required=False,
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
flags.DEFINE_string("run_time", default="00:20:00", help="The run time per variant")
flags.DEFINE_integer("num_runs", default=1, help="The number of runs per variation")
flags.DEFINE_string(
    "data_dir",
    default=None,
    required=True,
    help="The directory storing the expert datasets",
)
flags.DEFINE_list(
    "num_tasks_variants",
    default=None,
    help="A list of number of tasks",
    required=True,
)
flags.DEFINE_integer(
    "num_heldouts", default=None, required=True, help="The set of held-out tasks"
)
flags.DEFINE_string(
    "hyperparam_set",
    default=None,
    required=True,
    help="The hyperparameter configuration set to use, see search_config.py",
)
flags.DEFINE_integer(
    "checkpoint_interval",
    default=None,
    help="The frequency to checkpoint the model",
)
flags.DEFINE_integer(
    "num_samples",
    default=None,
    required=True,
    help="Amount of total data to use",
)


def main(config):
    assert os.path.isfile(config.main_path), f"{config.main_path} is not a file"
    assert os.path.isfile(
        config.config_template
    ), f"{config.config_template} is not a file"
    with open(config.config_template, "r") as f:
        template = json.load(f)

    assert (
        config.num_samples > 0
    ), f"num_samples needs to be at least 1, got {config.num_samples}"
    assert (
        config.num_runs > 0
    ), f"num_runs needs to be at least 1, got {config.num_runs}"
    assert (
        config.num_variations > 0
    ), f"num_variations needs to be at least 1, got {config.num_variations}"
    assert (
        len(config.run_time.split(":")) == 3
    ), f"run_time needs to be in format hh:mm:ss, got {config.run_time}"
    assert (
        config.hyperparam_set in HYPERPARAM_SETS
    ), f"{config.hyperparam_set} not in {HYPERPARAM_SETS.keys()}"

    assert os.path.isdir(config.data_dir), f"{config.data_dir} is not a directory"
    assert (
        config.num_heldouts > 0
    ), f"num_heldouts needs to be at least 1, got {config.num_heldouts}"

    num_tasks_variants = np.array(
        [int(num_tasks) for num_tasks in config.num_tasks_variants]
    )
    assert np.all(num_tasks_variants > 0), f"need at least one task for MTBC"

    # Gather expert datasets
    dataset_paths = []
    for data_path in os.listdir(config.data_dir)[config.num_heldouts :]:
        dataset_paths.append(os.path.join(config.data_dir, data_path))
    dataset_paths = sorted(dataset_paths)

    os.makedirs(config.out_dir, exist_ok=True)

    hyperparam_set = HYPERPARAM_SETS[config.hyperparam_set]

    algo = template["learner_config"]["learner"]
    if algo == "mtbc":
        template_setter = set_mtbc
    else:
        raise ValueError(f"{algo} not supported")

    # Set action-space specific hyperparameters
    control_mode = "discrete" if config.discrete_control else "continuous"
    for key, val in hyperparam_set[algo][control_mode].items():
        if key == "hyperparameters":
            continue

        template_setter(template=template, key=key, val=val)

    template["logging_config"]["checkpoint_interval"] = config.checkpoint_interval
    if config.checkpoint_interval:
        template["logging_config"]["checkpoint_interval"] = False

    rng = np.random.RandomState(config.run_seed)
    model_seeds = rng.permutation(2**10)[: config.num_runs]
    dataset_sample_seeds = rng.permutation(2**10)[: config.num_variations]

    # Hyperparameter list
    hyperparamss = (
        list(hyperparam_set[algo]["general"].values())
        + list(hyperparam_set[algo][control_mode]["hyperparameters"].values())
        + [model_seeds, dataset_sample_seeds, num_tasks_variants]
    )
    hyperparam_keys = (
        list(hyperparam_set[algo]["general"].keys())
        + list(hyperparam_set[algo][control_mode]["hyperparameters"].keys())
        + ["model_seed", "dataset_sample_seed", "num_tasks_variant"]
    )

    with open(
        os.path.join(
            config.out_dir,
            f"hyperparameters-pretrain_mtbc-vary_source_tasks-{config.hyperparam_set}-{config.exp_name}_{control_mode}.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump([hyperparam_keys, hyperparamss], f)

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

        template_setter(
            template=template,
            hyperparam_keys=hyperparam_keys,
            hyperparam_map=hyperparam_map,
        )

        dataset_sample_seed = int(hyperparam_map("dataset_sample_seed"))
        num_tasks = int(hyperparam_map("num_tasks_variant"))
        variant_name = "num_tasks_{}.dataset_sample_seed_{}".format(num_tasks, dataset_sample_seed)
        curr_script_dir = os.path.join(base_script_dir, variant_name)
        curr_run_dir = os.path.join(base_run_dir, variant_name)
        os.makedirs(curr_script_dir, exist_ok=True)
        os.makedirs(curr_run_dir, exist_ok=True)

        template["learner_config"]["buffer_configs"] = []
        template["learner_config"]["env_configs"] = []

        dataset_rng = np.random.RandomState(dataset_sample_seed)
        randomized_dataset_paths = dataset_rng.permutation(dataset_paths)
        for task_i in range(num_tasks):
            template["learner_config"]["buffer_configs"].append(
                {
                    "load_buffer": randomized_dataset_paths[task_i],
                    "buffer_type": "default",
                    "set_size": config.num_samples // num_tasks,
                }
            )

            dataset_info = randomized_dataset_paths[task_i].split(".")
            env_config = {
                "env_type": "gym",
                "env_name": dataset_info[0],
                "env_kwargs": {
                    "use_default": False,
                    "seed": int(dataset_info[2].split("env_seed_")[-1]),
                    "control_mode": control_mode,
                },
            }
            assert (
                control_mode == dataset_info[1].split("control_mode_")[-1]
            ), "control mode is inconsistent with dataset"
            template["learner_config"]["env_configs"].append(env_config)

        template["model_config"]["predictor"]["num_models"] = num_tasks

        model_seed = int(hyperparam_map("model_seed"))
        template["learner_config"]["seeds"]["buffer_seed"] = model_seed
        template["learner_config"]["seeds"]["model_seed"] = model_seed

        variant = "model_seed_{}".format(model_seed)
        template["logging_config"]["experiment_name"] = variant
        template["logging_config"]["save_path"] = curr_run_dir

        out_path = os.path.join(curr_script_dir, variant)
        with open(f"{out_path}.json", "w+") as f:
            json.dump(template, f)

        num_runs += 1
        dat_content += "export run_seed={} ".format(config.run_seed)
        dat_content += "config_path={}.json \n".format(out_path)

    dat_path = os.path.join(
        f"./export-pretrain-mtbc-vary_source_tasks-{config.hyperparam_set}-{config.exp_name}_{control_mode}.dat"
    )
    with open(dat_path, "w+") as f:
        f.writelines(dat_content)

    os.makedirs(
        "/home/chanb/scratch/run_reports/pretrain-mtbc-vary_source_tasks-{}-{}_{}".format(
            config.hyperparam_set, config.exp_name, control_mode
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
    sbatch_content += "#SBATCH --output=/home/chanb/scratch/run_reports/pretrain-mtbc-vary_source_tasks-{}-{}_{}/%j.out\n".format(
        config.hyperparam_set, config.exp_name, control_mode
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
        os.path.join(
            f"./run_all-pretrain-mtbc-vary_source_tasks-{config.hyperparam_set}-{config.exp_name}_{control_mode}.sh"
        ),
        "w+",
    ) as f:
        f.writelines(sbatch_content)


if __name__ == "__main__":
    jax.config.config_with_absl()

    def _main(argv):
        del argv
        main(FLAGS)

    app.run(_main)