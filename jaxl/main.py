"""
This script is the entrypoint for any experiment.
XXX: Try not to modify this.
"""
from absl import app, flags
from datetime import datetime
from pprint import pprint

import jax
import json
import logging
import os
import timeit
import uuid

from jaxl.learning_utils import get_learner, train
from jaxl.utils import flatten_dict, parse_dict, set_seed

CONST_CPU = "cpu"
CONST_GPU = "gpu"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config_path", default=None, help="Training configuration", required=True
)
flags.DEFINE_integer("run_seed", default=None, help="Seed for the run", required=False)
flags.DEFINE_string(
    "device",
    default=CONST_CPU,
    help="JAX device to use. To specify specific GPU device, do gpu:<device_ids>",
    required=False,
)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


"""
This function constructs the model, optimizer, and learner and executes training.
"""


def main(config_path: str, run_seed: int = None, device: str = CONST_CPU):
    """
    Orchestrates the experiment.

    :param config_path: the experiment configuration file path
    :param run_seed: the seed to initialize the random number generators
    :param device: the JAX device to use, supports [`cpu`, `gpu:<device_ids>`]
    :type config_path: str
    :type run_seed: int: (Default value = None)
    :type device: str: (Default value = cpu)

    """
    tic = timeit.default_timer()

    (device_name, *device_ids) = device.split(":")
    if device_name == CONST_CPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device_name == CONST_GPU:
        assert (
            len(device_ids) > 0
        ), f"at least one device_id is needed, got {device_ids}"
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids[0]
    else:
        raise ValueError(f"{device_name} is not a supported device.")

    set_seed(run_seed)
    assert os.path.isfile(config_path), f"{config_path} is not a file"
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        hyperparameter_str = "|param|value|\n|-|-|\n%s" % (
            "\n".join(
                [
                    f"|{key}|{value}|"
                    for key, value in dict(flatten_dict(config_dict)).items()
                ]
            )
        )
        config = parse_dict(config_dict)

    pprint(config)

    save_path = None
    if config.logging_config.save_path:
        optional_prefix = ""
        if config.logging_config.experiment_name:
            optional_prefix += f"{config.logging_config.experiment_name}-"
        time_tag = datetime.strftime(datetime.now(), "%m-%d-%y_%H_%M_%S")
        run_id = str(uuid.uuid4())
        save_path = os.path.join(
            config.logging_config.save_path, f"{optional_prefix}{time_tag}-{run_id}"
        )
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "config.json"), "w+") as f:
            json.dump(config_dict, f)

    learner = get_learner(
        config.learner_config, config.model_config, config.optimizer_config
    )
    train(learner, config, hyperparameter_str, save_path)
    toc = timeit.default_timer()
    print(f"Experiment Time: {toc - tic}s")


if __name__ == "__main__":
    jax.config.config_with_absl()

    def _main(argv):
        """
        Runs the experiment.

        :param argv: the arguments provided by the user

        """
        del argv
        config_path = FLAGS.config_path
        seed = FLAGS.run_seed
        device = FLAGS.device
        main(config_path, seed, device)

    app.run(_main)
