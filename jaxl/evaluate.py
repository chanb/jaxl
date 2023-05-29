"""
This script is the entrypoint for evaluating policies.
XXX: Try not to modify this.
"""
from absl import app, flags
from datetime import datetime
from pprint import pprint

import _pickle as pickle
import jax
import json
import logging
import os
import timeit
import uuid

from learning_utils import get_learner, train
from jaxl.utils import flatten_dict, parse_dict, set_seed


FLAGS = flags.FLAGS
flags.DEFINE_string("run_path", default=None, help="The saved run", required=True)
flags.DEFINE_integer("run_seed", default=None, help="Seed for the run", required=False)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


"""
This function constructs the model and executes evaluation.
"""


def main(run_path: str, run_seed: int = None):
    """
    Orchestrates the evaluation.

    :param run_path: the saved run path
    :param run_seed: the seed to initialize the random number generators
    :type run_path: str
    :type run_seed: int: (Default value = None)

    """
    tic = timeit.default_timer()
    set_seed(run_seed)
    assert os.path.isdir(run_path), f"{run_path} is not a directory"

    # TODO: Load model and evaluation
    config_path = os.path.join(run_path, "config.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    learner = get_learner(
        config.learner_config, config.model_config, config.optimizer_config
    )
    learner.load_checkpoint(os.path.join(run_path, "termination_model"))

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
        run_path = FLAGS.run_path
        seed = FLAGS.run_seed
        main(run_path, seed)

    app.run(_main)
