"""
This script is the entrypoint for evaluating RL-trained policies.
XXX: Try not to modify this.
"""
from absl import app, flags
from absl.flags import FlagValues

import _pickle as pickle
import jax
import logging
import numpy as np
import os
import timeit

from jaxl.constants import (
    CONST_EPISODE_LENGTHS,
    CONST_EPISODIC_RETURNS,
    CONST_RUN_PATH,
    CONST_BUFFER_PATH,
)
from jaxl.envs.rollouts import EvaluationRollout
from jaxl.learning_utils import load_latest_agent
from jaxl.utils import set_seed


FLAGS = flags.FLAGS
flags.DEFINE_string("run_path", default=None, help="The saved run", required=True)
flags.DEFINE_integer("run_seed", default=None, help="Seed for the run", required=False)
flags.DEFINE_integer(
    "num_episodes", default=None, help="Number of evaluation episodes", required=True
)
flags.DEFINE_string(
    "save_stats",
    default=None,
    help="Where to save the episodic statistics",
    required=False,
)
flags.DEFINE_string(
    "save_buffer",
    default=None,
    help="Where to save the evaluation episodes",
    required=False,
)
flags.DEFINE_integer(
    "buffer_size", default=1000, help="Number of transitions to store", required=False
)
flags.DEFINE_integer(
    "env_seed",
    default=None,
    help="Environment seed for resetting episodes",
    required=False,
)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


"""
This function constructs the model and executes evaluation.
"""


def main(
    config: FlagValues,
):
    """Orchestrates the evaluation."""
    tic = timeit.default_timer()
    set_seed(config.run_seed)
    assert os.path.isdir(config.run_path), f"{config.run_path} is not a directory"

    policy, policy_params, obs_rms, buffer, env, env_seed = load_latest_agent(
        config.run_path, config.buffer_size
    )

    if config.env_seed is not None:
        env_seed = config.env_seed

    rollout = EvaluationRollout(env, seed=env_seed)
    rollout.rollout(policy_params, policy, obs_rms, config.num_episodes, buffer)
    if config.save_buffer:
        print("Saving buffer with {} transitions".format(len(buffer)))
        buffer.save(config.save_buffer)

    if config.save_stats:
        print("Saving episodic statistics")
        with open(config.save_stats, "wb") as f:
            pickle.dump(
                {
                    CONST_EPISODIC_RETURNS: rollout.episodic_returns,
                    CONST_EPISODE_LENGTHS: rollout.episode_lengths,
                    CONST_RUN_PATH: config.run_path,
                    CONST_BUFFER_PATH: config.save_buffer,
                },
                f,
            )

    toc = timeit.default_timer()
    print(
        "Expected return: {} -- Expected episode length: {}".format(
            np.mean(rollout.episodic_returns), np.mean(rollout.episode_lengths)
        )
    )
    print(f"Evaluation Time: {toc - tic}s")


if __name__ == "__main__":
    jax.config.config_with_absl()

    def _main(argv):
        """
        Runs the evaluation.

        :param argv: the arguments provided by the user

        """
        del argv
        main(FLAGS)

    app.run(_main)
