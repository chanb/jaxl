"""
This script is the entrypoint for evaluating RL-trained policies.
XXX: Try not to modify this.
"""
from absl import app, flags
from orbax.checkpoint import PyTreeCheckpointer

import _pickle as pickle
import jax
import json
import logging
import os
import timeit

from jaxl.buffers import get_buffer
from jaxl.constants import CONST_DEFAULT, CONST_MODEL, CONST_OBS_RMS, CONST_POLICY
from jaxl.models import (
    get_model,
    get_policy,
    policy_output_dim,
)
from jaxl.envs import get_environment
from jaxl.envs.rollouts import EvaluationRollout
from jaxl.utils import set_seed, parse_dict


FLAGS = flags.FLAGS
flags.DEFINE_string("run_path", default=None, help="The saved run", required=True)
flags.DEFINE_integer("run_seed", default=None, help="Seed for the run", required=False)
flags.DEFINE_integer(
    "num_episodes", default=None, help="Number of evaluation episodes", required=True
)
flags.DEFINE_string(
    "save_path",
    default=None,
    help="Where to save the evaluation episodes",
    required=False,
)
flags.DEFINE_integer(
    "buffer_size", default=1000, help="Number of transitions to store", required=False
)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


"""
This function constructs the model and executes evaluation.
"""


def main(
    run_path: str,
    num_episodes: int,
    run_seed: int = None,
    save_path: str = None,
    buffer_size: int = 1000,
):
    """
    Orchestrates the evaluation.

    :param run_path: the saved run path
    :param num_episodes: number of evaluation episodes
    :param run_seed: the seed to initialize the random number generators
    :param save_path: the path to store the buffer
    :param buffer_size: the buffer size
    :type run_path: str
    :type num_episodes: int
    :type run_seed: int: (Default value = None)
    :type save_path: str: (Default value = None)
    :type buffer_size: int: (Default value = 1000)

    """
    tic = timeit.default_timer()
    set_seed(run_seed)
    assert os.path.isdir(run_path), f"{run_path} is not a directory"

    config_path = os.path.join(run_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config_dict["learner_config"]["buffer_config"]["buffer_size"] = buffer_size
    config_dict["learner_config"]["buffer_config"]["buffer_type"] = CONST_DEFAULT
    config = parse_dict(config_dict)

    h_state_dim = (1,)
    if hasattr(config.model_config, "h_state_dim"):
        h_state_dim = config.model_config.h_state_dim
    env = get_environment(config.learner_config.env_config)
    buffer = get_buffer(
        config.learner_config.buffer_config,
        config.learner_config.seeds.buffer_seed,
        env,
        h_state_dim,
    )
    input_dim = buffer.input_dim
    output_dim = policy_output_dim(buffer.output_dim, config.learner_config)
    model = get_model(input_dim, output_dim, config.model_config)
    policy = get_policy(model, config.learner_config)

    run_path = os.path.join(run_path, "termination_model")
    checkpointer = PyTreeCheckpointer()
    model_dict = checkpointer.restore(run_path)
    policy_params = model_dict[CONST_MODEL][CONST_POLICY]
    with open(os.path.join(run_path, "learner_dict.pkl"), "rb") as f:
        learner_dict = pickle.load(f)
        obs_rms = learner_dict[CONST_OBS_RMS]

    rollout = EvaluationRollout(env, seed=run_seed)
    rollout.rollout(policy_params, policy, obs_rms, num_episodes, buffer)
    if save_path:
        buffer.save(save_path)

    toc = timeit.default_timer()
    print(f"Evaluation Time: {toc - tic}s")


if __name__ == "__main__":
    jax.config.config_with_absl()

    def _main(argv):
        """
        Runs the evaluation.

        :param argv: the arguments provided by the user

        """
        del argv
        run_path = FLAGS.run_path
        run_seed = FLAGS.run_seed
        num_episodes = FLAGS.num_episodes
        save_path = FLAGS.save_path
        buffer_size = FLAGS.buffer_size
        main(run_path, num_episodes, run_seed, save_path, buffer_size)

    app.run(_main)
