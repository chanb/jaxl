"""
This script is the entrypoint for evaluating policies.
XXX: Try not to modify this.
"""
from absl import app, flags

import _pickle as pickle
import jax
import logging
import os
import timeit

from jaxl.constants import CONST_DEFAULT
from jaxl.learning_utils import get_learner
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

    # TODO: Load model and evaluation
    config_path = os.path.join(run_path, "config.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    config_dict = vars(config)
    config_dict["learner_config"]["buffer_config"]["buffer_size"] = buffer_size
    config_dict["learner_config"]["buffer_config"]["buffer_type"] = CONST_DEFAULT
    config = parse_dict(config_dict)
    learner = get_learner(
        config.learner_config, config.model_config, config.optimizer_config
    )
    learner.load_checkpoint(os.path.join(run_path, "termination_model"))
    policy = learner.policy
    policy_params = learner.policy_params
    obs_rms = learner.obs_rms
    env = learner.env
    buffer = learner.buffer
    del learner

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
