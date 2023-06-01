"""
This script is the entrypoint for evaluating RL-trained policies.
XXX: Try not to modify this.
"""
from absl import app, flags
from absl.flags import FlagValues
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

    agent_config_path = os.path.join(config.run_path, "config.json")
    with open(agent_config_path, "r") as f:
        agent_config_dict = json.load(f)

    agent_config_dict["learner_config"]["buffer_config"][
        "buffer_size"
    ] = config.buffer_size
    agent_config_dict["learner_config"]["buffer_config"]["buffer_type"] = CONST_DEFAULT
    agent_config = parse_dict(agent_config_dict)

    h_state_dim = (1,)
    if hasattr(agent_config.model_config, "h_state_dim"):
        h_state_dim = agent_config.model_config.h_state_dim
    env = get_environment(agent_config.learner_config.env_config)
    buffer = get_buffer(
        agent_config.learner_config.buffer_config,
        agent_config.learner_config.seeds.buffer_seed,
        env,
        h_state_dim,
    )
    input_dim = buffer.input_dim
    output_dim = policy_output_dim(buffer.output_dim, agent_config.learner_config)
    model = get_model(input_dim, output_dim, agent_config.model_config)
    policy = get_policy(model, agent_config.learner_config)

    run_path = os.path.join(config.run_path, "termination_model")
    checkpointer = PyTreeCheckpointer()
    model_dict = checkpointer.restore(run_path)
    policy_params = model_dict[CONST_MODEL][CONST_POLICY]
    with open(os.path.join(run_path, "learner_dict.pkl"), "rb") as f:
        learner_dict = pickle.load(f)
        obs_rms = learner_dict[CONST_OBS_RMS]

    env_seed = config.env_seed
    if env_seed is None:
        env_seed = agent_config.learner_config.seeds.env_seed
    rollout = EvaluationRollout(env, seed=env_seed)
    rollout.rollout(policy_params, policy, obs_rms, config.num_episodes, buffer)
    if config.save_path:
        print("Saving buffer with {} transitions".format(len(buffer)))
        buffer.save(config.save_path)

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
        main(FLAGS)

    app.run(_main)
