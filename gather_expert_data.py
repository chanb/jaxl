"""
This script is the entrypoint for evaluating RL-trained policies.
XXX: Try not to modify this.
"""

from absl import app, flags
from absl.flags import FlagValues
from gymnasium import Env
from gymnasium.experimental.wrappers import RecordVideoV0
from typing import Any, Dict, Tuple, Union

import _pickle as pickle
import dill
import jax
import json
import logging
import numpy as np
import os
import timeit

from jaxl.constants import *
from jaxl.buffers import get_buffer, ReplayBuffer
from jaxl.envs import get_environment
from jaxl.envs.rollouts import EvaluationRollout
from jaxl.models import get_model, get_policy, policy_output_dim, Policy
from jaxl.utils import set_seed, get_device, parse_dict, RunningMeanStd


FLAGS = flags.FLAGS
flags.DEFINE_string("run_path", default=None, help="The saved run", required=True)
flags.DEFINE_integer("run_seed", default=None, help="Seed for the run", required=True)
flags.DEFINE_integer(
    "num_samples", default=None, help="Number of samples", required=True
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
    help="Where to save the samples",
    required=False,
)
flags.DEFINE_integer(
    "subsampling_length",
    default=1,
    help="The length of subtrajectories to gather per episode",
    required=False,
)
flags.DEFINE_integer(
    "env_seed",
    default=None,
    help="Environment seed for resetting episodes",
    required=False,
)
flags.DEFINE_boolean(
    "record_video",
    default=False,
    help="Whether or not to record video. Only enabled when save_stats=True",
    required=False,
)
flags.DEFINE_integer(
    "max_episode_length",
    default=None,
    help="Maximum episode length",
    required=False,
)
flags.DEFINE_string(
    "device",
    default=CONST_CPU,
    help="JAX device to use. To specify specific GPU device, do gpu:<device_ids>",
    required=False,
)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def load_evaluation_components(
    run_path: str,
    buffer_size: int,
) -> Tuple[
    Policy,
    Dict[str, Any],
    Union[RunningMeanStd, bool],
    Env,
    ReplayBuffer,
    int,
]:
    """

    Loads the latest checkpointed agent, the buffer, and the environment.

    :param run_path: the configuration file path to load the components from
    :param buffer_size: the buffer size
    :type run_path: str
    :type buffer_size: int
    :return: the latest checkpointed agent, the buffer, and the environment
    :rtype: Tuple[Policy, Dict[str, Any], Union[RunningMeanStd, bool], ReplayBuffer, Env, int,]

    """
    assert buffer_size > 0, f"buffer_size {buffer_size} needs to be at least 1."
    assert os.path.isdir(run_path), f"{run_path} is not a directory"

    agent_config_path = os.path.join(run_path, "config.json")
    with open(agent_config_path, "r") as f:
        agent_config_dict = json.load(f)

    agent_config_dict["learner_config"]["buffer_config"]["buffer_size"] = buffer_size
    agent_config_dict["learner_config"]["buffer_config"]["buffer_type"] = CONST_DEFAULT
    agent_config_dict["learner_config"]["env_config"]["env_kwargs"][
        "render_mode"
    ] = "rgb_array"
    agent_config = parse_dict(agent_config_dict)

    h_state_dim = (1,)
    if hasattr(agent_config.model_config, "h_state_dim"):
        h_state_dim = agent_config.model_config.h_state_dim
    env = get_environment(agent_config.learner_config.env_config)
    env_seed = agent_config.learner_config.seeds.env_seed

    buffer = get_buffer(
        agent_config.learner_config.buffer_config,
        agent_config.learner_config.seeds.buffer_seed,
        env,
        h_state_dim,
    )
    input_dim = buffer.input_dim
    output_dim = policy_output_dim(buffer.output_dim, agent_config.learner_config)
    model = get_model(
        input_dim,
        output_dim,
        getattr(agent_config.model_config, "policy", agent_config.model_config),
    )
    policy = get_policy(model, agent_config.learner_config)

    all_steps = sorted(
        os.listdir(os.path.join(run_path, "models"))
    )
    params = dill.load(
        open(
            os.path.join(run_path, "models", all_steps[-1]),
            "rb",
        )
    )
    model_dict = params[CONST_MODEL_DICT]
    policy_params = model_dict[CONST_MODEL][CONST_POLICY]
    obs_rms = False
    if CONST_OBS_RMS in params:
        obs_rms = RunningMeanStd()
        obs_rms.set_state(params[CONST_OBS_RMS])
    return policy, policy_params, obs_rms, buffer, env, env_seed


"""
This function constructs the model and executes evaluation.
"""


def main(
    config: FlagValues,
):
    """Orchestrates the evaluation."""
    tic = timeit.default_timer()
    get_device(config.device)
    set_seed(config.run_seed)
    assert os.path.isdir(config.run_path), f"{config.run_path} is not a directory"
    assert (
        config.subsampling_length > 0
    ), f"subsampling_length should be at least 1, got {config.subsampling_length}"
    assert (
        config.num_samples > 0
    ), f"num_samples should be at least 1, got {config.num_samples}"
    assert (
        config.max_episode_length is None or config.max_episode_length > 0
    ), f"max_episode_length should be at least 1, got {config.max_episode_length}"

    policy, policy_params, obs_rms, buffer, env, env_seed = load_evaluation_components(
        config.run_path, config.num_samples
    )

    if config.env_seed is not None:
        env_seed = config.env_seed

    if config.save_stats and config.record_video:
        env = RecordVideoV0(
            env, f"{os.path.dirname(config.save_stats)}/videos", disable_logger=True
        )

    rollout = EvaluationRollout(env, seed=env_seed)
    rollout.rollout_with_subsampling(
        policy_params,
        policy,
        obs_rms,
        buffer,
        config.num_samples,
        config.subsampling_length,
        config.max_episode_length,
    )
    if config.save_buffer:
        print("Saving buffer with {} transitions".format(len(buffer)))
        buffer.save(config.save_buffer, end_with_done=False)

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
