"""
This script is the entrypoint for evaluating RL-trained policies.
XXX: Try not to modify this.
"""
from absl import app, flags
from absl.flags import FlagValues
from gymnasium.experimental.wrappers import RecordVideoV0

import _pickle as pickle
import jax
import logging
import metaworld
import numpy as np
import os
import timeit

from jaxl.buffers.ram_buffers import MemoryEfficientNumPyBuffer
from jaxl.constants import (
    CONST_EPISODE_LENGTHS,
    CONST_EPISODIC_RETURNS,
    CONST_RUN_PATH,
    CONST_BUFFER_PATH,
)
from jaxl.envs.metaworld.policies import get_policy
from jaxl.envs.metaworld.rollouts import MetaWorldRollout
from jaxl.utils import set_seed


FLAGS = flags.FLAGS
flags.DEFINE_integer("env_seed", default=None, help="The environment seed", required=True)
flags.DEFINE_integer("scrambling_step", default=0, help="The number of random initialization steps", required=False)
flags.DEFINE_integer("run_seed", default=None, help="Seed for the run", required=False)
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

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


"""
This function constructs the model and executes evaluation.
"""


TASK_NAME = "drawer-open-v2"
HEIGHT = 64
WIDTH = 64
def get_env(env_seed):
    ml1 = metaworld.ML1(TASK_NAME) # Construct the benchmark, sampling tasks

    env = ml1.train_classes[TASK_NAME](render_mode="rgb_array")  # Create an environment with task `pick_place`
    task = ml1.train_tasks[np.random.RandomState(env_seed).choice(len(ml1.train_tasks))]
    env.set_task(task)  # Set task
    env.camera_name = "corner2"
    return env


def main(
    config: FlagValues,
):
    """Orchestrates the evaluation."""
    tic = timeit.default_timer()
    set_seed(config.run_seed)
    assert (
        config.subsampling_length > 0
    ), f"subsampling_length should be at least 1, got {config.subsampling_length}"
    assert (
        config.num_samples > 0
    ), f"num_samples should be at least 1, got {config.num_samples}"
    assert (
        config.max_episode_length is None or config.max_episode_length > 0
    ), f"max_episode_length should be at least 1, got {config.max_episode_length}"

    if config.env_seed is not None:
        env_seed = config.env_seed

    policy = get_policy(TASK_NAME)
    env = get_env(env_seed)
    buffer = MemoryEfficientNumPyBuffer(
        buffer_size=config.num_samples,
        obs_dim=(3, HEIGHT, WIDTH),
        h_state_dim=(1,),
        act_dim=(*env.action_space.shape, 1),
        rew_dim=(1,),
    )

    if config.save_stats and config.record_video:
        env = RecordVideoV0(
            env, f"{os.path.dirname(config.save_stats)}/videos", disable_logger=True
        )

    rollout = MetaWorldRollout(env, seed=env_seed, num_scrambling_steps=config.scrambling_step)
    rollout.rollout_with_subsampling(
        None,
        policy,
        False,
        buffer,
        config.num_samples,
        config.subsampling_length,
        config.max_episode_length,
        get_image=True,
        width=WIDTH,
        height=HEIGHT,
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
