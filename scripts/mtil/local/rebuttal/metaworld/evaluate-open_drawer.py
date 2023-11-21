"""
This script is the entrypoint for evaluating RL-trained policies.
XXX: Try not to modify this.
"""
from absl import app, flags
from absl.flags import FlagValues
from gymnasium.experimental.wrappers import RecordVideoV0
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager

import _pickle as pickle
import jax
import json
import logging
import metaworld
import numpy as np
import os
import timeit

import jaxl.envs.metaworld.policies as metaworld_policies
from jaxl.constants import *
from jaxl.envs.metaworld.rollouts import MetaWorldRollout
from jaxl.models import get_model, get_policy, policy_output_dim
from jaxl.utils import set_seed, parse_dict

CONST_CPU = "cpu"
CONST_GPU = "gpu"

TASK_NAME = "drawer-open-v2"

FLAGS = flags.FLAGS
flags.DEFINE_string("run_path", default=None, help="The saved run", required=True)
flags.DEFINE_boolean("expert", default=False, help="Whether or not to use expert policy", required=False)
flags.DEFINE_integer("img_res", default=64, help="Image resolution", required=False)
flags.DEFINE_integer(
    "env_seed", default=None, help="The environment seed", required=True
)
flags.DEFINE_integer(
    "scrambling_step",
    default=0,
    help="The number of random initialization steps",
    required=False,
)
flags.DEFINE_integer("run_seed", default=None, help="Seed for the run", required=False)
flags.DEFINE_integer(
    "num_episodes", default=None, help="Number of episodes", required=True
)
flags.DEFINE_string(
    "save_stats",
    default=None,
    help="Where to save the episodic statistics",
    required=False,
)
flags.DEFINE_boolean(
    "record_video",
    default=False,
    help="Whether or not to record video. Only enabled when save_stats=True",
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


"""
This function constructs the model and executes evaluation.
"""

def get_env(env_seed):
    ml1 = metaworld.ML1(TASK_NAME)  # Construct the benchmark, sampling tasks

    env = ml1.train_classes[TASK_NAME](
        render_mode="rgb_array"
    )  # Create an environment with task `pick_place`
    task = ml1.train_tasks[np.random.RandomState(env_seed).choice(len(ml1.train_tasks))]
    env.set_task(task)  # Set task
    env.camera_name = "corner3"
    return env


def main(
    config: FlagValues,
):
    """Orchestrates the evaluation."""
    tic = timeit.default_timer()
    set_seed(config.run_seed)

    (device_name, *device_ids) = config.device.split(":")
    if device_name == CONST_CPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device_name == CONST_GPU:
        assert (
            len(device_ids) > 0
        ), f"at least one device_id is needed, got {device_ids}"
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids[0]
    else:
        raise ValueError(f"{device_name} is not a supported device.")


    assert (
        config.num_episodes > 0
    ), f"num_episodes should be at least 1, got {config.num_episodes}"

    if config.env_seed is not None:
        env_seed = config.env_seed

    env = get_env(env_seed)
    act_dim = (*env.action_space.shape, 1)

    with open(os.path.join(config.run_path, "config.json"), "r") as f:
        agent_config_dict = json.load(f)
        agent_config_dict["learner_config"]["policy_distribution"] = "deterministic"
        agent_config = parse_dict(agent_config_dict)

    # input_dim = env.observation_space.shape
    input_dim = (3, config.img_res, config.img_res)
    output_dim = policy_output_dim(act_dim, agent_config.learner_config)
    if config.expert:
        policy = metaworld_policies.get_policy(TASK_NAME)
        agent_policy_params = None
    else:
        model = get_model(
            input_dim,
            output_dim,
            getattr(agent_config.model_config, "policy", agent_config.model_config),
        )
        policy = get_policy(model, agent_config.learner_config)

        if config.save_stats and config.record_video:
            env = RecordVideoV0(
                env, f"{os.path.dirname(config.save_stats)}/videos", disable_logger=True
            )

        checkpoint_manager = CheckpointManager(
            os.path.join(config.run_path, "models"),
            PyTreeCheckpointer(),
        )

        checkpoint_step = checkpoint_manager.latest_step()
        params = checkpoint_manager.restore(checkpoint_step)
        model_dict = params[CONST_MODEL_DICT]
        agent_policy_params = model_dict[CONST_MODEL][CONST_POLICY]

    rollout = MetaWorldRollout(
        env, seed=env_seed, num_scrambling_steps=config.scrambling_step
    )
    rollout.rollout(
        agent_policy_params,
        policy,
        False,
        config.num_episodes,
        None,
        use_image_for_inference=True,
        get_image=True,
        width=config.img_res,
        height=config.img_res,
    )

    os.makedirs(os.path.dirname(config.save_stats), exist_ok=True)
    if config.save_stats:
        print("Saving episodic statistics")
        with open(config.save_stats, "wb") as f:
            pickle.dump(
                {
                    CONST_EPISODIC_RETURNS: rollout.episodic_returns,
                    CONST_EPISODE_LENGTHS: rollout.episode_lengths,
                    CONST_RUN_PATH: config.run_path,
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
