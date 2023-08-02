"""
This script performs hyperparameter search on multiple environment variations.

Then, to generate the data, run the generated script run_all-*.sh ${run_seed}
"""

from absl import app, flags
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager

import _pickle as pickle
import jax
import numpy as np
import os


from jaxl.constants import *
from jaxl.envs.rollouts import EvaluationRollout
from jaxl.utils import RunningMeanStd
from jaxl.plot_utils import get_evaluation_components


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_seed",
    default=None,
    help="Environment seed",
    required=True,
)
flags.DEFINE_integer(
    "rollout_seed",
    default=None,
    help="Rollout seed",
    required=True,
)
flags.DEFINE_integer(
    "num_evaluation_episodes",
    default=10,
    help="Number of evaluation episodes",
    required=False,
)
flags.DEFINE_string(
    "variant_name",
    default=None,
    help="The variant name",
    required=True,
)
flags.DEFINE_string(
    "runs_path",
    default=None,
    help="The path storing the runs",
    required=True,
)
flags.DEFINE_string(
    "save_dir",
    default=None,
    help="The directory to save the result",
    required=True,
)
flags.DEFINE_string(
    "reference_agent_path",
    default=None,
    help="The agent that contians all the environment information",
    required=False,
)


def get_returns(
    agent_path,
    env_seed,
    rollout_seed,
    num_evaluation_episodes,
    reference_agent_path=None,
):
    env, policy = get_evaluation_components(
        agent_path,
        env_seed,
        ref_agent_path=reference_agent_path,
    )

    checkpoint_manager = CheckpointManager(
        os.path.join(agent_path, "models"),
        PyTreeCheckpointer(),
    )

    checkpoint_step = checkpoint_manager.latest_step()
    params = checkpoint_manager.restore(checkpoint_step)
    model_dict = params[CONST_MODEL_DICT]
    agent_policy_params = model_dict[CONST_MODEL][CONST_POLICY]
    agent_obs_rms = False
    if CONST_OBS_RMS in params:
        agent_obs_rms = RunningMeanStd()
        agent_obs_rms.set_state(params[CONST_OBS_RMS])

    agent_rollout = EvaluationRollout(env, seed=rollout_seed)
    agent_rollout.rollout(
        agent_policy_params,
        policy,
        agent_obs_rms,
        num_evaluation_episodes,
        None,
        use_tqdm=False,
    )
    env.close()
    return agent_rollout.episodic_returns


def main(config):
    assert (
        config.num_evaluation_episodes > 0
    ), "num_evaluation_episodes needs to be at least 1"
    assert config.reference_agent_path is None or os.path.isdir(
        config.reference_agent_path
    ), f"{config.reference_agent_path} is not a directory"
    assert os.path.isdir(config.runs_path), f"{config.runs_path} is not a directory"
    assert os.path.isdir(config.save_dir), f"{config.save_dir} is not a directory"

    assert (config.variant_name == "expert") == (
        config.reference_agent_path == config.runs_path
    )

    env_seed_int = int(config.env_seed.split("env_seed_")[-1])
    if config.variant_name == "expert":
        result = np.mean(
            get_returns(
                config.runs_path,
                env_seed_int,
                config.rollout_seed,
                config.num_evaluation_episodes,
            )
        )
    else:
        result = []
        for agent_path, _, filenames in os.walk(config.runs_path):
            for filename in filenames:
                if filename != "config.json":
                    continue

                result.append(
                    np.mean(
                        get_returns(
                            agent_path,
                            env_seed_int,
                            config.rollout_seed,
                            config.num_evaluation_episodes,
                            config.reference_agent_path,
                        )
                    )
                )
    save_path = os.path.join(config.save_dir, config.env_seed, config.variant_name)
    os.makedirs(os.path.join(config.save_dir, config.env_seed), exist_ok=True)
    with open("{}.pkl".format(save_path), "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    jax.config.config_with_absl()

    def _main(argv):
        del argv
        main(FLAGS)

    app.run(_main)
