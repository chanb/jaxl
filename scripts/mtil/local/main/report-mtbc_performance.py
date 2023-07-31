from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager
from pprint import pprint

import _pickle as pickle
import numpy as np
import os

from jaxl.constants import *
from jaxl.envs.rollouts import EvaluationRollout
from jaxl.utils import RunningMeanStd
from jaxl.plot_utils import get_evaluation_components


num_evaluation_episodes = 30
rollout_seed = 9999
record_video = False


def get_returns(agent_path, reference_agent_path=None):
    env, policy = get_evaluation_components(
        agent_path,
        use_default=True,
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


def plot_all(task, control_mode):
    mtbc_dir = f"/Users/chanb/research/personal/mtil_results/final_results/data/finetune_mtbc_main"
    expert_dir = f"/Users/chanb/research/personal/mtil_results/final_results/data/experts"
    experiment_name = "mtbc_performance-{}_{}".format(task, control_mode)
    save_path = f"./results-{experiment_name}"
    curr_mtbc_dir = os.path.join(mtbc_dir, task, control_mode, "runs")
    curr_expert_dir = os.path.join(expert_dir, task, control_mode)

    assert os.path.isdir(curr_mtbc_dir), f"{curr_mtbc_dir} is not a directory"
    assert os.path.isdir(curr_expert_dir), f"{curr_expert_dir} is not a directory"

    os.makedirs(save_path, exist_ok=True)
    if os.path.isfile(f"{save_path}/returns.pkl"):
        (result_per_env, env_configs, env_seeds, expert_paths, bc_paths) = pickle.load(
            open(f"{save_path}/returns.pkl", "rb")
        )
    else:
        result_per_env = {}
        env_configs = {}

        expert_paths = {}
        bc_paths = {}
        env_seeds = []
        for bc_variant_name in os.listdir(curr_mtbc_dir):
            env_seeds.append(bc_variant_name.split(".")[2])
            bc_paths[env_seeds[-1]] = os.path.join(curr_mtbc_dir, bc_variant_name)

        for expert_dir, _, filenames in os.walk(curr_expert_dir):
            env_seed = os.path.basename(os.path.dirname(expert_dir))
            if env_seed not in env_seeds:
                continue

            for filename in filenames:
                if filename != "config.json":
                    continue

                expert_paths[env_seed] = expert_dir
            
        for env_i, env_seed in enumerate(env_seeds):
            print(f"Processing {env_seed} ({env_i} / {len(env_seeds)} envs)")

            reference_agent_path = expert_paths[env_seed]
            env_config_path = os.path.join(reference_agent_path, "env_config.pkl")
            env_config = None
            if os.path.isfile(env_config_path):
                env_config = pickle.load(open(env_config_path, "rb"))

            env_configs[env_seed] = env_config

            bc_dir = bc_paths[env_seed]
            num_datasets = len(os.listdir(bc_dir))

            episodic_returns = {}
            for dataset_i, dataset_name in enumerate(["expert", *os.listdir(bc_dir)]):
                print(f"Processing {dataset_name} ({dataset_i + 1} / {num_datasets + 1} datasets)")

                if dataset_name == "expert":
                    agent_path = reference_agent_path
                    episodic_returns.setdefault(dataset_name, [])
                    episodic_returns[dataset_name].append(
                        np.mean(get_returns(agent_path))
                    )
                else:
                    episodic_returns.setdefault(dataset_name, {})
                    dataset_path = os.path.join(bc_dir, dataset_name)
                    for variant in os.listdir(dataset_path):
                        variant_path = os.path.join(dataset_path, variant)

                        for agent_path, _, filenames in os.walk(variant_path):
                            for filename in filenames:
                                if filename != "config.json":
                                    continue

                                episodic_returns[dataset_name].setdefault(variant, [])
                                episodic_returns[dataset_name][variant].append(
                                    np.mean(get_returns(agent_path, reference_agent_path))
                                )

            result_per_env[env_seed] = episodic_returns

        with open(f"{save_path}/returns.pkl", "wb") as f:
            pickle.dump((result_per_env, env_configs, env_seeds, expert_paths, bc_paths), f)

    # Report performance
    final_result = {}
    for env_seed, returns in result_per_env.items():
        means = []
        for variant in returns:
            if variant == "expert":
                continue

            means.append(returns[variant])

        final_result[env_seed] = {
            "expert": np.mean(returns["expert"]),
            "bc": (np.mean(means), np.std(means))
        }
    pprint(final_result)


for task in ["pendulum", "cheetah", "walker"]:
    for control_mode in ["discrete", "continuous"]:
        print(task, control_mode)
        plot_all(task, control_mode)