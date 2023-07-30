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

def plot_all(task, control_mode):
    exp_id = "bc_less_data"
    bc_dir = f"/Users/chanb/research/personal/mtil_results/final_results/data/{exp_id}"
    expert_dir = f"/Users/chanb/research/personal/mtil_results/final_results/data/experts"
    experiment_name = "bc_performance-{}_{}".format(task, control_mode)
    save_path = f"./{exp_id}-results-{experiment_name}"
    curr_bc_dir = os.path.join(bc_dir, task, control_mode, "runs")
    curr_expert_dir = os.path.join(expert_dir, task, control_mode)

    assert os.path.isdir(curr_bc_dir), f"{curr_bc_dir} is not a directory"
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
        for bc_variant_name in os.listdir(curr_bc_dir):
            env_seeds.append(bc_variant_name.split(".")[2])
            bc_paths[env_seeds[-1]] = os.path.join(curr_bc_dir, bc_variant_name)

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
            num_runs = len(os.listdir(bc_dir))

            episodic_returns = {}
            for run_i, run_name in enumerate(["expert", *os.listdir(bc_dir)]):
                print(f"Processing {run_name} ({run_i + 1} / {num_runs + 1} runs)")

                if run_name == "expert":
                    agent_path = reference_agent_path
                else:
                    agent_path = os.path.join(bc_dir, run_name)

                env, policy = get_evaluation_components(
                    agent_path,
                    use_default=True,
                    ref_agent_path=reference_agent_path if run_name != "expert" else None,
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

                episodic_returns.setdefault(run_name, [])
                episodic_returns[run_name].append(
                    np.mean(agent_rollout.episodic_returns)
                )
                env.close()

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
