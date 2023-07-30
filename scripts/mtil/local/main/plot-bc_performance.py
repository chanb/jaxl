from gymnasium.experimental.wrappers import RecordVideoV0
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager
from typing import Iterable

import _pickle as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

from jaxl.constants import *
from jaxl.envs.rollouts import EvaluationRollout
from jaxl.utils import RunningMeanStd, flatten_dict
from jaxl.plot_utils import set_size, pgf_with_latex, get_evaluation_components


# Use the seborn style
plt.style.use("seaborn")
# But with fonts from the document body
plt.rcParams.update(pgf_with_latex)

# Using the set_size function as defined earlier
doc_width_pt = 452.9679
num_evaluation_episodes = 30
rollout_seed = 9999
record_video = False

def plot_all(task, control_mode):
    bc_dir = f"/Users/chanb/research/personal/mtil_results/final_results/data/bc_main"
    expert_dir = f"/Users/chanb/research/personal/mtil_results/final_results/data/experts"
    experiment_name = "bc_performance-{}_{}".format(task, control_mode)
    save_path = f"./results-{experiment_name}"
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
            print(f"Processing {env_seed} ({env_i} / {len(env_seeds)})")

            reference_agent_path = expert_paths[env_seed]
            env_config_path = os.path.join(reference_agent_path, "env_config.pkl")
            env_config = None
            if os.path.isfile(env_config_path):
                env_config = pickle.load(open(env_config_path, "rb"))

            env_configs[env_seed] = env_config

            bc_dir = bc_paths[env_seed]
            num_runs = len(os.listdir(bc_dir))

            episodic_returns = {}
            for run_i, run_name in enumerate(["expert", os.listdir(bc_dir)]):
                print(f"Processing {run_i + 1} / {num_runs + 1} runs")

                if run_name == "expert":
                    agent_path = reference_agent_path
                else:
                    agent_path = os.path.join(bc_dir, run_name)

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

                episodic_returns.setdefault(run_name, [])
                episodic_returns[run_name].append(
                    np.mean(agent_rollout.episodic_returns)
                )
                env.close()

            result_per_env[env_seed] = episodic_returns

        with open(f"{save_path}/returns.pkl", "wb") as f:
            pickle.dump((result_per_env, env_configs, env_seeds, expert_paths, bc_paths), f)

    # Plot main return
    # fig, axes = plt.subplots(
    #     1, 2, figsize=set_size(doc_width_pt, 0.95, (1, 2)), layout="constrained"
    # )

    # ax = axes[0]
    # for variant_name, returns in result_per_variant.items():
    #     iteration = list(returns.keys())
    #     means = []
    #     stds = []
    #     for val in returns.values():
    #         means.append(np.mean(val))
    #         stds.append(np.std(val))
    #     means = np.array(means)
    #     stds = np.array(stds)

    #     sort_idxes = np.argsort(iteration)
    #     iteration = np.array(iteration)
    #     ax.plot(
    #         iteration[sort_idxes], means[sort_idxes], marker="x", label=variant_name
    #     )
    #     ax.fill_between(
    #         iteration[sort_idxes],
    #         means[sort_idxes] + stds[sort_idxes],
    #         means[sort_idxes] - stds[sort_idxes],
    #         alpha=0.3,
    #     )

    # ax.set_ylabel("Expected Return")
    # ax.legend()

    # fig.supxlabel("Iterations")
    # fig.savefig(
    #     f"{save_path}/returns-{task}_{control_mode}.pdf",
    #     format="pdf",
    #     bbox_inches="tight",
    #     dpi=600,
    # )


# for task in ["pendulum", "cheetah", "walker"]:
#     for control_mode in ["discrete", "continuous"]:

for task in ["cheetah"]:
    for control_mode in ["discrete"]:
        plot_all(task, control_mode)
