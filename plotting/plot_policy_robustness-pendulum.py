from gymnasium.experimental.wrappers import RecordVideoV0
from itertools import product
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager

import _pickle as pickle
import json
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

from jaxl.constants import *
from jaxl.envs.rollouts import EvaluationRollout
from jaxl.utils import RunningMeanStd, flatten_dict
from plot_utils import set_size, pgf_with_latex, get_evaluation_components


# Use the seborn style
plt.style.use("seaborn")
# But with fonts from the document body
plt.rcParams.update(pgf_with_latex)

# Using the set_size function as defined earlier
doc_width_pt = 452.9679


expert_dir = "/Users/chanb/research/personal/mtil_results/data_without_pretrain/experts"
tasks = ["pendulum"]
control_modes = ["continuous"]

save_path = f"./results_policy_robustness-pendulum"
os.makedirs(save_path, exist_ok=True)

seed = 1000

rollout_seed = 1000
env_seed_range = 1000
num_envs_to_test = 9
pretrain_dir = (
    "/Users/chanb/research/personal/mtil_results/data_from_pretrain/pretrained_ppo/"
)

num_evaluation_episodes = 50
record_video = False

assert os.path.isdir(expert_dir), f"{expert_dir} is not a directory"
assert num_envs_to_test > 0, f"num_envs_to_test needs to be at least 1"

all_res = {}

for task, control_mode in product(tasks, control_modes):
    print(task, control_mode)
    rng = np.random.RandomState(seed)
    experiment_dir = os.path.join(expert_dir, task, control_mode, "runs/0")
    variants = np.array(os.listdir(experiment_dir))
    num_variants = len(variants)

    env_seeds = rng.randint(0, env_seed_range, size=num_envs_to_test)
    while len(np.unique(env_seeds)) != num_envs_to_test:
        env_seeds = rng.randint(0, env_seed_range, size=num_envs_to_test)

    default_agent_path = os.path.join(pretrain_dir, "{}_{}".format(task, control_mode))
    if os.path.isfile(f"{save_path}/{task}_{control_mode}-returns_{seed}.pkl"):
        (
            episodic_returns_per_seed,
            env_seeds,
            all_env_configs,
        ) = pickle.load(
            open(f"{save_path}/{task}_{control_mode}-returns_{seed}.pkl", "rb")
        )
    else:
        all_env_configs = {}
        episodic_returns_per_seed = {}
        all_env_seeds = [None, *env_seeds]

        env_config_path = os.path.join(default_agent_path, "env_config.pkl")
        env_config = None
        if os.path.isfile(env_config_path):
            env_config = pickle.load(open(env_config_path, "rb"))

        checkpoint_manager = CheckpointManager(
            os.path.join(default_agent_path, "models"),
            PyTreeCheckpointer(),
        )

        for env_seed in all_env_seeds:
            env, policy = (
                get_evaluation_components(default_agent_path, None, True)
                if env_seed is None
                else get_evaluation_components(default_agent_path, env_seed, False)
            )
            if record_video:
                env = RecordVideoV0(
                    env,
                    f"{save_path}/videos/{task}-{control_mode}/env_seed_{env_seed}",
                    disable_logger=True,
                )
            all_env_configs[env_seed] = env.get_config()["modified_attributes"]
            print(
                env_seed,
                env.get_config()["modified_attributes"],
            )
            params = checkpoint_manager.restore(checkpoint_manager.latest_step())
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

            episodic_returns_per_seed.setdefault(env_seed, [])
            episodic_returns_per_seed[env_seed].append(agent_rollout.episodic_returns)
            env.close()

        with open(f"{save_path}/{task}_{control_mode}-returns_{seed}.pkl", "wb") as f:
            pickle.dump((episodic_returns_per_seed, env_seeds, all_env_configs), f)

    all_res[(task, control_mode)] = (
        episodic_returns_per_seed,
        env_seeds,
        all_env_configs,
    )

# Plot main return
num_rows = len(tasks)
num_cols = len(control_modes)
fig, axes = plt.subplots(
    num_rows,
    num_cols,
    figsize=set_size(doc_width_pt, 0.95, (num_rows, num_cols)),
    layout="constrained",
)

for row_i, task in enumerate(tasks):
    for col_i, control_mode in enumerate(control_modes):
        (
            episodic_returns_per_seed,
            env_seeds,
            all_env_configs,
        ) = all_res[(task, control_mode)]

        all_env_seeds = [None, *env_seeds]
        seeds_to_plot = np.array(all_env_seeds)

        torques = np.array(
            [
                2.0 if env_seed is None else all_env_configs[env_seed]["max_torque"]
                for env_seed in all_env_seeds
            ]
        )
        sort_idxes = np.argsort(torques)

        if num_cols == num_rows == 1:
            ax = axes
        elif num_cols == 1 or num_rows == 1:
            ax = axes[row_i * num_cols + col_i]
        else:
            ax = axes[row_i, col_i]

        means = []
        stds = []
        for val in episodic_returns_per_seed.values():
            means.append(np.mean(val))
            stds.append(np.std(val))
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(
            torques[sort_idxes],
            means[sort_idxes],
            marker="^",
            ms=3.0,
            linewidth=0.75,
        )
        ax.axvline(2.0, label="trained", linewidth=0.5, linestyle="--", c="black")
        ax.fill_between(
            torques[sort_idxes],
            means[sort_idxes] + stds[sort_idxes],
            means[sort_idxes] - stds[sort_idxes],
            alpha=0.3,
        )
        ax.legend()

        if row_i == len(tasks) - 1:
            ax.set_xlabel(control_mode)

fig.supylabel("Expected Return")
fig.supxlabel("Maximum Torque")
fig.savefig(
    f"{save_path}/policy_robustness.pdf", format="pdf", bbox_inches="tight", dpi=600
)
