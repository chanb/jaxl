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
from jaxl.plot_utils import set_size, pgf_with_latex, get_evaluation_components


# Use the seborn style
plt.style.use("seaborn")
# But with fonts from the document body
plt.rcParams.update(pgf_with_latex)

# Using the set_size function as defined earlier
doc_width_pt = 452.9679


expert_dir = "/Users/chanb/research/personal/mtil_results/data_from_pretrain/experts"

# tasks = ["pendulum_no_act_cost", "cheetah", "walker"]
# task_renames = ["pendulum", "cheetah", "walker"]

tasks = ["cheetah", "cheetah_hard"]
task_renames = ["small range", "large range"]
control_modes = ["discrete", "continuous"]


save_path = f"./results_policy_robustness"
os.makedirs(save_path, exist_ok=True)

seed = 1000

rollout_seed = 1000
env_seed_range = 1000
num_envs_to_test = 5
num_agents_to_test = 5

num_evaluation_episodes = 30
record_video = True

assert os.path.isdir(expert_dir), f"{expert_dir} is not a directory"
assert num_envs_to_test > 0, f"num_envs_to_test needs to be at least 1"
assert num_agents_to_test > 0, f"num_agents_to_test needs to be at least 1"


all_res = {}

for task, control_mode in product(tasks, control_modes):
    print(task, control_mode)
    rng = np.random.RandomState(seed)
    experiment_dir = os.path.join(expert_dir, task, control_mode, "runs/0")
    variants = np.array(os.listdir(experiment_dir))
    num_variants = len(variants)
    variants_sample_idxes = rng.randint(0, num_variants, size=num_agents_to_test)
    while len(np.unique(variants_sample_idxes)) != num_agents_to_test:
        variants_sample_idxes = rng.randint(0, num_variants, size=num_agents_to_test)

    env_seeds = rng.randint(0, env_seed_range, size=num_envs_to_test)
    while len(np.unique(env_seeds)) != num_envs_to_test:
        env_seeds = rng.randint(0, env_seed_range, size=num_envs_to_test)

    dirs_to_load = variants[variants_sample_idxes]
    if os.path.isfile(f"{save_path}/{task}_{control_mode}-returns_{seed}.pkl"):
        (result_per_variant, env_configs, default_env_seeds) = pickle.load(
            open(f"{save_path}/{task}_{control_mode}-returns_{seed}.pkl", "rb")
        )
    else:
        result_per_variant = {}
        env_configs = {}
        default_env_seeds = {}
        agent_paths = {}
        for variant_i, variant_name in enumerate(dirs_to_load):
            variant_path = os.path.join(experiment_dir, variant_name)
            for agent_path, _, filenames in os.walk(variant_path):
                for filename in filenames:
                    if filename != "config.json":
                        continue
                    agent_paths[variant_name] = agent_path

                    agent_config_path = os.path.join(agent_path, "config.json")
                    with open(agent_config_path, "r") as f:
                        agent_config_dict = json.load(f)
                        default_env_seed = agent_config_dict["learner_config"][
                            "env_config"
                        ]["env_kwargs"]["seed"]
                        default_env_seeds[variant_name] = default_env_seed

        all_env_seeds = [*default_env_seeds.values(), *env_seeds]

        for variant_i, variant_name in enumerate(agent_paths):
            print(f"Processing variant {variant_i + 1} / {num_agents_to_test}")
            agent_path = agent_paths[variant_name]
            env_config_path = os.path.join(agent_path, "env_config.pkl")
            env_config = None
            if os.path.isfile(env_config_path):
                env_config = pickle.load(open(env_config_path, "rb"))

            checkpoint_manager = CheckpointManager(
                os.path.join(agent_path, "models"),
                PyTreeCheckpointer(),
            )

            episodic_returns_per_variant = {}
            for env_seed in all_env_seeds:
                env, policy = get_evaluation_components(agent_path, env_seed)
                if record_video:
                    env = RecordVideoV0(
                        env,
                        f"{save_path}/videos/{task}-{control_mode}-variant_{variant_name}-default_seed_{default_env_seeds[variant_name]}/env_seed_{env_seed}",
                        disable_logger=True,
                    )
                print(env_seed, env.get_config()["modified_attributes"])
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

                episodic_returns_per_variant.setdefault(env_seed, [])
                episodic_returns_per_variant[env_seed].append(
                    agent_rollout.episodic_returns
                )
                env.close()
            result_per_variant[variant_name] = episodic_returns_per_variant
            env_configs[variant_name] = env_config["modified_attributes"]

        with open(f"{save_path}/{task}_{control_mode}-returns_{seed}.pkl", "wb") as f:
            pickle.dump((result_per_variant, env_configs, default_env_seeds), f)

    all_res[(task, control_mode)] = (
        result_per_variant,
        env_configs,
        default_env_seeds,
        env_seeds,
    )

# Plot main return
num_rows = len(control_modes)
num_cols = len(tasks)
fig, axes = plt.subplots(
    num_rows,
    num_cols,
    figsize=set_size(doc_width_pt, 0.95, (num_rows, num_cols)),
    layout="constrained",
)

for row_i, control_mode in enumerate(control_modes):
    for col_i, task in enumerate(tasks):
        (result_per_variant, env_configs, default_env_seeds, env_seeds) = all_res[
            (task, control_mode)
        ]

        all_env_seeds = [*default_env_seeds.values(), *env_seeds]
        seeds_to_plot = np.array(all_env_seeds)

        if num_cols == num_rows == 1:
            ax = axes
        elif num_cols == 1 or num_rows == 1:
            ax = axes[row_i * num_cols + col_i]
        else:
            ax = axes[row_i, col_i]

        for variant_name, returns in result_per_variant.items():
            means = []
            stds = []
            for val in returns.values():
                means.append(np.mean(val))
                stds.append(np.std(val))
            means = np.array(means)
            stds = np.array(stds)

            variant_idx = np.where(seeds_to_plot == default_env_seeds[variant_name])[0]
            ax.plot(
                np.arange(len(seeds_to_plot)),
                means,
                label="env-{}".format(np.arange(len(means))[variant_idx[0]])
                if row_i + col_i == 0
                else "",
                markevery=variant_idx,
                marker="*",
                linewidth=1.0,
            )
            ax.fill_between(
                np.arange(len(seeds_to_plot)),
                means + stds,
                means - stds,
                alpha=0.3,
            )

        if col_i == 0:
            ax.set_ylabel(control_mode)
        if row_i + 1 == num_rows:
            ax.set_xlabel(task_renames[col_i])


fig.legend(
    bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
    loc="lower center",
    ncols=num_agents_to_test,
    borderaxespad=0.0,
    frameon=True,
    fontsize="5",
)

fig.supylabel("Expected Return")
fig.supxlabel("Environment Variant")
fig.savefig(
    f"{save_path}/policy_robustness.pdf", format="pdf", bbox_inches="tight", dpi=600
)
