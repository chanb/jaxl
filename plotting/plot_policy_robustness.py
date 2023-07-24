from gymnasium.experimental.wrappers import RecordVideoV0
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager
from typing import Iterable

import _pickle as pickle
import json
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

# Pendulum continuous expert
experiment_name = "pendulum_cont"
experiment_dir = "/Users/chanb/research/personal/mtil_results/data/experts/pendulum/continuous/runs/0/"

# Pendulum discrete expert
experiment_name = "pendulum_disc"
experiment_dir = (
    "/Users/chanb/research/personal/mtil_results/data/experts/pendulum/discrete/runs/0/"
)

# Cheetah continuous expert
# experiment_name = "cheetah_cont"
# experiment_dir = (
#     "/Users/chanb/research/personal/mtil_results/data/experts/cheetah/continuous/runs/0/"
# )

# Cheetah discrete expert
# experiment_name = "cheetah_disc"
# experiment_dir = (
#     "/Users/chanb/research/personal/mtil_results/data/experts/cheetah/discrete/runs/0/"
# )

# Walker continuous expert
# experiment_name = "walker_cont"
# experiment_dir = (
#     "/Users/chanb/research/personal/mtil_results/data/experts/walker/continuous/runs/0/"
# )

# Walker discrete expert
# experiment_name = "walker_disc"
# experiment_dir = (
#     "/Users/chanb/research/personal/mtil_results/data/experts/walker/discrete/runs/0/"
# )

save_path = f"./results_policy_robustness-{experiment_name}"
os.makedirs(save_path, exist_ok=True)

seed = 0
rng = np.random.RandomState(seed)

rollout_seed = 9999
env_seed_range = 1000
num_envs_to_test = 1
num_agents_to_test = 5

num_evaluation_episodes = 10
record_video = False

assert os.path.isdir(experiment_dir), f"{experiment_dir} is not a directory"
assert num_envs_to_test > 0, f"num_envs_to_test needs to be at least 1"
assert num_agents_to_test > 0, f"num_agents_to_test needs to be at least 1"


variants = np.array(os.listdir(experiment_dir))
num_variants = len(variants)
variants_sample_idxes = rng.randint(0, num_variants, size=num_agents_to_test)
while len(np.unique(variants_sample_idxes)) != num_agents_to_test:
    variants_sample_idxes = rng.randint(0, num_variants, size=num_agents_to_test)

env_seeds = rng.randint(0, env_seed_range, size=num_envs_to_test)
while len(np.unique(env_seeds)) != num_envs_to_test:
    env_seeds = rng.randint(0, env_seed_range, size=num_envs_to_test)

dirs_to_load = variants[variants_sample_idxes]
if os.path.isfile(f"{save_path}/returns_{seed}.pkl"):
    (result_per_variant, env_configs, default_env_seeds) = pickle.load(
        open(f"{save_path}/returns_{seed}.pkl", "rb")
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
                        default_env_seed = agent_config_dict["learner_config"]["env_config"]["env_kwargs"][
                            "env_seed"
                        ]
                        default_env_seeds[variant_name] = default_env_seed
    
    all_env_seeds = [*env_seeds, *default_env_seeds.values()]

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
                    f"{save_path}/videos/variant_{variant_name}/env_seed_{env_seed}",
                    disable_logger=True,
                )
            params = checkpoint_manager.restore(
                checkpoint_manager.latest_step()
            )
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
                np.mean(agent_rollout.episodic_returns)
            )
        result_per_variant[variant_name] = episodic_returns_per_variant
        env_configs[variant_name] = env_config["modified_attributes"]

    with open(f"{save_path}/returns_{seed}.pkl", "wb") as f:
        pickle.dump((result_per_variant, env_configs, default_env_seeds), f)

# Plot main return
fig, ax = plt.subplots(1, 1, figsize=set_size(doc_width_pt, 0.95, (1, 1)))

seeds_to_plot = np.array(all_env_seeds)
sort_idxes = np.argsort(seeds_to_plot)
seeds_to_plot = seeds_to_plot[sort_idxes]

for variant_name, returns in result_per_variant.items():
    means = []
    stds = []
    for val in returns.values():
        means.append(np.mean(val))
        stds.append(np.std(val))
    means = np.array(means)
    stds = np.array(stds)

    ax.plot(
        seeds_to_plot,
        means[sort_idxes],
        label=variant_name,
        markevery=np.where(seeds_to_plot == default_env_seeds[variant_name])[0],
        marker="*",
        linewidth=1.0,
    )
    ax.fill_between(
        seeds_to_plot,
        means[sort_idxes] + stds[sort_idxes],
        means[sort_idxes] - stds[sort_idxes],
        alpha=0.3,
    )
    # ax.scatter([default_env_seeds[variant_name]], [means[0]], marker="*")

ax.set_ylabel("Expected Return")
ax.set_xlabel("Environment Variant")
ax.legend()

fig.tight_layout()
fig.savefig(
    f"{save_path}/returns_{seed}.pdf", format="pdf", bbox_inches="tight", dpi=600
)
