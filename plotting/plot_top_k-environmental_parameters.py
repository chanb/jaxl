from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager
from pprint import pprint

import _pickle as pickle
import json
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os

from jaxl.constants import *
from plot_utils import set_size, pgf_with_latex


# Use the seborn style
plt.style.use("seaborn")
# But with fonts from the document body
plt.rcParams.update(pgf_with_latex)

plot_reference = False
reference_config_path = None
top_k = 8
smoothing = 20
num_evaluation_episodes = 5
env_seed = 9999
record_video = False

# Using the set_size function as defined earlier
doc_width_pt = 452.9679

# Experiment to filter from 32 configurations to 8 configurations
# experiment_name = "cheetah_sweep"
# experiment_dir = (
#     "/Users/chanb/research/personal/mtil_results/data/cheetah_sweep/cheetah/discrete"
# )
# hyperparameter_path = "/Users/chanb/research/personal/mtil_results/data/cheetah_sweep/hyperparameters-cheetah_sweep-cheetah_discrete.pkl"

# Experiment to choose continuous pendulum
# experiment_name = "hyperparam_sweep-pendulum_cont"
# experiment_dir = "/Users/chanb/research/personal/mtil_results/data/hyperparam_sweep/pendulum/continuous"
# hyperparameter_path = (
#     "/Users/chanb/research/personal/mtil_results/data/hyperparam_sweep/hyperparameters-hyperparam_sweep-pendulum_continuous.pkl"
# )

# Experiment to choose discrete pendulum
# experiment_name = "hyperparam_sweep-pendulum_disc"
# experiment_dir = "/Users/chanb/research/personal/mtil_results/data/hyperparam_sweep/pendulum/discrete"
# hyperparameter_path = (
#     "/Users/chanb/research/personal/mtil_results/data/hyperparam_sweep/hyperparameters-hyperparam_sweep-pendulum_discrete.pkl"
# )

# Experiment to choose continuous cheetah
# experiment_name = "hyperparam_sweep-cheetah_cont"
# experiment_dir = "/Users/chanb/research/personal/mtil_results/data/hyperparam_sweep/cheetah/continuous"
# hyperparameter_path = (
#     "/Users/chanb/research/personal/mtil_results/data/hyperparam_sweep/hyperparameters-hyperparam_sweep-cheetah_continuous.pkl"
# )

# Experiment to choose discrete cheetah
# experiment_name = "hyperparam_sweep-cheetah_disc"
# experiment_dir = "/Users/chanb/research/personal/mtil_results/data/hyperparam_sweep/cheetah/discrete"
# hyperparameter_path = (
#     "/Users/chanb/research/personal/mtil_results/data/hyperparam_sweep/hyperparameters-hyperparam_sweep-cheetah_discrete.pkl"
# )

# Experiment to choose continuous walker
# experiment_name = "hyperparam_sweep-walker_cont"
# experiment_dir = "/Users/chanb/research/personal/mtil_results/data/hyperparam_sweep/walker/continuous"
# hyperparameter_path = (
#     "/Users/chanb/research/personal/mtil_results/data/hyperparam_sweep/hyperparameters-hyperparam_sweep-walker_continuous.pkl"
# )

# Experiment to choose discrete walker
# experiment_name = "hyperparam_sweep-walker_disc"
# experiment_dir = "/Users/chanb/research/personal/mtil_results/data/hyperparam_sweep/walker/discrete"
# hyperparameter_path = (
#     "/Users/chanb/research/personal/mtil_results/data/hyperparam_sweep/hyperparameters-hyperparam_sweep-walker_discrete.pkl"
# )


save_path = f"./results-{experiment_name}"
os.makedirs(save_path, exist_ok=True)

assert os.path.isdir(experiment_dir), f"{experiment_dir} is not a directory"
assert os.path.isfile(hyperparameter_path), f"{hyperparameter_path} is not a file"

if plot_reference:
    assert os.path.isfile(
        reference_config_path
    ), f"{reference_config_path} is not a file"

    with open(reference_config_path, "r") as f:
        reference_config = json.load(f)
        del reference_config["learner_config"]["seeds"]
        del reference_config["learner_config"]["env_config"]

with open(hyperparameter_path, "rb") as f:
    (hyperparam_keys, hyperparamss) = pickle.load(f)
    num_envs = len(hyperparamss[-2])
    num_models = len(hyperparamss[-1])
    hyperparams_comb = list(itertools.product(*hyperparamss))
    num_variants = len(hyperparams_comb)
    num_hyperparamss = num_variants // (num_envs * num_models)
    if not top_k:
        top_k = num_hyperparamss

if os.path.isfile(f"{save_path}/returns.pkl"):
    (result_per_variant, env_configs, match_hyperparams_i) = pickle.load(
        open(f"{save_path}/returns.pkl", "rb")
    )
else:
    result_per_variant = {}
    env_configs = {}
    agent_paths = {}
    for agent_path, _, filenames in os.walk(experiment_dir):
        variant_name = os.path.basename(os.path.dirname(agent_path))
        if not variant_name.startswith("variant-"):
            continue

        for filename in filenames:
            if filename == "config.json":
                break
        agent_paths[int(variant_name.split("-")[1])] = agent_path

    match_hyperparams_i = None
    for variant_i, hyperparams in enumerate(hyperparams_comb):
        if (variant_i + 1) % 10 == 0:
            print(f"Processed {variant_i + 1}/{num_variants} variants")

        model_seed = hyperparams[-1]
        env_seed = hyperparams[-2]
        hyperparam_i = variant_i // (num_envs * num_models)

        variant_name = f"hyperparam_{hyperparam_i}-env_seed_{env_seed}"

        agent_path = agent_paths[variant_i]

        if plot_reference and match_hyperparams_i is None:
            match_hyperparams = True
            curr_config_path = os.path.join(agent_path, "config.json")
            with open(curr_config_path, "r") as f:
                curr_config = json.load(f)
                del curr_config["learner_config"]["seeds"]
                del curr_config["learner_config"]["env_config"]

                for key in ("model_config", "learner_config", "optimizer_config"):
                    match_hyperparams = match_hyperparams and (
                        curr_config[key] == reference_config[key]
                    )
            if match_hyperparams:
                match_hyperparams_i = hyperparam_i

        env_config_path = os.path.join(agent_path, "env_config.pkl")
        env_config = None
        if os.path.isfile(env_config_path):
            env_config = pickle.load(open(env_config_path, "rb"))

        checkpoint_manager = CheckpointManager(
            os.path.join(agent_path, "models"),
            PyTreeCheckpointer(),
        )

        checkpoint = checkpoint_manager.restore(checkpoint_manager.latest_step())

        result_per_variant.setdefault(env_seed, {})
        result_per_variant[env_seed].setdefault(hyperparam_i, [])
        result_per_variant[env_seed][hyperparam_i].append(
            checkpoint[CONST_AUX][CONST_EPISODIC_RETURNS][:-1]
        )
        env_configs[env_seed] = env_config["modified_attributes"]

    with open(f"{save_path}/returns.pkl", "wb") as f:
        pickle.dump((result_per_variant, env_configs, match_hyperparams_i), f)

# Plot main return
num_cols = 3
num_rows = math.ceil(num_envs / num_cols)
fig, axes = plt.subplots(
    num_rows,
    num_cols,
    figsize=set_size(doc_width_pt, 0.95, (num_rows, num_cols)),
    layout="constrained",
)

aucs_per_seed = {}
top_k_per_seed = []
agg_auc_list = {}
all_top_ks = np.zeros(num_hyperparamss)
for ax_i, (env_seed, result_per_hyperparam) in enumerate(result_per_variant.items()):
    aucs_per_seed[env_seed] = []
    for hyperparam_i in range(num_hyperparamss):
        aucs_per_seed[env_seed].append(
            np.sum(np.mean(result_per_hyperparam[hyperparam_i], axis=0))
        )
    aucs_per_seed[env_seed] = np.array(aucs_per_seed[env_seed])
    top_k_idxes = np.argsort(aucs_per_seed[env_seed])[-top_k:]
    all_top_ks[np.argsort(aucs_per_seed[env_seed])] += np.arange(num_hyperparamss)

    if num_cols == num_rows == 1:
        ax = axes
    elif num_cols == 1 or num_rows == 1:
        ax = axes[ax_i]
    else:
        ax = axes[ax_i // num_cols, ax_i % num_cols]
    for iter_i, hyperparam_i in enumerate([*top_k_idxes, match_hyperparams_i]):
        if hyperparam_i is None:
            continue

        agg_auc_list.setdefault(hyperparam_i, [])
        smoothed_returns = []

        for returns in result_per_hyperparam[hyperparam_i]:
            cumsum_returns = np.cumsum(returns)
            last_t_episodic_returns = cumsum_returns - np.concatenate(
                (np.zeros(smoothing), cumsum_returns[:-smoothing])
            )
            smoothed_returns.append(
                last_t_episodic_returns
                / np.concatenate(
                    (
                        np.arange(1, smoothing + 1),
                        np.ones(len(returns) - smoothing) * smoothing,
                    )
                )
            )

        agg_auc_list[hyperparam_i].append(np.mean(result_per_hyperparam[hyperparam_i], axis=1))
        smoothed_returns_mean = np.mean(smoothed_returns, axis=0)
        smoothed_returns_std = np.std(smoothed_returns, axis=0)
        num_episodes = np.arange(len(smoothed_returns_mean))

        label = hyperparam_i
        if iter_i == 0:
            label = "{} - worst".format(hyperparam_i)
        elif iter_i == len(top_k_idxes) - 1:
            label = "{} - best".format(hyperparam_i)
        ax.plot(
            num_episodes,
            smoothed_returns_mean,
            label=label,
            linewidth=0.5,
            alpha=0.7,
            linestyle="-" if iter_i != len(top_k_idxes) else ":",
        )
        ax.fill_between(
            num_episodes,
            smoothed_returns_mean + smoothed_returns_std,
            smoothed_returns_mean - smoothed_returns_std,
            alpha=0.3,
        )
    ax.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncols=math.ceil(top_k / 2),
        mode="expand",
        borderaxespad=0.0,
        frameon=True,
        fontsize="4",
    )

num_blanks = num_cols - num_envs % num_cols
if num_cols > num_blanks > 0:
    for ax_i in range(1, num_blanks + 1):
        axes[-1, -ax_i].axis("off")

fig.supylabel("Expected Return")
fig.supxlabel("Training Episode")
fig.savefig(f"{save_path}/returns.pdf", format="pdf", bbox_inches="tight", dpi=1000)

total_aucs_means = {k: np.mean(np.array(v)) for k, v in agg_auc_list.items()}
total_aucs_stds = {k: np.std(np.array(v)) for k, v in agg_auc_list.items()}
hyperparam_list = list(total_aucs_means.keys())
hyperparam_total_auc_means = list(total_aucs_means.values())
hyperparam_total_auc_stds = list(total_aucs_stds.values())
total_aucs_sort_idxes = np.argsort(hyperparam_total_auc_means)

top_hyperparam = np.array(hyperparam_list)[total_aucs_sort_idxes][-top_k:]
hyperparams_comb = list(itertools.product(*hyperparamss[:-2]))
for idx, top_param_idx in enumerate(top_hyperparam):
    print("=" * 50)
    print(f"RANK {top_k - idx} - PARAM: {top_param_idx}")
    for key, val in zip(hyperparam_keys, hyperparams_comb[top_param_idx]):
        print("{}: {}".format(key, val))
print(np.argsort(all_top_ks)[-top_k:])
print(np.stack((hyperparam_list, hyperparam_total_auc_means, hyperparam_total_auc_stds)).T[np.argsort(hyperparam_list)])
