from gymnasium.experimental.wrappers import RecordVideoV0
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager
from typing import Iterable

import _pickle as pickle
import json
import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

from jaxl.constants import *
from jaxl.utils import RunningMeanStd, flatten_dict
from plot_utils import set_size, pgf_with_latex


# Use the seborn style
plt.style.use("seaborn")
# But with fonts from the document body
plt.rcParams.update(pgf_with_latex)

# Using the set_size function as defined earlier
doc_width_pt = 452.9679
experiment_name = "single_hyperparameter_robustness"
experiment_dir = (
    "/Users/chanb/research/personal/mtil_results/data/single_hyperparam_robustness/cheetah"
)
hyperparameter_path = (
    "/Users/chanb/research/personal/mtil_results/data/single_hyperparam_robustness/hyperparameters-single_hyperparam_robustness-cheetah_discrete.pkl"
)
save_path = f"./results-{experiment_name}"
os.makedirs(save_path, exist_ok=True)

plot_reference = False
reference_config_path = "/Users/chanb/research/personal/mtil_results/data/search_expert/cheetah/discrete/scripts/0/variant-71.json"
top_k = 5
smoothing = 20
num_evaluation_episodes = 5
env_seed = 9999
record_video = False

assert os.path.isdir(experiment_dir), f"{experiment_dir} is not a directory"
assert os.path.isfile(hyperparameter_path), f"{hyperparameter_path} is not a file"

if plot_reference:
    assert os.path.isfile(reference_config_path), f"{reference_config_path} is not a file"

    with open(reference_config_path, "r") as f:
        reference_config = json.load(f)
        del reference_config["learner_config"]["seeds"]
        del reference_config["learner_config"]["env_config"]

with open(hyperparameter_path, "rb") as f:
    (hyperparam_keys, hyperparamss) = pickle.load(f)
    num_envs = len(hyperparamss[-2])
    num_models = len(hyperparamss[-1])
    num_variants = len(list(itertools.product(*hyperparamss)))
    num_hyperparamss = num_variants // (num_envs * num_models)

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
    for variant_i, hyperparams in enumerate(itertools.product(*hyperparamss)):
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
                    match_hyperparams = match_hyperparams and (curr_config[key] == reference_config[key])
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

        checkpoint = checkpoint_manager.restore(
            checkpoint_manager.latest_step()
        )

        result_per_variant.setdefault(env_seed, {})
        result_per_variant[env_seed].setdefault(hyperparam_i, [])
        result_per_variant[env_seed][hyperparam_i].append(checkpoint[CONST_AUX][
            CONST_EPISODIC_RETURNS
        ][:-1])
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
    layout="constrained"
)

aucs_per_seed = {}
top_k_per_seed = []
for ax_i, (env_seed, result_per_hyperparam) in enumerate(result_per_variant.items()):
    aucs_per_seed[env_seed] = []
    for hyperparam_i in range(num_hyperparamss):
        aucs_per_seed[env_seed].append(
            np.sum(np.mean(result_per_hyperparam[hyperparam_i], axis=0))
        )
    aucs_per_seed[env_seed] = np.array(aucs_per_seed[env_seed])
    top_k_idxes = np.argsort(aucs_per_seed[env_seed])[-top_k:]
    # top_k_idxes = [22, 28] # Visually both 22 and 28 are robust.

    if num_cols == num_rows == 1:
        ax = axes
    elif num_cols == 1 or num_rows == 1:
        ax = axes[ax_i]
    else:
        ax = axes[ax_i // num_cols, ax_i % num_cols]
    for hyperparam_i, idx in enumerate([*top_k_idxes, match_hyperparams_i]):
        if idx is None:
            continue

        smoothed_returns = []
        
        for returns in result_per_hyperparam[idx]:
            cumsum_returns = np.cumsum(returns)
            last_t_episodic_returns = cumsum_returns - np.concatenate(
                (np.zeros(smoothing), cumsum_returns[:-smoothing])
            )
            smoothed_returns.append(
                last_t_episodic_returns / np.concatenate(
                    (
                        np.arange(1, smoothing + 1),
                        np.ones(len(returns) - smoothing) * smoothing,
                    )
                )
            )

        smoothed_returns_mean = np.mean(smoothed_returns, axis=0)
        smoothed_returns_std = np.std(smoothed_returns, axis=0)
        num_episodes = np.arange(len(smoothed_returns_mean))

        ax.plot(
            num_episodes,
            smoothed_returns_mean,
            label=idx
            if hyperparam_i != len(top_k_idxes)
            else "{} - ref.".format(idx),
            linewidth=0.5,
            alpha=0.7,
            linestyle="-" if hyperparam_i != len(top_k_idxes) else ":",
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
    )

num_blanks = num_cols - num_envs % num_cols
if num_blanks > 0:
    for ax_i in range(1, num_blanks + 1):
        axes[-1, -ax_i].axis('off')

fig.supylabel("Expected Return")
fig.supxlabel("Training Episode")
fig.savefig(f"{save_path}/returns.pdf", format="pdf", bbox_inches="tight", dpi=600)

# Plot return based on environmental parameter
max_return_means = []
max_return_stds = []
modified_attributes = {}
for variant_name in env_configs:
    env_config = env_configs[variant_name]
    returns = result_per_variant[variant_name]

    max_return_mean = -np.inf
    max_return_std = 0.0
    for val in returns.values():
        mean = np.mean(val)
        std = np.std(val)
        if max_return_mean < mean:
            max_return_mean = mean
            max_return_std = std

    max_return_means.append(max_return_mean)
    max_return_stds.append(max_return_std)

    for attr_name, attr_val in flatten_dict(env_config):
        if isinstance(attr_val, Iterable):
            for val_i in range(len(attr_val)):
                modified_attributes.setdefault(f"{attr_name}.{val_i}", [])
                modified_attributes[f"{attr_name}.{val_i}"].append(attr_val[val_i])
        else:
            modified_attributes.setdefault(attr_name, [])
            modified_attributes[attr_name].append(attr_val)

max_return_means = np.array(max_return_means)
max_return_stds = np.array(max_return_stds)

for attr_name, attr_vals in modified_attributes.items():
    modified_attributes[attr_name] = np.array(attr_vals)

# fraction = 0.9
# num_rows = 1
# num_cols = 1
# if len(modified_attributes) % 5 == 0:
#     num_cols = 5
#     num_rows = len(modified_attributes) // 5
# elif len(modified_attributes) % 4 == 0:
#     num_cols = 4
#     num_rows = len(modified_attributes) // 4
# elif len(modified_attributes) % 3 == 0:
#     num_cols = 3
#     num_rows = len(modified_attributes) // 3
# elif len(modified_attributes) % 2 == 0:
#     num_cols = 2
#     num_rows = len(modified_attributes) // 2
# else:
#     num_cols = len(modified_attributes)
#     num_rows = 1

# fig, axes = plt.subplots(
#     num_rows, num_cols, figsize=set_size(doc_width_pt, fraction, (num_rows, num_cols))
# )
# fig.supylabel("Expected Return")

# for attr_i, (attr_name, attr_vals) in enumerate(modified_attributes.items()):
#     sort_idxes = np.argsort(attr_vals)

#     if num_rows == num_cols == 1:
#         ax = axes
#     elif num_rows == 1 or num_cols == 1:
#         ax = axes[attr_i]
#     else:
#         ax = axes[attr_i // num_cols, attr_i % num_cols]

#     x_vals = attr_vals[sort_idxes]
#     means = max_return_means[sort_idxes]
#     stds = max_return_stds[sort_idxes]
#     ax.plot(x_vals, means, marker="x")
#     ax.fill_between(
#         x_vals,
#         means + stds,
#         means - stds,
#         alpha=0.3,
#     )

#     ax.set_xlabel(f"{attr_name}")

# fig.tight_layout()
# fig.savefig(
#     f"{save_path}/returns-variants.pdf", format="pdf", bbox_inches="tight", dpi=600
# )
