import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os

from jaxl.constants import *
from jaxl.plot_utils import set_size, pgf_with_latex


# Use the seborn style
plt.style.use("seaborn")
# But with fonts from the document body
plt.rcParams.update(pgf_with_latex)

# Using the set_size function as defined earlier
doc_width_pt = 452.9679

num_rows = 1
num_cols = 2
fig, axes = plt.subplots(
    num_rows,
    num_cols,
    figsize=set_size(doc_width_pt, 0.95, (num_rows, num_cols)),
    layout="constrained",
)

task = "cartpole"
control_mode = "continuous"


# Stiffness plot ====================================================================
save_path = f"../local/policy_robustness-example/results_policy_robustness-cartpole"
seed = 999

assert os.path.isfile(f"{save_path}/{task}_{control_mode}-returns_{seed}.pkl")
(
    default_episodic_returns,
    variant_episodic_returns,
    variant_env_seed,
    env_seeds,
    all_env_configs,
) = pickle.load(open(f"{save_path}/{task}_{control_mode}-returns_{seed}.pkl", "rb"))

all_env_seeds = [None, variant_env_seed, *env_seeds]
seeds_to_plot = np.array(all_env_seeds)

stiffness = np.array(
    [
        0.0
        if env_seed is None
        else all_env_configs[env_seed]["joint"]["hinge_1"]["stiffness"][0]
        for env_seed in all_env_seeds
    ]
)
sort_idxes = np.argsort(stiffness)
linestyles = ["--", "-."]

ax = axes[0]
for idx, returns_per_seed in enumerate(
    [default_episodic_returns, variant_episodic_returns]
):
    means = []
    stds = []
    for val in returns_per_seed.values():
        means.append(np.mean(val))
        stds.append(np.std(val))
    means = np.array(means)
    stds = np.array(stds)

    ax.axvline(stiffness[idx], linestyle="--", linewidth=0.5, color="black", alpha=0.5)
    ax.plot(
        stiffness[sort_idxes],
        means[sort_idxes],
        marker="^",
        ms=3.0,
        linewidth=0.75,
        label="stiffness @ {:.2f}".format(stiffness[idx]),
    )
    ax.fill_between(
        stiffness[sort_idxes],
        means[sort_idxes] + stds[sort_idxes],
        means[sort_idxes] - stds[sort_idxes],
        alpha=0.3,
    )
ax.legend()
ax.set_xlim(min(stiffness), max(stiffness))
ax.set_xlabel("Joint Stiffness")
# Stiffness plot ====================================================================

default_expert_mean = np.mean(default_episodic_returns[None])
default_expert_std = np.std(default_episodic_returns[None])

# BC plot ====================================================================
experiment_name = "bc_amount_data-{}_{}".format(task, control_mode)
save_path = f"../local/bc_amount_data/results-{experiment_name}"
assert os.path.isfile(f"{save_path}/returns.pkl"), f"{save_path}/returns.pkl not found"

(result_per_variant, env_configs) = pickle.load(open(f"{save_path}/returns.pkl", "rb"))

ax = axes[1]
num_sampless = sorted(
    [
        int(variant_name.split("buffer_size_")[-1])
        for variant_name in result_per_variant.keys()
    ]
)
means = []
stds = []

print(num_sampless)
for num_samples in num_sampless:
    variant_name = f"buffer_size_{num_samples}"
    iteration = sorted(list(result_per_variant[variant_name].keys()))[-1]

    means.append(np.mean(result_per_variant[variant_name][iteration]))
    stds.append(np.std(result_per_variant[variant_name][iteration]))

means = np.array(means)
stds = np.array(stds)

ax.axhline(
    default_expert_mean, label="Expert", linestyle="--", color="black", linewidth=0.5
)
ax.fill_between(
    [0, num_sampless[-1]],
    [
        default_expert_mean + default_expert_std,
        default_expert_mean + default_expert_std,
    ],
    [
        default_expert_mean - default_expert_std,
        default_expert_mean - default_expert_std,
    ],
    alpha=0.1,
    color="black",
)

ax.set_xlim(0, max(num_sampless))
ax.plot(num_sampless, means, marker="x", label="BC", linewidth=0.5)
ax.fill_between(
    num_sampless,
    means + stds,
    means - stds,
    alpha=0.3,
)

ax.set_xlabel("Amount of Training Data")
ax.legend()

# BC plot ====================================================================

fig.supylabel("Expected Return")
fig.savefig(f"./fig_1.pdf", format="pdf", bbox_inches="tight", dpi=600)
