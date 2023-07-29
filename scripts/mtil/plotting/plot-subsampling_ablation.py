import _pickle as pickle
import gzip
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

configs = []

task = "pendulum"
control_mode = "continuous"
subsamplings = [1, 20, 200]
num_samples_to_gather = [125, 250, 500, 1000, 1500, 2000, 2500]
save_path = f"../local/bc_subsampling"
expert_buffer = "../local/bc_subsampling/logs/demonstrations/expert_buffer-default-pendulum_continuous-num_samples_10000-subsampling_200.gzip"

configs.append(
    (
        task,
        control_mode,
        subsamplings,
        num_samples_to_gather,
        save_path,
        expert_buffer,
    )
)

task = "cheetah"
control_mode = "discrete"
subsamplings = [1, 20, 1000]
num_samples_to_gather = [1000, 2500, 5000, 10000]
save_path = f"../local/bc_subsampling"
expert_buffer = "../local/bc_subsampling/logs/demonstrations/expert_buffer-default-cheetah_discrete-num_samples_10000-subsampling_1000.gzip"

configs.append(
    (
        task,
        control_mode,
        subsamplings,
        num_samples_to_gather,
        save_path,
        expert_buffer,
    )
)

task = "cheetah"
control_mode = "continuous"
subsamplings = [1, 20, 1000]
num_samples_to_gather = [1000, 2500, 5000, 10000]
save_path = f"../local/bc_subsampling"
expert_buffer = "../local/bc_subsampling/logs/demonstrations/expert_buffer-default-cheetah_continuous-num_samples_10000-subsampling_1000.gzip"

configs.append(
    (
        task,
        control_mode,
        subsamplings,
        num_samples_to_gather,
        save_path,
        expert_buffer,
    )
)


num_rows = 1
num_cols = len(configs)
fig, axes = plt.subplots(
    num_rows,
    num_cols,
    figsize=set_size(doc_width_pt, 0.95, (num_rows, num_cols)),
    layout="constrained",
)


def get_result(
    task,
    control_mode,
    subsamplings,
    num_samples_to_gather,
    save_path,
):
    per_sample_means = []
    per_sample_stds = []

    for num_samples in num_samples_to_gather:
        curr_res_path = os.path.join(
            save_path,
            "results-bc_subsampling-size_{}-{}_{}".format(
                num_samples, task, control_mode
            ),
            "returns.pkl",
        )
        assert os.path.isfile(curr_res_path)
        (result_per_variant, _) = pickle.load(open(curr_res_path, "rb"))

        per_sample_means.append([])
        per_sample_stds.append([])
        for subsampling in subsamplings:
            variant_name = "subsampling_{}".format(subsampling)
            rets = result_per_variant[variant_name]

            last_iteration = max(rets.keys())
            per_sample_means[-1].append(np.mean(rets[last_iteration]))
            per_sample_stds[-1].append(np.std(rets[last_iteration]))

    per_sample_means = np.array(per_sample_means).T
    per_sample_stds = np.array(per_sample_stds).T
    return per_sample_means, per_sample_stds

for idx, config in enumerate(configs):
    (
        task,
        control_mode,
        subsamplings,
        num_samples_to_gather,
        save_path,
        expert_buffer,
    ) = config

    buffer_dict = pickle.load(gzip.open(expert_buffer, "rb"))
    num_episodes = int(np.sum(buffer_dict["dones"]))
    buffer_returns = np.sum(buffer_dict["rewards"].reshape((num_episodes, -1)), axis=-1)
    expert_mean = np.mean(buffer_returns)
    expert_std = np.std(buffer_returns)

    per_sample_means, per_sample_stds = get_result(
        task,
        control_mode,
        subsamplings,
        num_samples_to_gather,
        save_path,
    )

    ax = axes[idx]
    ax.axhline(
        expert_mean,
        color="black",
        alpha=0.5,
        label="$\pi^*$",
        linewidth=0.5,
        linestyle="--",
    )
    ax.fill_between(
        [num_samples_to_gather[0], num_samples_to_gather[-1]],
        [expert_mean + expert_std],
        [expert_mean - expert_std],
        alpha=0.1,
        color="black",
    )
    for idx, (means, stds) in enumerate(zip(per_sample_means, per_sample_stds)):
        ax.plot(
            num_samples_to_gather,
            means,
            marker="^",
            ms=3.0,
            linewidth=0.75,
            label="{}".format(subsamplings[idx]),
        )
        ax.fill_between(
            num_samples_to_gather,
            means + stds,
            means - stds,
            alpha=0.3,
        )
    ax.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncols=4,
        mode="expand",
        borderaxespad=0.0,
        frameon=True,
        fontsize="5",
    )
    ax.set_xlim(num_samples_to_gather[0], num_samples_to_gather[-1])
    ax.set_xlabel("{} {}".format(task, control_mode))

    fig.supylabel("Expected Return")
    fig.supxlabel("Amount of Transitions")
    fig.savefig(
        f"./subsampling_ablation.pdf", format="pdf", bbox_inches="tight", dpi=600
    )
