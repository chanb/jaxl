import _pickle as pickle
import gzip
import math
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
control_mode = "discrete"
num_samples_to_gather = [100, 500, 1000, 1500, 2000, 2500, 3000, 5000, 7500]
save_path = f"../local/bc_amount_data"
expert_buffer = "../local/bc_amount_data/logs/demonstrations/expert_buffer-default-pendulum_discrete-num_samples_100000-subsampling_200.gzip"

configs.append(
    (
        task,
        control_mode,
        num_samples_to_gather,
        save_path,
        expert_buffer,
    )
)

task = "cheetah"
control_mode = "discrete"
num_samples_to_gather = [100, 500, 1000, 2500, 5000, 7500, 10000]
save_path = f"../local/bc_amount_data/agg_results"
expert_buffer = "../local/bc_amount_data/logs/demonstrations/expert_buffer-default-cheetah_discrete-num_samples_100000-subsampling_1000.gzip"

configs.append(
    (
        task,
        control_mode,
        num_samples_to_gather,
        save_path,
        expert_buffer,
    )
)

task = "walker"
control_mode = "discrete"
num_samples_to_gather = [100, 500, 1000, 2500, 5000]
save_path = f"../local/bc_amount_data/agg_results"
expert_buffer = "../local/bc_amount_data/logs/demonstrations/expert_buffer-default-walker_discrete-num_samples_100000-subsampling_1000.gzip"

configs.append(
    (
        task,
        control_mode,
        num_samples_to_gather,
        save_path,
        expert_buffer,
    )
)

task = "pendulum"
control_mode = "continuous"
num_samples_to_gather = [100, 500, 1000, 1500, 2000, 2500]
save_path = f"../local/bc_amount_data/agg_results"
expert_buffer = "../local/bc_amount_data/logs/demonstrations/expert_buffer-default-pendulum_continuous-num_samples_100000-subsampling_200.gzip"

configs.append(
    (
        task,
        control_mode,
        num_samples_to_gather,
        save_path,
        expert_buffer,
    )
)

task = "cheetah"
control_mode = "continuous"
num_samples_to_gather = [100, 500, 1000, 2500, 5000, 7500, 10000]
save_path = f"../local/bc_amount_data/agg_results"
expert_buffer = "../local/bc_amount_data/logs/demonstrations/expert_buffer-default-cheetah_continuous-num_samples_100000-subsampling_1000.gzip"

configs.append(
    (
        task,
        control_mode,
        num_samples_to_gather,
        save_path,
        expert_buffer,
    )
)

task = "walker"
control_mode = "continuous"
num_samples_to_gather = [100, 500, 1000, 2500, 5000, 7500, 10000]
save_path = f"../local/bc_amount_data/agg_results"
expert_buffer = "../local/bc_amount_data/logs/demonstrations/expert_buffer-default-walker_continuous-num_samples_100000-subsampling_1000.gzip"

configs.append(
    (
        task,
        control_mode,
        num_samples_to_gather,
        save_path,
        expert_buffer,
    )
)


num_rows = math.ceil(len(configs) / 3)
num_cols = 3
fig, axes = plt.subplots(
    num_rows,
    num_cols,
    figsize=set_size(doc_width_pt, 0.95, (num_rows, num_cols)),
    layout="constrained",
)


def get_result(
    task,
    control_mode,
    num_samples_to_gather,
    save_path,
):
    per_sample_means = []
    per_sample_stds = []

    curr_res_path = os.path.join(
        save_path,
        "results-bc_amount_data-{}_{}".format(task, control_mode),
        "returns.pkl",
    )
    assert os.path.isfile(curr_res_path)
    (result_per_variant, _) = pickle.load(open(curr_res_path, "rb"))

    for num_samples in num_samples_to_gather:
        variant_name = "buffer_size_{}".format(num_samples)
        print(result_per_variant.keys())
        rets = result_per_variant[variant_name]

        last_iteration = max(rets.keys())
        per_sample_means.append(np.mean(rets[last_iteration]))
        per_sample_stds.append(np.std(rets[last_iteration]))

    per_sample_means = np.array(per_sample_means)
    per_sample_stds = np.array(per_sample_stds)
    return per_sample_means, per_sample_stds


for idx, config in enumerate(configs):
    (
        task,
        control_mode,
        num_samples_to_gather,
        save_path,
        expert_buffer,
    ) = config

    buffer_dict = pickle.load(gzip.open(expert_buffer, "rb"))
    num_episodes = int(np.sum(buffer_dict["dones"]))
    buffer_returns = np.sum(buffer_dict["rewards"].reshape((num_episodes, -1)), axis=-1)
    expert_mean = np.mean(buffer_returns)
    expert_std = np.std(buffer_returns)

    means, stds = get_result(
        task,
        control_mode,
        num_samples_to_gather,
        save_path,
    )

    ax = axes[idx // num_cols, idx % num_cols]
    ax.axhline(
        expert_mean,
        color="black",
        alpha=0.5,
        label="$\pi^*$" if idx == 0 else "",
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
    ax.plot(
        num_samples_to_gather,
        means,
        marker="^",
        ms=3.0,
        linewidth=0.75,
        label="BC" if idx == 0 else "",
    )
    ax.fill_between(
        num_samples_to_gather,
        means + stds,
        means - stds,
        alpha=0.3,
    )
    ax.set_xlim(num_samples_to_gather[0], num_samples_to_gather[-1])
    if idx == 0:
        ax.legend()

    row_i = idx // num_cols
    col_i = idx % num_cols
    if col_i == 0:
        ax.set_ylabel(f"{control_mode}")
    if row_i + 1 == num_rows:
        ax.set_xlabel(f"{task}")

fig.supylabel("Expected Return")
fig.supxlabel("Amount of Transitions")
fig.savefig(f"./data_ablation.pdf", format="pdf", bbox_inches="tight", dpi=600)
