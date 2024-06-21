import _pickle as pickle
import argparse
import chex
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from tqdm import tqdm

from jaxl.plot_utils import set_size, pgf_with_latex


# Use the seborn style
sns.set_style("darkgrid")
sns.set_palette("colorblind")

# But with fonts from the document body
# plt.rcParams.update(pgf_with_latex)

# Using the set_size function as defined earlier
doc_width_pt = 1000.0

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_dir",
    type=str,
    default="./results",
    help="The directory that stores all experiments",
)
parser.add_argument("--ablation_name", type=str, default="", help="Ablation name")
parser.add_argument("--variant", type=str, default="", help="The evaluation variant")
parser.add_argument(
    "--logs_dir", type=str, default=None, help="The directory that logs the results"
)
args = parser.parse_args()

results_dir = args.results_dir
ablation_name = args.ablation_name
variant = args.variant
logs_dir = args.logs_dir

# FILTERS
include_prefix = None
include_suffix = None
exclude_prefix = None
exclude_suffix = None
include_evals = None
map_eval_to_title = {}

num_cols = 2
interp_gap_size = 1000

agg_result_path = os.path.join(
    results_dir, ablation_name, "agg_data/accuracies{}.pkl".format(variant)
)
plot_path = os.path.join(results_dir, ablation_name, "plots")

os.makedirs(plot_path, exist_ok=True)

agg_result = pickle.load(open(agg_result_path, "rb"))


def get_baseline_accuracy(zipf_exp, num_classes):
    zipf_weights = np.array([1 / j**zipf_exp for j in range(num_classes, 0, -1)])
    zipf_weights /= np.sum(zipf_weights)
    return [np.sum(zipf_weights**2) * 100]


max_num_evals = 0
max_checkpoint_steps = 0
max_context_len = 0
for exp_name, exp_runs in agg_result.items():
    if include_prefix and not exp_name.startswith(include_prefix):
        continue
    if include_suffix and not exp_name.endswith(include_suffix):
        continue
    if exclude_prefix and exp_name.startswith(exclude_prefix):
        continue
    if exclude_suffix and exp_name.endswith(exclude_suffix):
        continue
    for run_name, exp_run in exp_runs.items():
        curr_checkpoint_steps = np.max(exp_run["checkpoint_steps"])
        curr_num_evals = len([eval_name for eval_name in exp_run["accuracies"]])
        curr_context_len = max(
            len(context_lens[-1]) - 1 for context_lens in exp_run["auxes"].values()
        )

        if curr_checkpoint_steps > max_checkpoint_steps:
            max_checkpoint_steps = curr_checkpoint_steps

        if curr_num_evals > max_num_evals:
            max_num_evals = curr_num_evals

        if curr_context_len > max_context_len:
            max_context_len = curr_context_len


def process_exp_runs(exp_runs: dict, x_range: chex.Array):
    interpolated_results = dict()
    for run_i, (run_name, exp_run) in enumerate(exp_runs.items()):
        curr_checkpoint_steps = exp_run["checkpoint_steps"]
        curr_accuracies = exp_run["accuracies"]

        for eval_name, accuracies in curr_accuracies.items():
            interpolated_results.setdefault(
                eval_name, np.zeros((len(exp_runs), len(x_range)))
            )
            interpolated_results[eval_name][run_i] = np.interp(
                x_range, curr_checkpoint_steps, accuracies
            )
    return interpolated_results


def context_len_exp_runs(exp_runs: dict, x_range: chex.Array):
    interpolated_results = dict()
    ratio_results = dict()
    for run_i, (run_name, exp_run) in enumerate(exp_runs.items()):
        for eval_name, eval_result in exp_run["auxes"].items():
            curr_context_lens = [
                len_i for len_i in range(max_context_len) if len_i in eval_result[-1]
            ]
            curr_accuracies = [
                eval_result[-1][len_i]["accuracy"] for len_i in curr_context_lens
            ]
            curr_ratios = [
                eval_result[-1][len_i]["query_class_in_context_ratio"]
                for len_i in curr_context_lens
            ]
            if len(curr_context_lens) == 0:
                continue

            interpolated_results.setdefault(
                eval_name, np.zeros((len(exp_runs), len(x_range)))
            )
            interpolated_results[eval_name][run_i] = np.interp(
                x_range, curr_context_lens, curr_accuracies
            )

            ratio_results.setdefault(eval_name, np.zeros((len(exp_runs), len(x_range))))
            ratio_results[eval_name][run_i] = np.interp(
                x_range, curr_context_lens, curr_ratios
            )
    return interpolated_results, ratio_results


def plot_context_length_plot():
    save_path = os.path.join(
        plot_path, "{}{}-context_len.pdf".format(ablation_name, variant)
    )
    num_rows = math.ceil(max_num_evals / num_cols)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=set_size(doc_width_pt, 0.95, (num_rows, num_cols)),
        layout="constrained",
    )

    map_eval_to_ax = {}
    max_count = -1

    if include_evals:
        for eval_name in include_evals:
            max_count += 1
            map_eval_to_ax[eval_name] = (
                axes[max_count // num_cols, max_count % num_cols],
                max_count,
            )
            map_eval_to_ax[eval_name][0].set_title(
                map_eval_to_title.get(eval_name, eval_name)
            )
            map_eval_to_ax[eval_name][0].set_ylim(-1.0, 101.0)

    x_range = np.arange(1, max_context_len + 1, 1)
    for exp_name, exp_runs in tqdm(agg_result.items()):
        if include_prefix and not exp_name.startswith(include_prefix):
            continue
        if include_suffix and not exp_name.endswith(include_suffix):
            continue
        if exclude_prefix and exp_name.startswith(exclude_prefix):
            continue
        if exclude_suffix and exp_name.endswith(exclude_suffix):
            continue

        processed_results, ratio_results = context_len_exp_runs(exp_runs, x_range)

        for eval_name, processed_result in processed_results.items():
            if include_evals and eval_name not in include_evals:
                continue
            update_ax = False
            if eval_name not in map_eval_to_ax:
                max_count += 1
                map_eval_to_ax[eval_name] = (
                    axes[max_count // num_cols, max_count % num_cols],
                    max_count,
                )
                update_ax = True

            y_means = np.nanmean(processed_result, axis=0)
            y_stderrs = np.nanstd(processed_result, axis=0) / np.sqrt(
                len(processed_result)
            )

            (ax, ax_i) = map_eval_to_ax[eval_name]
            ax.plot(x_range, y_means, label=exp_name if ax_i == 3 else "")
            ax.fill_between(
                x_range, (y_means - y_stderrs), (y_means + y_stderrs), alpha=0.1
            )

            if update_ax:
                ax.set_title(map_eval_to_title.get(eval_name, eval_name))
                ax.set_ylim(-1.0, 101.0)

    remaining_idx = num_cols * num_rows - (max_count + 1)
    if remaining_idx > 0:
        for ii in range(remaining_idx):
            ax_i = ii + max_count + 1
            ax = axes[ax_i // num_cols, ax_i % num_cols]
            ax.axis("off")

    fig.supxlabel("Context Length")
    fig.supylabel("Accuracy")
    fig.legend(
        bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
        loc="lower center",
        ncols=len(agg_result),
        borderaxespad=0.0,
        frameon=True,
        fontsize="8",
    )

    fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=600)


def plot_main_plot():
    save_path = os.path.join(plot_path, "{}{}.pdf".format(ablation_name, variant))
    num_rows = math.ceil(max_num_evals / num_cols)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=set_size(doc_width_pt, 0.95, (num_rows, num_cols)),
        layout="constrained",
    )

    map_eval_to_ax = {}
    max_count = -1

    if include_evals:
        for eval_name in include_evals:
            max_count += 1
            map_eval_to_ax[eval_name] = (
                axes[max_count // num_cols, max_count % num_cols],
                max_count,
            )
            map_eval_to_ax[eval_name][0].set_title(
                map_eval_to_title.get(eval_name, eval_name)
            )
            map_eval_to_ax[eval_name][0].set_ylim(-1.0, 101.0)

    x_range = np.arange(0, max_checkpoint_steps + 1, interp_gap_size)
    for exp_name, exp_runs in tqdm(agg_result.items()):
        if include_prefix and not exp_name.startswith(include_prefix):
            continue
        if include_suffix and not exp_name.endswith(include_suffix):
            continue
        if exclude_prefix and exp_name.startswith(exclude_prefix):
            continue
        if exclude_suffix and exp_name.endswith(exclude_suffix):
            continue
        processed_results = process_exp_runs(exp_runs, x_range)

        baseline_acc = None
        if logs_dir is not None:
            config_path = os.path.join(
                logs_dir, list(exp_runs.keys())[0], "config.json"
            )
            config_dict = json.load(open(config_path, "r"))
            zipf_exp = config_dict["learner_config"]["dataset_config"][
                "dataset_kwargs"
            ]["zipf_exp"]
            num_abstract_classes = config_dict["learner_config"]["dataset_config"][
                "dataset_kwargs"
            ]["num_abstract_classes"]
            baseline_acc = get_baseline_accuracy(zipf_exp, num_abstract_classes)

        for eval_name, processed_result in processed_results.items():
            if include_evals and eval_name not in include_evals:
                continue
            update_ax = False
            if eval_name not in map_eval_to_ax:
                max_count += 1
                map_eval_to_ax[eval_name] = (
                    (
                        axes[max_count // num_cols, max_count % num_cols]
                        if num_rows > 1
                        else axes[max_count]
                    ),
                    max_count,
                )
                update_ax = True

            y_means = np.nanmean(processed_result, axis=0)
            y_stderrs = np.nanstd(processed_result, axis=0) / np.sqrt(
                len(processed_result)
            )

            (ax, ax_i) = map_eval_to_ax[eval_name]

            line = ax.plot(x_range, y_means, label=exp_name if ax_i == 0 else "")[0]
            ax.fill_between(
                x_range, (y_means - y_stderrs), (y_means + y_stderrs), alpha=0.1
            )

            if eval_name == "start_pos_0" and baseline_acc is not None:
                for curr_baseline, curr_acc, ls in zip(
                    ["$\mathbb{E}[\mathbb{I}(y = \hat{y})]$"], baseline_acc, ["-."]
                ):
                    ax.axhline(
                        curr_acc,
                        label=f"{exp_name}-{curr_baseline}" if ax_i == 0 else "",
                        c=line.get_color(),
                        linestyle=ls,
                        linewidth=1,
                    )

            if update_ax:
                ax.set_title(map_eval_to_title.get(eval_name, eval_name))
                ax.set_ylim(-1.0, 101.0)

    remaining_idx = num_cols * num_rows - (max_count + 1)
    if remaining_idx > 0:
        for ii in range(remaining_idx):
            ax_i = ii + max_count + 1
            ax = axes[ax_i // num_cols, ax_i % num_cols]
            ax.axis("off")

    fig.supxlabel("Number of updates")
    fig.supylabel("Accuracy")
    fig.legend(
        bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
        loc="lower center",
        ncols=len(agg_result),
        borderaxespad=0.0,
        frameon=True,
        fontsize="8",
    )

    fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=600)


# plot_context_length_plot()
plot_main_plot()
