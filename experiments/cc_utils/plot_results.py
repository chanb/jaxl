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
doc_width_pt = 1000.0
interp_gap_size = 100
parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_dir",
    type=str,
    required=True,
    help="The directory that stores all experiments",
)
parser.add_argument(
    "--save_path",
    type=str,
    required=True,
    help="The directory to store the plots",
)
parser.add_argument(
    "--key",
    choices=["losses", "accuracies"],
    required=True,
    help="The key of the statistics to plot",
)
parser.add_argument(
    "--context",
    choices=["none", "last", "half"],
    required=True,
    help="The number of context from query class",
)
args = parser.parse_args()

results_dir = args.results_dir
save_path = args.save_path
key = args.key
context = args.context

os.makedirs(save_path, exist_ok=True)

# FILTERS
include_prefix = None
include_suffix = None
exclude_prefix = None
exclude_suffix = None

if context == "last":
    title = "Last Context from Query Class"
elif context == "half":
    title = "Half Contexts from Query Class"
elif context == "none":
    title = "No Context from Query Class"

if title == "Last Context from Query Class":
    eval_type = "icl"
    include_evals = [
        # ICL - last context
        "pretrain-sample_high_prob_class_only-start_pos_1",
        "pretrain-sample_low_prob_class_only-start_pos_1",
        "pretrain-sample_high_prob_class_only-start_pos_1-flip_label",
        "pretrain-sample_low_prob_class_only-start_pos_1-flip_label",
    ]
elif title == "No Context from Query Class":
    eval_type = "iwl"
    include_evals = [
        # IWL
        "pretrain-sample_high_prob_class_only-start_pos_0",
        "pretrain-sample_low_prob_class_only-start_pos_0",
        "pretrain-sample_high_prob_class_only-start_pos_0-flip_label",
        "pretrain-sample_low_prob_class_only-start_pos_0-flip_label",
    ]
elif title == "Half Contexts from Query Class":
    eval_type = "icl_half_contexts"
    include_evals = [
        # ICL - half contexts
        "pretrain-sample_high_prob_class_only-start_pos_4",
        "pretrain-sample_low_prob_class_only-start_pos_4",
        "pretrain-sample_high_prob_class_only-start_pos_4-flip_label",
        "pretrain-sample_low_prob_class_only-start_pos_4-flip_label",
    ]
else:
    include_evals = None

map_eval_to_title = {
    "pretrain-sample_high_prob_class_only-start_pos_0": "High Freq. Only",
    "pretrain-sample_low_prob_class_only-start_pos_0": "Low Freq. Only",
    "pretrain-sample_high_prob_class_only-start_pos_0-flip_label": "High Freq. Only w/ Flipped Label",  # flip
    "pretrain-sample_low_prob_class_only-start_pos_0-flip_label": "Low Freq. Only w/ Flipped Label",
    "pretrain-sample_high_prob_class_only-start_pos_1": "High Freq. Only",
    "pretrain-sample_low_prob_class_only-start_pos_1": "Low Freq. Only",
    "pretrain-sample_high_prob_class_only-start_pos_1-flip_label": "High Freq. Only w/ Flipped Label",
    "pretrain-sample_low_prob_class_only-start_pos_1-flip_label": "Low Freq. Only w/ Flipped Label",
    "pretrain-sample_high_prob_class_only-start_pos_4": "High Freq. Only",
    "pretrain-sample_low_prob_class_only-start_pos_4": "Low Freq. Only",
    "pretrain-sample_high_prob_class_only-start_pos_4-flip_label": "High Freq. Only w/ Flipped Label",
    "pretrain-sample_low_prob_class_only-start_pos_4-flip_label": "Low Freq. Only w/ Flipped Label",
}
max_checkpoint_steps = 0
max_num_evals = 0

exp_runs = dict()
for run in tqdm(os.listdir(results_dir)):
    exp_name = run.split("-")[:-8]
    variant = "-".join(exp_name[:-1])
    seed = int(exp_name[-1].split("_")[-1])
    if include_prefix and not variant.startswith(include_prefix):
        continue
    if include_suffix and not variant.endswith(include_suffix):
        continue
    if exclude_prefix and variant.startswith(exclude_prefix):
        continue
    if exclude_suffix and variant.endswith(exclude_suffix):
        continue

    exp_runs.setdefault(variant, dict())
    data = pickle.load(open(os.path.join(results_dir, run, "evaluation.pkl"), "rb"))
    exp_runs[variant][seed] = dict(
        checkpoint_steps=data["checkpoint_steps"],
        accuracies=data["accuracies"],
        losses=data["losses"],
    )
    max_checkpoint_steps = max(max_checkpoint_steps, np.max(data["checkpoint_steps"]))
    max_num_evals = max(
        max_num_evals, len([eval_name for eval_name in data["accuracies"]])
    )

if include_evals:
    max_num_evals = len(include_evals)


def process_exp_runs(exp_runs: dict, x_range: chex.Array, key="accuracies"):
    interpolated_results = dict()
    for run_i, (run_name, exp_run) in enumerate(exp_runs.items()):
        curr_checkpoint_steps = exp_run["checkpoint_steps"]
        curr_accuracies = exp_run[key]

        for eval_name, accuracies in curr_accuracies.items():
            interpolated_results.setdefault(
                eval_name, np.zeros((len(exp_runs), len(x_range)))
            )
            interpolated_results[eval_name][run_i] = np.interp(
                x_range, curr_checkpoint_steps, accuracies
            )
    return interpolated_results


num_cols = 2

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
        if key == "accuracies":
            map_eval_to_ax[eval_name][0].set_ylim(-1.0, 101.0)

variants = sorted(exp_runs.keys())
print(variants)
for variant in tqdm(variants):
    exp_run = exp_runs[variant]
    x_range = np.arange(0, max_checkpoint_steps + 1, interp_gap_size)
    processed_results = process_exp_runs(exp_run, x_range, key)

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
        y_stderrs = np.nanstd(processed_result, axis=0) / np.sqrt(len(processed_result))

        (ax, ax_i) = map_eval_to_ax[eval_name]

        line = ax.plot(x_range, y_means, label=variant if ax_i == 0 else "")[0]
        ax.fill_between(
            x_range, (y_means - y_stderrs), (y_means + y_stderrs), alpha=0.1
        )

        if update_ax:
            ax.set_title(map_eval_to_title.get(eval_name, eval_name))
            if key == "accuracies":
                ax.set_ylim(-1.0, 101.0)

remaining_idx = num_cols * num_rows - (max_count + 1)
if remaining_idx > 0:
    for ii in range(remaining_idx):
        ax_i = ii + max_count + 1
        ax = axes[ax_i // num_cols, ax_i % num_cols]
        ax.axis("off")

fig.supxlabel("Number of updates")
fig.supylabel(key)
fig.suptitle(title)
fig.legend(
    bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
    loc="lower center",
    ncols=len(exp_runs.keys()),
    borderaxespad=0.0,
    frameon=True,
    fontsize="8",
)

fig.savefig(
    os.path.join(save_path, "{}-{}.pdf".format(eval_type, key)),
    format="pdf",
    bbox_inches="tight",
    dpi=600,
)
