import _pickle as pickle
import chex
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

results_dir = "./results"
ablation_name = "single_sample-pixel_noise_0.1"

agg_result_path = os.path.join(results_dir, ablation_name, "agg_data/accuracies.pkl")
plot_path = os.path.join(results_dir, ablation_name, "plots")
save_path = os.path.join(plot_path, "{}.pdf".format(ablation_name))

os.makedirs(plot_path, exist_ok=True)

agg_result = pickle.load(open(agg_result_path, "rb"))

max_num_evals = 0
max_checkpoint_steps = 0
for exp_name, exp_runs in agg_result.items():
    for run_name, exp_run in exp_runs.items():
        curr_checkpoint_steps = np.max(exp_run["checkpoint_steps"])
        curr_num_evals = len(exp_run["accuracies"])

        if curr_checkpoint_steps > max_checkpoint_steps:
            max_checkpoint_steps = curr_checkpoint_steps
        
        if curr_num_evals > max_num_evals:
            max_num_evals = curr_num_evals

def process_exp_runs(exp_runs: dict, x_range: chex.Array):
    interpolated_results = dict()
    for run_i, (run_name, exp_run) in enumerate(exp_runs.items()):
        curr_checkpoint_steps = exp_run["checkpoint_steps"]
        curr_accuracies = exp_run["accuracies"]

        for eval_name, accuracies in curr_accuracies.items():
            interpolated_results.setdefault(
                eval_name,
                np.zeros((len(exp_runs), len(x_range)))
            )
            interpolated_results[eval_name][run_i] = np.interp(
                x_range,
                curr_checkpoint_steps,
                accuracies
            )
    return interpolated_results

num_cols = 3
num_rows = math.ceil(max_num_evals / num_cols)
fig, axes = plt.subplots(
    num_rows,
    num_cols,
    figsize=set_size(doc_width_pt, 0.95, (num_rows, num_cols)),
    layout="constrained",
)

map_eval_to_ax = {}
max_count = -1
x_range = np.arange(0, max_checkpoint_steps + 1, 1000)
for exp_name, exp_runs in tqdm(agg_result.items()):
    processed_results = process_exp_runs(exp_runs, x_range)

    for eval_name, processed_result in processed_results.items():
        update_ax = False
        if eval_name not in map_eval_to_ax:
            max_count += 1
            map_eval_to_ax[eval_name] = (axes[max_count // num_cols, max_count % num_cols], max_count)
            update_ax = True

        y_means = np.nanmean(processed_result, axis=0)
        y_stderrs = np.nanstd(processed_result, axis=0) / np.sqrt(len(processed_result))

        (ax, ax_i) = map_eval_to_ax[eval_name]
        ax.plot(x_range, y_means, label=exp_name if ax_i == 0 else "")
        ax.fill_between(
            x_range,
            (y_means - y_stderrs),
            (y_means + y_stderrs),
            alpha=0.1
        )

        if update_ax:
            ax.set_title(eval_name)
            ax.set_ylim(-1.0, 101.0)

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
