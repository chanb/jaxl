import _pickle as pickle
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
control_mode = "continuous"
save_path = f"../local/main"

configs.append(
    (
        task,
        control_mode,
        save_path,
    )
)

task = "pendulum"
control_mode = "discrete"
save_path = f"../local/main"

configs.append(
    (
        task,
        control_mode,
        save_path,
    )
)


def get_result(
    task,
    control_mode,
    save_path,
):
    mtbc_path = os.path.join(
        save_path,
        "finetune_mtbc_main-more_target_data-results-mtbc_performance-{}_{}".format(
            task, control_mode
        ),
        "final_result.pkl",
    )

    bc_path = os.path.join(
        save_path,
        "bc_main-results-bc_performance-{}_{}".format(
            task, control_mode
        ),
        "final_result.pkl",
    )

    bc_less_data_path = os.path.join(
        save_path,
        "bc_less_data-results-bc_performance-{}_{}".format(
            task, control_mode
        ),
        "final_result.pkl",
    )

    named_paths = {
        "mtbc": mtbc_path,
        "bc": bc_path,
        "bc_less_data": bc_less_data_path
    }

    results = {}

    for name, curr_res_path in named_paths.items():
        assert os.path.isfile(curr_res_path)
        result_per_env = pickle.load(open(curr_res_path, "rb"))

        for env_seed, res in result_per_env.items():
            results.setdefault(env_seed, {})
            if name == "mtbc":
                results[env_seed]["expert"] = res["expert"]
                num_tasks = np.sort([int(num_task.split("num_tasks_")[-1]) for num_task in res if num_task != "expert"])
                keys = ["num_tasks_{}".format(num_task) for num_task in num_tasks]
                results[env_seed]["mtbc"] = {
                    "num_tasks": num_tasks,
                    "means": np.array([res[num_task][0] for num_task in keys]),
                    "stds": np.array([res[num_task][1] for num_task in keys])
                }
            else:
                results[env_seed][name] = {
                    "means": res["bc"][0],
                    "stds": res["bc"][1],
                }

    return results

for config in configs:
    (
        task,
        control_mode,
        save_path,
    ) = config

    results = get_result(
        task,
        control_mode,
        save_path,
    )

    num_rows = math.ceil(len(results)) // 2
    num_cols = len(results)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=set_size(doc_width_pt, 0.95, (num_rows, num_cols)),
        layout="constrained",
    )

    named_keys = ["mtbc", "bc", "bc_less_data", "expert"]
    for ax_idx, env_seed in enumerate(results):
        curr_env_result = results[env_seed]

        if len(results) == 1:
            ax = axes
        elif num_rows == 1:
            ax = axes[ax_idx]
        else:
            row_i = ax_idx // num_cols
            col_i = ax_idx % num_cols
            ax = axes[row_i, col_i]
        
        for name in named_keys:
            vals = curr_env_result[name]
            if name == "mtbc":
                ax.plot(
                    vals["num_tasks"],
                    vals["means"],
                    label=name
                )
                ax.fill_between(
                    vals["num_tasks"],
                    vals["means"] + vals["stds"],
                    vals["means"] - vals["stds"],
                    alpha=0.3,
                )
            else:
                if isinstance(vals, dict):
                    ax.axhline(
                        vals["means"],
                        label=name
                    )
                    ax.fill_between(
                        curr_env_result["mtbc"]["num_tasks"],
                        vals["means"] + vals["stds"],
                        vals["means"] - vals["stds"],
                        alpha=0.3,
                    )
                else:
                    ax.axhline(
                        vals,
                        label=name
                    )

    fig.legend(
        bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
        loc="lower center",
        ncols=4,
        borderaxespad=0.0,
        frameon=True,
        fontsize="5",
    )

    fig.supylabel("Expected Return")
    fig.supxlabel("Amount of Source Tasks")
    fig.savefig(f"./main_result-{task}_{control_mode}.pdf", format="pdf", bbox_inches="tight", dpi=600)
