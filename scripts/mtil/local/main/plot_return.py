import _pickle as pickle
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
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

experiment_name = "results-finetune_mtbc_main"
experiment_name_suffixes = (
    "",
    "-double_source_data",
    "-quadruple_source_data",
    "-eightfold_source_data",
)
bc_name = "results-bc_less_data"

results_per_experiment = {}
experiment_dir = f"/Users/chanb/research/personal/mtil_results/final_results/data/evaluations/{experiment_name}"
bc_dir = f"/Users/chanb/research/personal/mtil_results/final_results/data/evaluations/{bc_name}"
assert os.path.isdir(experiment_dir), f"{experiment_dir} is not a directory"

for exp_i, suffix in enumerate(experiment_name_suffixes):
    curr_exp = f"{experiment_name}{suffix}"
    print(f"Processing {curr_exp}")
    save_path = f"./{curr_exp}-results"
    os.makedirs(save_path, exist_ok=True)

    if os.path.isfile(f"{save_path}/results.pkl"):
        results = pickle.load(open(f"{save_path}/results.pkl", "rb"))
    else:
        results = {}
        for eval_path, _, filenames in os.walk(experiment_dir):
            for filename in filenames:
                if not filename.endswith(".pkl"):
                    continue

                print("Processing {}".format(eval_path))
                variant_name = eval_path.split("/")[-3:]
                (env_name, control_mode, env_seed) = variant_name
                env_name = env_name.split("-")[0]

                if f"{env_name}{suffix}" not in eval_path.split("/"):
                    continue

                results.setdefault((env_name, control_mode), {})
                results[(env_name, control_mode)].setdefault(env_seed, {})

                with open(os.path.join(eval_path, filename), "rb") as f:
                    (data, paths) = pickle.load(f)

                if filename == "expert.pkl":
                    if exp_i > 0:
                        continue
                    results[(env_name, control_mode)][env_seed]["expert"] = data

                    # Get BC
                    results[(env_name, control_mode)][env_seed]["bc"] = []
                    bc_run_dir = os.path.join(bc_dir, *variant_name)
                    for bc_run in os.listdir(bc_run_dir):
                        if bc_run == "expert.pkl":
                            continue
                        bc_run_result = pickle.load(
                            open(os.path.join(bc_run_dir, bc_run), "rb")
                        )
                        results[(env_name, control_mode)][env_seed][
                            "bc"
                        ] += bc_run_result[0]
                else:
                    num_tasks = int(filename[:-4].split("num_tasks_")[-1])
                    results[(env_name, control_mode)][env_seed].setdefault("mtbc", [])
                    results[(env_name, control_mode)][env_seed]["mtbc"].append(
                        (num_tasks, paths, data)
                    )

        with open(f"{save_path}/results.pkl", "wb") as f:
            pickle.dump(results, f)

    results_per_experiment[curr_exp] = results


map_env = {
    "frozenlake": "frozen lake",
    "cheetah": "cheetah",
    "walker": "walker",
    "cartpole": "cartpole",
    "pendulum": "pendulum",
}
map_control = {
    "discrete": "discrete",
    "continuous": "continuous",
}

env_names = [
    ("frozenlake", "discrete"),
    ("cartpole", "continuous"),
    ("pendulum", "discrete"),
    ("pendulum", "continuous"),
    ("cheetah", "discrete"),
    ("cheetah", "continuous"),
    ("walker", "discrete"),
    ("walker", "continuous"),
]

save_plot_dir = "./agg_plots"
os.makedirs(save_plot_dir, exist_ok=True)


def map_exp(name):
    splitted_name = name.split("-")
    if len(splitted_name) == 2:
        return "$N = M$"
    else:
        map_amount = {
            "double": 2,
            "quadruple": 4,
            "eightfold": 8,
        }
        return "${}N$".format(map_amount[splitted_name[-1].split("_")[0]])


# Plot main return
for env_name in env_names:
    num_envs = len(results[env_name])
    num_rows = math.ceil(num_envs / 2)
    num_cols = 2

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=set_size(doc_width_pt, 0.95, (num_rows, num_cols)),
        layout="constrained",
    )

    ref_result = results_per_experiment[experiment_name][env_name]
    for ax_i, env_seed in enumerate(ref_result):
        if axes.ndim == 2:
            ax = axes[ax_i // num_cols, ax_i % num_cols]
        else:
            ax = axes[ax_i]

        (expert_rets, random_rets) = ref_result[env_seed]["expert"]

        def normalize(rets):
            return (rets - random_rets) / (expert_rets - random_rets)
        
        ax.axhline(
            1.0,
            label="Expert" if ax_i == 0 else "",
            color="black",
            linestyle="--",
        )

        normalized_bc_rets = normalize(ref_result[env_seed]["bc"])
        bc_mean = np.mean(normalized_bc_rets)
        bc_std = np.std(normalized_bc_rets)
        ax.axhline(
            bc_mean,
            label="BC" if ax_i == 0 else "",
            color="grey",
            linestyle="--",
        )
        num_tasks, _, _ = list(zip(*ref_result[env_seed]["mtbc"]))
        num_tasks = np.array(num_tasks)
        unique_num_tasks = np.unique(num_tasks)
        ax.fill_between(
            (unique_num_tasks[0], unique_num_tasks[-1]),
            bc_mean + bc_std,
            bc_mean - bc_std,
            color="grey",
            alpha=0.3,
        )

        for suffix in experiment_name_suffixes:
            curr_exp = f"{experiment_name}{suffix}"
            res = results_per_experiment[curr_exp][env_name][env_seed]
            num_tasks, _, returns = list(zip(*res["mtbc"]))
            num_tasks = np.array(num_tasks)
            returns = np.array(returns)

            means = []
            stds = []

            # if env_name[0] == "cheetah" and env_name[1] == "discrete":
            #     print("-----")
            #     print(env_seed, num_tasks, returns)
            if len(num_tasks) < 5:
                print(suffix, env_name, env_seed, num_tasks, returns)
            for num_task in unique_num_tasks:
                curr_num_task_rets = normalize(returns[num_tasks == num_task])
                means.append(np.mean(curr_num_task_rets))
                stds.append(np.std(curr_num_task_rets))

            means = np.array(means)
            stds = np.array(stds)

            ax.plot(
                unique_num_tasks,
                means,
                marker="^",
                label=map_exp(curr_exp) if ax_i == 0 else "",
            )
            ax.fill_between(
                unique_num_tasks,
                means + stds,
                means - stds,
                alpha=0.3,
            )

        ax.xaxis.set_major_locator(tck.MultipleLocator(4))
        ax.set_xlim(unique_num_tasks[0], unique_num_tasks[-1])
        # ax.legend()

    fig.supylabel("Expected Return")
    fig.supxlabel("Number of Tasks")
    fig.legend(
        bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
        loc="lower center",
        ncols=4,
        borderaxespad=0.0,
        frameon=True,
        fontsize="5",
    )
    # fig.suptitle("{} {}".format(map_control[env_name[1]], map_env[env_name[0]]))
    fig.savefig(
        f"{save_plot_dir}/returns-{env_name[0]}_{env_name[1]}.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=600,
    )
