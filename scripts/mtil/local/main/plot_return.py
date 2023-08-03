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

experiment_name = "results-finetune_mtbc_main"
bc_name = "results-bc_less_data"
save_path = f"./{experiment_name}-results"
experiment_dir = f"/Users/chanb/research/personal/mtil_results/final_results/data/evaluations/{experiment_name}"
bc_dir = f"/Users/chanb/research/personal/mtil_results/final_results/data/evaluations/{bc_name}"

assert os.path.isdir(experiment_dir), f"{experiment_dir} is not a directory"

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

            results.setdefault((env_name, control_mode), {})
            results[(env_name, control_mode)].setdefault(env_seed, {})

            with open(os.path.join(eval_path, filename), "rb") as f:
                data = pickle.load(f)

            if filename == "expert.pkl":
                results[(env_name, control_mode)][env_seed]["expert"] = data

                # Get BC
                results[(env_name, control_mode)][env_seed]["bc"] = []
                bc_run_dir = os.path.join(bc_dir, *variant_name)
                for bc_run in os.listdir(bc_run_dir):
                    if bc_run == "expert.pkl":
                        continue
                    bc_run_result = pickle.load(open(os.path.join(bc_run_dir, bc_run), "rb"))
                    results[(env_name, control_mode)][env_seed]["bc"] += bc_run_result
            else:
                num_tasks = int(filename[:-4].split("num_tasks_")[-1])
                results[(env_name, control_mode)][env_seed].setdefault("mtbc", [])
                results[(env_name, control_mode)][env_seed]["mtbc"].append((num_tasks, data))

    with open(f"{save_path}/results.pkl", "wb") as f:
        pickle.dump(results, f)


map_env = {
    "frozenlake": "Frozen Lake",
    "cheetah": "Cheetah Run",
    "walker": "Walker Walk",
    "cartpole": "Cartpole Swing Up",
    "pendulum": "Pendulum",
}
map_control = {
    "discrete": "Discrete",
    "continuous": "Continuous",
}

env_names = [
    # ("frozenlake", "discrete"),
    # ("cartpole", "continuous"),
    ("pendulum", "discrete"),
    ("pendulum", "continuous"),
    ("cheetah", "discrete"),
    ("cheetah", "continuous"),
    ("walker", "discrete"),
    ("walker", "continuous"),
]

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
    for ax_i, (env_seed, res) in enumerate(results[env_name].items()):
        if axes.ndim == 2:
            ax = axes[ax_i // num_cols, ax_i % num_cols]
        else:
            ax = axes[ax_i]

        num_tasks, returns = list(zip(*res["mtbc"]))
        num_tasks = np.array(num_tasks)
        returns = np.array(returns)
        unique_num_tasks = np.unique(num_tasks)

        ax.axhline(
            res["expert"],
            label="expert" if ax_i == 0 else "",
            color="black",
            linestyle="--",
        )

        bc_mean = np.mean(res["bc"])
        bc_std = np.std(res["bc"])
        ax.axhline(
            bc_mean,
            label="BC" if ax_i == 0 else "",
            color="grey",
            linestyle="--",
        )
        ax.fill_between(
            (unique_num_tasks[0], unique_num_tasks[-1]),
            bc_mean + bc_std,
            bc_mean - bc_std,
            color="grey",
            alpha=0.3,
        )

        means = []
        stds = []

        for num_task in unique_num_tasks:
            means.append(np.mean(returns[num_tasks == num_task]))
            stds.append(np.std(returns[num_tasks == num_task]))

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(
            unique_num_tasks, means, marker="x", label="MTBC" if ax_i == 0 else ""
        )
        ax.fill_between(
            unique_num_tasks,
            means + stds,
            means - stds,
            alpha=0.3,
        )

        ax.set_xlim(unique_num_tasks[0], unique_num_tasks[-1])
        ax.set_title(env_seed)
        ax.legend()

    fig.supylabel("Expected Return")
    fig.supxlabel("Number of Tasks")
    fig.suptitle("{} {}".format(map_env[env_name[0]], map_control[env_name[1]]))
    fig.savefig(f"{save_path}/returns-{env_name[0]}_{env_name[1]}.pdf", format="pdf", bbox_inches="tight", dpi=600)
