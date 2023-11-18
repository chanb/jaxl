import _pickle as pickle
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import os
import seaborn as sns

from cycler import cycler

from jaxl.constants import *
from jaxl.plot_utils import set_size, pgf_with_latex


# Use the seborn style
sns.set_style("darkgrid")
sns.set_palette("colorblind")

# But with fonts from the document body
plt.rcParams.update(pgf_with_latex)


linestyle_cycler = (
    cycler(color=sns.color_palette()[:4]) +
    cycler(linestyle=['-','--',':','-.'])
)
plt.rc('axes', prop_cycle=linestyle_cycler)

# Using the set_size function as defined earlier
# doc_width_pt = 397.48499 # neurips
doc_width_pt = 452.9679 # iclr

exp_suffixes = [
    "1x_target_data",
    "2x_target_data",
    "4x_target_data",
    "8x_target_data",
]

results_per_experiment = {}
eval_dir = f"/Users/chanb/research/personal/mtil_results/final_results/data/rebuttal/evaluations/results-bc"
assert os.path.isdir(eval_dir), f"{eval_dir} is not a directory"

save_path = f"./bc_ablation-results/"
os.makedirs(save_path, exist_ok=True)
if os.path.isfile(f"{save_path}/results.pkl"):
    results = pickle.load(open(f"{save_path}/results.pkl", "rb"))
else:
    results = {}
    for eval_path, _, filenames in os.walk(eval_dir):
        for filename in filenames:
            if not filename.endswith(".pkl"):
                continue
            
            print("Processing {}/{}".format(eval_path, filename))
            variant_name = eval_path.split("/")[-3:]
            (env_name, control_mode, env_seed) = variant_name
            num_target_data = env_name.split("-")[1]
            env_name = env_name.split("-")[0]

            if num_target_data not in exp_suffixes:
                assert 0

            results.setdefault((env_name, control_mode), {})
            results[(env_name, control_mode)].setdefault(env_seed, {})
            results[(env_name, control_mode)][env_seed].setdefault(num_target_data, [])

            with open(os.path.join(eval_path, filename), "rb") as f:
                (data, paths) = pickle.load(f)

            if filename == "expert.pkl":
                if "expert" in results[(env_name, control_mode)][env_seed]:
                    continue
                results[(env_name, control_mode)][env_seed]["expert"] = data
            else:
                bc_run_result = pickle.load(
                    open(os.path.join(eval_path, filename), "rb")
                )
                results[(env_name, control_mode)][env_seed][
                    num_target_data
                ] += bc_run_result[0]

    with open(f"{save_path}/results.pkl", "wb") as f:
        pickle.dump(results, f)

def map_task(name):
    return int(name.split("x_target_data")[0])

map_env = {
    "frozenlake": "Frozen Lake",
    "cheetah": "Cheetah",
    "walker": "Walker",
    "cartpole": "Cartpole",
    "pendulum": "Pendulum",
}
map_control = {
    "discrete": "discrete",
    "continuous": "continuous",
}

env_names = [
    ("frozenlake", "discrete"),
    ("pendulum", "discrete"),
    ("cheetah", "discrete"),
    ("walker", "discrete"),
    ("cartpole", "continuous"),
    ("pendulum", "continuous"),
    ("cheetah", "continuous"),
    ("walker", "continuous"),
]

save_plot_dir = "./agg_plots"
os.makedirs(save_plot_dir, exist_ok=True)


def map_exp(name):
    splitted_name = name.split("-")
    if len(splitted_name) == 2:
        return "$N =  \\lvert \\mathcal{D} \\rvert$"
    else:
        map_amount = {
            "double": 2,
            "quadruple": 4,
            "eightfold": 8,
        }
        return "${}\\lvert \\mathcal{{D}} \\rvert$".format(map_amount[splitted_name[-1].split("_")[0]])


num_plots_per_fig = 4
num_rows = 1
num_cols = 4
from pprint import pprint
pprint(results)
for env_i, env_name in enumerate(env_names):
    ax_i = env_i % num_plots_per_fig

    if ax_i == 0:
        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=set_size(doc_width_pt, 0.95, (num_rows, num_cols), use_golden_ratio=False),
            layout="constrained",
        )

    if axes.ndim == 2:
        ax = axes[ax_i // num_cols, ax_i % num_cols]
    else:
        ax = axes[ax_i]

    ref_result = results[env_name]
    all_bc_rets = {}
    for env_seed in ref_result:
        (expert_rets, random_rets) = ref_result[env_seed]["expert"]

        def normalize(rets):
            return (rets - random_rets) / (expert_rets - random_rets)
        keys = [bc_name for bc_name in ref_result[env_seed].keys() if bc_name != "expert"]
        num_datas = [map_task(key) for key in keys]
        idxes = np.argsort(num_datas)

        for idx in idxes:
            key = keys[idx]
            num_data = num_datas[idx]

            res = ref_result[env_seed][key]
            normalized_bc_rets = normalize(res)
            all_bc_rets.setdefault(num_data, [])
            all_bc_rets[num_data].append(np.mean(normalized_bc_rets))

    num_datas = [num_datas[idx] for idx in idxes]
    means = np.array(
        [
            np.mean(all_bc_rets[num_data])
            for num_data in num_datas
        ]
    )
    stds = np.array(
        [
            np.std(all_bc_rets[num_data])
            for num_data in num_datas
        ]
    )
    print(stds)
    ax.plot(
        num_datas,
        means,
        label="BC" if ax_i == 0 else "",
        marker="^",
        ms=3.0,
    )
    ax.fill_between(
        num_datas,
        means + stds,
        means - stds,
        alpha=0.3,
    )
    ax.axhline(
        1.0,
        label="Expert" if ax_i == 0 else "",
        color="black",
        linestyle="--",
        linewidth=0.5,
    )

    ax.set_ylim(0, 1.1)

    ax.set_title(map_env[env_name[0]], fontsize=8)
    ax.xaxis.set_major_locator(tck.MultipleLocator(2))
    ax.set_xlim(num_datas[0] - 0.1, num_datas[-1] + 0.1)

    if ax_i > 0:
        ax.grid(True)
        ax.set_yticklabels([])

    if ax_i + 1 == num_plots_per_fig:
        fig.supylabel("Normalized Returns", fontsize=8)
        fig.supxlabel("Factor of $\\lvert \mathcal{D} \\rvert$", fontsize=8)
        fig.legend(
            bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
            loc="lower center",
            ncols=6,
            borderaxespad=0.0,
            frameon=True,
            fontsize="7",
        )
        # fig.suptitle("{} {}".format( map_env[env_name[0]], map_control[env_name[1]]))
        fig.savefig(
            f"{save_plot_dir}/returns-agg_{env_i // num_plots_per_fig}.pdf",
            format="pdf",
            bbox_inches="tight",
            dpi=600,
        )
