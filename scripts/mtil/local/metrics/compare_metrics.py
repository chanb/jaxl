from itertools import chain
from matplotlib.ticker import FormatStrFormatter
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager
from scipy.stats import linregress, spearmanr, kendalltau, pearsonr
from typing import Iterable

import _pickle as pickle
import gymnasium as gym
import gzip
import jax
import jax.numpy as jnp
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os

from jaxl.losses import get_loss_function
from jaxl.models import get_model
from jaxl.plot_utils import set_size, pgf_with_latex
from jaxl.utils import parse_dict

import jaxl.envs


# Use the seborn style
plt.style.use("seaborn")
# But with fonts from the document body
plt.rcParams.update(pgf_with_latex)

# Using the set_size function as defined earlier
doc_width_pt = 452.9679

eps = 1e-3


def l2_distance(finetune_config, pretrain_config):
    target_env_config = finetune_config["learner_config"]["env_configs"][0]
    target_env_config["env_name"] = target_env_config["env_name"].split("/")[-1]
    target_env = gym.make(
        target_env_config["env_name"], **target_env_config["env_kwargs"]
    )
    target_env_params = target_env.get_config()["modified_attributes"]
    target_env.close()

    source_env_configs = pretrain_config["learner_config"]["env_configs"]
    source_env_paramss = []
    for source_env_config in source_env_configs:
        source_env_config["env_name"] = source_env_config["env_name"].split("/")[-1]
        source_env = gym.make(
            source_env_config["env_name"], **source_env_config["env_kwargs"]
        )
        source_env_paramss.append(source_env.get_config()["modified_attributes"])
        source_env.close()

    target_env_vec = np.array(
        list(
            chain(
                *[
                    val if isinstance(val, Iterable) else [val]
                    for val in jax.tree_util.tree_leaves(target_env_params)
                ]
            )
        )
    )
    source_env_vecs = np.array(
        [
            list(
                chain(
                    *[
                        val if isinstance(val, Iterable) else [val]
                        for val in jax.tree_util.tree_leaves(source_env_params)
                    ]
                )
            )
            for source_env_params in source_env_paramss
        ]
    )

    pairwise_distance = np.sum(
        (source_env_vecs - target_env_vec[None, :]) ** 2, axis=-1
    )
    avg_distance = np.mean(pairwise_distance)
    std_distance = np.std(pairwise_distance)
    min_distance = np.min(pairwise_distance)
    return avg_distance, std_distance, min_distance


def approx_kl(
    pretrain_run_dir,
    finetune_config,
    pretrain_config,
    finetune_dataset_path,
    pretrain_dataset_paths,
):
    target_env_buffer = pickle.load(gzip.open(finetune_dataset_path, "rb"))
    target_buffer_size = finetune_config["learner_config"]["buffer_configs"][0][
        "set_size"
    ]

    source_statess = []
    source_h_statess = []
    source_actionss = []
    num_tasks = 0
    for buffer_config, pretrain_dataset_path in zip(
        pretrain_config["learner_config"]["buffer_configs"], pretrain_dataset_paths
    ):
        source_env_buffer = pickle.load(gzip.open(pretrain_dataset_path, "rb"))
        source_buffer_size = buffer_config["set_size"]
        source_statess.append(source_env_buffer["observations"][:source_buffer_size])
        source_h_statess.append(source_env_buffer["hidden_states"][:source_buffer_size])
        source_actionss.append(source_env_buffer["actions"][:source_buffer_size])
        num_tasks += 1

    source_statess = np.stack(source_statess)
    source_h_statess = np.stack(source_h_statess)
    source_actionss = np.stack(source_actionss)

    target_states = np.tile(
        target_env_buffer["observations"][:target_buffer_size][None, ...],
        (num_tasks, 1, 1),
    )
    target_h_states = np.tile(
        target_env_buffer["hidden_states"][:target_buffer_size][None, ...],
        (num_tasks, 1, 1),
    )
    target_actions = np.tile(
        target_env_buffer["actions"][:target_buffer_size][None, ...], (num_tasks, 1, 1)
    )

    model = get_model(
        target_states.shape[1:],
        target_env_buffer["act_dim"],
        parse_dict(pretrain_config["model_config"]),
    )

    checkpoint_manager = CheckpointManager(
        os.path.join(pretrain_run_dir, "models"),
        PyTreeCheckpointer(),
    )
    all_params = checkpoint_manager.restore(checkpoint_manager.latest_step())
    model_dict = all_params["model_dict"]

    loss_fn = get_loss_function(
        model.predictor.model,
        pretrain_config["learner_config"]["losses"][0],
        parse_dict(pretrain_config["learner_config"]["loss_settings"][0]),
        num_classes=target_env_buffer["act_dim"][-1],
    )

    def loss(
        model_dicts,
        obss,
        h_states,
        acts,
        *args,
        **kwargs,
    ):
        reps, _ = jax.vmap(model.encode, in_axes=[None, 0, 0])(
            model_dicts["encoder"], obss, h_states
        )

        bc_loss, bc_aux = jax.vmap(loss_fn)(
            model_dicts["predictor"],
            reps,
            h_states,
            acts,
        )

        return bc_loss, bc_aux

    compute_source_loss = jax.jit(loss)
    compute_target_loss = jax.jit(loss)

    source_loss, _ = compute_source_loss(
        model_dict["model"]["policy"],
        source_statess,
        source_h_statess,
        source_actionss,
    )

    target_loss, _ = compute_target_loss(
        model_dict["model"]["policy"],
        target_states,
        target_h_states,
        target_actions,
    )
    best_target_loss = jnp.min(target_loss)

    diversity = jnp.mean(source_loss) / (best_target_loss + eps)
    return diversity


def expert_data_performance(
    pretrain_config, finetune_dataset_path, pretrain_dataset_paths
):
    target_env_buffer = pickle.load(gzip.open(finetune_dataset_path, "rb"))
    target_buffer_size = finetune_config["learner_config"]["buffer_configs"][0][
        "set_size"
    ]
    target_data_performance = np.mean(target_env_buffer["rewards"][:target_buffer_size])

    source_data_performances = []
    for buffer_config, pretrain_dataset_path in zip(
        pretrain_config["learner_config"]["buffer_configs"], pretrain_dataset_paths
    ):
        source_env_buffer = pickle.load(gzip.open(pretrain_dataset_path, "rb"))
        source_buffer_size = buffer_config["set_size"]
        source_data_performances.append(
            np.mean(source_env_buffer["rewards"][:source_buffer_size])
        )

    source_data_performances = np.array(source_data_performances)

    pairwise_distance = source_data_performances - target_data_performance

    avg_distance = np.mean(pairwise_distance)
    std_distance = np.std(pairwise_distance)
    min_distance = np.min(pairwise_distance)
    return avg_distance, std_distance, min_distance


dataset_dir = (
    "/Users/chanb/research/personal/mtil_results/final_results/data/expert_data"
)
base_dir = "/Users/chanb/research/personal/mtil_results/final_results/data/"
finetune_dir = "finetune_mtbc_main"
pretrain_dir = "pretrain_mtbc_main"
suffixes = (
    "",
    "-double_source_data",
    "-quadruple_source_data",
    "-eightfold_source_data",
)


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
save_path = f"./final_results/{finetune_dir}"
os.makedirs(save_path, exist_ok=True)

if os.path.isfile(f"{save_path}/diversity.pkl"):
    results = pickle.load(open(f"{save_path}/diversity.pkl", "rb"))
else:
    results = {}
    for env_name in env_names:
        print("Processing {}".format(env_name))
        diversities = {}

        (task, control_mode) = env_name

        for suffix in suffixes:
            finetune_runs_dir = os.path.join(
                base_dir, f"{finetune_dir}{suffix}", task, control_mode, "runs"
            )

            for finetune_run_dir, _, filenames in os.walk(finetune_runs_dir):
                for filename in filenames:
                    if filename != "config.json":
                        continue

                    (
                        env_variant,
                        num_task,
                        pretrain_model_seed,
                    ) = finetune_run_dir.split("/")[-4:-1]
                    dataset_task_name = "{}_{}".format(task, control_mode[:4])
                    num_task_int = int(num_task.split("num_tasks_")[-1])
                    pretrain_model_seed_int = int(
                        pretrain_model_seed.split("pretrained_model_seed_")[-1]
                    )
                    env_seed = env_variant.split(".")[2]

                    diversities.setdefault(num_task_int, [])

                    with open(os.path.join(finetune_run_dir, "config.json"), "r") as f:
                        finetune_config = json.load(f)

                    pretrain_run_dir = os.path.join(
                        base_dir,
                        f"{pretrain_dir}{suffix}",
                        task,
                        control_mode,
                        "runs",
                        num_task,
                        os.path.basename(
                            os.path.dirname(
                                finetune_config["learner_config"]["load_encoder"]
                            )
                        ),
                    )
                    with open(os.path.join(pretrain_run_dir, "config.json"), "r") as f:
                        pretrain_config = json.load(f)

                    finetune_dataset_path = os.path.join(
                        dataset_dir,
                        dataset_task_name,
                        os.path.basename(
                            finetune_config["learner_config"]["buffer_configs"][0][
                                "load_buffer"
                            ]
                        ),
                    )

                    pretrain_dataset_paths = [
                        os.path.join(
                            dataset_dir,
                            dataset_task_name,
                            os.path.basename(buffer_config["load_buffer"]),
                        )
                        for buffer_config in pretrain_config["learner_config"][
                            "buffer_configs"
                        ]
                    ]

                    avg_distance, std_distance, min_distance = l2_distance(
                        finetune_config, pretrain_config
                    )
                    l2_diversity = (avg_distance, std_distance, min_distance)

                    avg_distance, std_distance, min_distance = expert_data_performance(
                        pretrain_config, finetune_dataset_path, pretrain_dataset_paths
                    )
                    data_performance_diversity = (
                        avg_distance,
                        std_distance,
                        min_distance,
                    )

                    kl_diversity = approx_kl(
                        pretrain_run_dir,
                        finetune_config,
                        pretrain_config,
                        finetune_dataset_path,
                        pretrain_dataset_paths,
                    )

                    diversities[num_task_int].append(
                        (
                            env_seed,
                            pretrain_model_seed_int,
                            suffix,
                            (
                                l2_diversity,
                                data_performance_diversity,
                                kl_diversity,
                            ),
                        )
                    )
        results[env_name] = diversities

    with open(os.path.join(save_path, "diversity.pkl"), "wb") as f:
        pickle.dump(results, f)


# Plot diversity
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
map_diversity = [
    "L2",
    "Data Performance",
    "Approx. KL",
]

returns = {}

for suffix in suffixes:
    return_pkl = f"/Users/chanb/research/personal/jaxl/scripts/mtil/local/main/results-finetune_mtbc_main{suffix}-results/results.pkl"
    with open(return_pkl, "rb") as f:
        exp_returns = pickle.load(f)
    returns[suffix] = exp_returns

correlation_per_env = {}
for env_name in env_names:
    num_envs = len(results[env_name])
    num_rows = num_envs
    num_cols = 3

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=set_size(doc_width_pt, 0.95, (num_rows, num_cols)),
        layout="constrained",
    )
    all_correlations = []
    num_tasks = sorted(list(results[env_name].keys()))
    for row_i, num_task in enumerate(num_tasks):
        res = results[env_name][num_task]
        xs = []
        ys = []
        curr_env_seed = None
        diversities_to_add = []
        returns_to_add = None
        for env_seed, pretrain_model_seed, suffix, diversities in res:
            (expert_rets, random_rets) = returns[""][env_name][env_seed]["expert"]

            def normalize(rets):
                return (rets - random_rets) / (expert_rets - random_rets)

            mtbc_variants = returns[suffix][env_name][env_seed]["mtbc"]
            for mtbc_variant in mtbc_variants:
                if mtbc_variant[0] != num_task:
                    continue

                for variant_i, variant_path in enumerate(mtbc_variant[1]):
                    if (
                        int(
                            variant_path.split("/")[-2].split("pretrained_model_seed_")[
                                -1
                            ]
                        )
                        == pretrain_model_seed
                    ):
                        to_add = (
                            diversities[0][0] / (diversities[0][2] + eps),
                            diversities[1][0] / (diversities[1][2] + eps),
                            diversities[2],
                        )
                        xs.append(to_add)
                        ys.append(normalize(mtbc_variant[2][variant_i]))

        xs = np.array(xs).T
        ys = np.array(ys)

        correlations = []
        for col_i, x in enumerate(xs):
            ax = axes[row_i, col_i]

            # if col_i == 0:
            #     x = 1 - jax.nn.sigmoid(x / 100)
            # elif col_i == 1:
            #     x = 1 - jax.nn.sigmoid(x)
            # elif col_i == 2:
            #     x = jax.nn.sigmoid(x)

            res = linregress(x, ys)
            lin_x = np.array([np.min(x), np.max(x)])
            lin_y = res.intercept + res.slope * lin_x

            correlations.append(
                (
                    pearsonr(x, ys).statistic,
                    spearmanr(x, ys).statistic,
                    kendalltau(x, ys).statistic,
                )
            )

            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            lin_x = np.array([0, 1])
            ax.plot(
                lin_x,
                lin_y,
                "red",
                label=f"fitted line" if row_i + col_i == 0 else "",
                linewidth=1.0,
                linestyle="--",
            )
            ax.scatter(
                x,
                ys,
                s=1,
            )
            # if col_i == 0:
            #     ax.set_ylabel(num_task)
            if row_i + 1 == num_envs:
                ax.set_xlabel(map_diversity[col_i])
            if row_i + col_i == 0:
                ax.legend()
            # ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.set_ylim(0.0, 1.1)

        all_correlations.append(correlations)

    (task, control_mode) = env_name
    # fig.suptitle("{} {}".format(map_env[task], map_control[control_mode]))
    fig.supylabel("Expected Return")
    fig.supxlabel("Diversity")
    fig.savefig(
        f"{save_path}/diversity_return-{task}_{control_mode}.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=600,
    )
    print(env_name)
    print(all_correlations)
    np_correlations = np.array(all_correlations)
    print(
        "AVG CORRELATION {} +/- {}".format(
            np.mean(np_correlations, axis=0), np.std(np_correlations, axis=0)
        )
    )

    correlation_per_env[env_name] = np_correlations

num_envs = len(env_names)
control_modes = ["discrete", "continuous"]
envs = (
    ("frozenlake", "pendulum", "cheetah", "walker"),
    ("cartpole", "pendulum", "cheetah", "walker"),
)
diversity_names = (
    "L2",
    "Data Perf.",
    "Approx. KL",
)
map_env = {
    "frozenlake": "Frozen Lake",
    "cheetah": "Cheetah",
    "walker": "Walker",
    "cartpole": "Cartpole",
    "pendulum": "Pendulum",
}
map_control = {
    "discrete": "Discrete",
    "continuous": "Continuous",
}

for idx_i in range(len(control_modes)):
    control_mode = control_modes[idx_i]
    latex_content_to_write = "\\begin{table}[t]\n"
    latex_content_to_write += "  \\caption{{The correlations between various task diversity metrics and the normalized returns in {} tasks.\n  Bolded text means highest mean.}}\n".format(
        control_mode
    )
    latex_content_to_write += (
        "  \\label{{tab:diversity_return_correlation_{}}}\n".format(control_mode)
    )
    latex_content_to_write += "  \\begin{center}\n"
    latex_content_to_write += "  \\begin{tabular}{lcccc}\n"
    latex_content_to_write += "    {}\\\\\n".format(
        " ".join(["& {}".format(map_env[env]) for env in envs[idx_i]])
    )

    for row_i, row_name in enumerate(["Pearson", "Spearman", "Kendall"]):
        latex_content_to_write += "    \\hline\n"
        latex_content_to_write += (
            "    \\multicolumn{{5}}{{c}}{{\\textbf{{{}}}}}\\\\\n".format(row_name)
        )
        latex_content_to_write += "    \\hline\n"

        means = []
        stds = []
        for env_name in envs[idx_i]:
            means.append([])
            stds.append([])
            for div_i, div_name in enumerate(diversity_names):
                correlations = correlation_per_env[(env_name, control_mode)][row_i][
                    div_i
                ]
                means[-1].append(np.mean(correlations))
                stds[-1].append(np.std(correlations))
        best_idxes = np.argmax(means, axis=-1)

        for div_i, div_name in enumerate(diversity_names):
            latex_content_to_write += f"    {div_name}"

            for env_i, env_name in enumerate(envs[idx_i]):
                if div_i == best_idxes[env_i]:
                    latex_content_to_write += (
                        " & $\mathbf{{{:.3f} \\pm {:.3f}}}$".format(
                            means[env_i][div_i], stds[env_i][div_i]
                        )
                    )
                else:
                    latex_content_to_write += " & ${:.3f} \\pm {:.3f}$".format(
                        means[env_i][div_i], stds[env_i][div_i]
                    )
            latex_content_to_write += "\\\\\n"

    latex_content_to_write += "  \\end{tabular}\n"
    latex_content_to_write += "  \\end{center}\n"
    latex_content_to_write += "\\end{table}\n"

    print(latex_content_to_write)
