from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager

import _pickle as pickle
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

from jaxl.constants import *
from plot_utils import set_size, pgf_with_latex


doc_width_pt = 452.9679
top_k = 5
smoothing = 20
cc = True

experiment_name = "search_expert"
experiment_dir = "/Users/chanb/research/personal/mtil_results/data/search_expert/"
tasks = ["pendulum", "cheetah", "walker"]

assert os.path.isdir(experiment_dir), f"{experiment_dir} is not a directory"

# Use the seborn style
plt.style.use("seaborn")
# But with fonts from the document body
plt.rcParams.update(pgf_with_latex)
fig, axes = plt.subplots(
    3, 2, figsize=set_size(doc_width_pt, 0.95, (3, 2)), layout="constrained"
)

result_name = f"top_{top_k}-returns"
control_results = {}
for row_i, task in enumerate(tasks):
    for col_i, control_mode in enumerate(["discrete", "continuous"]):
        save_path = f"./results-{experiment_name}"
        os.makedirs(save_path, exist_ok=True)

        agent_paths = []
        if os.path.isfile(f"{save_path}/{result_name}-{task}-{control_mode}.pkl"):
            (episodic_returns_per_variant, agent_paths) = pickle.load(
                open(f"{save_path}/{result_name}-{task}-{control_mode}.pkl", "rb")
            )
        else:
            episodic_returns_per_variant = {}
            variant_i = 0
            for agent_path, _, filenames in os.walk(
                os.path.join(experiment_dir, task, control_mode)
            ):
                for filename in filenames:
                    if filename != "config.json":
                        continue

                    variant_i += 1
                    if variant_i % 10 == 0:
                        print(f"Processed {variant_i} variants")

                    variant_name = os.path.basename(os.path.dirname(agent_path))
                    agent_paths.append(os.path.join(agent_path, "config.json"))

                    checkpoint_manager = CheckpointManager(
                        os.path.join(agent_path, "models"),
                        PyTreeCheckpointer(),
                    )

                    checkpoint = checkpoint_manager.restore(
                        checkpoint_manager.latest_step()
                    )

                    episodic_returns_per_variant[variant_name] = checkpoint[CONST_AUX][
                        CONST_EPISODIC_RETURNS
                    ][:-1]

            with open(
                f"{save_path}/{result_name}-{task}-{control_mode}.pkl", "wb"
            ) as f:
                pickle.dump((episodic_returns_per_variant, agent_paths), f)

        control_results[(task, control_mode)] = (
            episodic_returns_per_variant,
            agent_paths,
        )

        aucs = []
        for variant_name, returns in episodic_returns_per_variant.items():
            aucs.append(np.sum(returns))
        aucs = np.array(aucs)
        agent_paths = np.array(agent_paths)

        top_k_idxes = np.argsort(aucs)[-top_k:]

        variant_names = np.array(list(episodic_returns_per_variant.keys()))
        print(agent_paths[top_k_idxes][::-1])

        ax = axes[row_i, col_i]
        res = list(episodic_returns_per_variant.items())
        for idx in top_k_idxes:
            (variant_name, returns) = res[idx]

            num_episodes = np.arange(len(returns))
            cumsum_returns = np.cumsum(returns)
            last_t_episodic_returns = cumsum_returns - np.concatenate(
                (np.zeros(smoothing), cumsum_returns[:-smoothing])
            )
            avg_episodic_returns = last_t_episodic_returns / np.concatenate(
                (
                    np.arange(1, smoothing + 1),
                    np.ones(len(returns) - smoothing) * smoothing,
                )
            )

            ax.plot(
                num_episodes,
                avg_episodic_returns,
                label=variant_name.split("-")[1]
                if idx != top_k_idxes[-1]
                else "{} - best".format(variant_name.split("-")[1]),
                linewidth=1.0,
                alpha=0.7,
                linestyle="-" if idx != top_k_idxes[-1] else "--",
            )
        ax.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncols=math.ceil(top_k / 2),
            mode="expand",
            borderaxespad=0.0,
            frameon=True,
        )

        if col_i == 0:
            ax.set_ylabel(task)
        if row_i == len(tasks) - 1:
            ax.set_xlabel(control_mode)

fig.supylabel("Expected Return")
fig.supxlabel("Training Episode")
fig.savefig(
    f"{save_path}/{result_name}.pdf", format="pdf", bbox_inches="tight", dpi=600
)
