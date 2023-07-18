from gymnasium.experimental.wrappers import RecordVideoV0
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager

import _pickle as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

from jaxl.constants import *
from jaxl.envs.rollouts import EvaluationRollout
from jaxl.utils import RunningMeanStd
from plot_utils import set_size, pgf_with_latex


# Use the seborn style
plt.style.use("seaborn")
# But with fonts from the document body
plt.rcParams.update(pgf_with_latex)

# Using the set_size function as defined earlier
doc_width_pt = 452.9679
experiment_name = "objective_comparison"
experiment_dir = (
    f"/Users/chanb/research/personal/jaxl/jaxl/logs/dmc/cheetah/{experiment_name}"
)
save_path = f"./results-{experiment_name}"

os.makedirs(save_path, exist_ok=True)

num_evaluation_episodes = 10
env_seed = 9999
record_video = True

assert os.path.isdir(experiment_dir), f"{experiment_dir} is not a directory"

if os.path.isfile(f"{save_path}/entropies.pkl"):
    result_per_variant = pickle.load(open(f"{save_path}/entropies.pkl", "rb"))
else:
    result_per_variant = {}
    for variant_name in os.listdir(experiment_dir):
        variant_path = os.path.join(experiment_dir, variant_name)
        entropies = {}
        for agent_path, _, filenames in os.walk(variant_path):
            for filename in filenames:
                if filename != "config.json":
                    continue

                auxes = os.path.join(agent_path, "auxes")
                for checkpoint_name in os.listdir(auxes):
                    checkpoint_i = int(checkpoint_name[:-4].split("-")[-1])
                    data = pickle.load(open(os.path.join(auxes, checkpoint_name), "rb"))
                    entropies.setdefault(checkpoint_i, [])
                    entropies[checkpoint_i].append(
                        np.mean(data[CONST_POLICY][CONST_ENTROPY])
                    )
        result_per_variant[variant_name.split("-")[0]] = entropies

fig, ax = plt.subplots(1, 1, figsize=set_size(doc_width_pt, 0.49, (1, 1)))

for variant_name, returns in result_per_variant.items():
    iteration = list(returns.keys())
    means = []
    stds = []
    for val in returns.values():
        means.append(np.mean(val))
        stds.append(np.std(val))
    means = np.array(means)
    stds = np.array(stds)

    sort_idxes = np.argsort(iteration)
    iteration = np.array(iteration)
    ax.plot(iteration[sort_idxes], means[sort_idxes], marker="x", label=variant_name)
    ax.fill_between(
        iteration[sort_idxes],
        means[sort_idxes] + stds[sort_idxes],
        means[sort_idxes] - stds[sort_idxes],
        alpha=0.3,
    )

ax.set_ylabel("Policy Entropy")
ax.set_xlabel("Iterations")
ax.legend()

fig.tight_layout()
fig.savefig(
    f"{save_path}/policy_entropies.pdf", format="pdf", bbox_inches="tight", dpi=600
)

with open(f"{save_path}/entropies.pkl", "wb") as f:
    pickle.dump(result_per_variant, f)
