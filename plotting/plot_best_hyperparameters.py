from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager

import _pickle as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

from jaxl.constants import *
from plot_utils import set_size, pgf_with_latex


doc_width_pt = 452.9679
top_k = 10
cc = True

experiment_name = "single_hyperparameter_robustness"
experiment_dir = (
    f"/Users/chanb/research/personal/jaxl/jaxl/logs/dmc/cheetah/{experiment_name}"
)
save_path = f"./results-{experiment_name}"
os.makedirs(save_path, exist_ok=True)

assert os.path.isdir(experiment_dir), f"{experiment_dir} is not a directory"


result_name = f"top_{top_k}-returns"
if os.path.isfile(f"{save_path}/{result_name}.pkl"):
    episodic_returns_per_variant = pickle.load(
        open(f"{save_path}/{result_name}.pkl", "rb")
    )
else:
    episodic_returns_per_variant = {}
    variant_i = 0
    for agent_path, _, filenames in os.walk(experiment_dir):
        for filename in filenames:
            if filename != "config.json":
                continue

            variant_i += 1
            if variant_i % 10 == 0:
                print(f"Processed {variant_i} variants")

            variant_name = os.path.basename(os.path.dirname(agent_path))

            checkpoint_manager = CheckpointManager(
                os.path.join(agent_path, "models"),
                PyTreeCheckpointer(),
            )

            checkpoint = checkpoint_manager.restore(checkpoint_manager.latest())
            
            episodic_returns_per_variant[variant_name] = checkpoint[CONST_AUX][CONST_EPISODIC_RETURNS]

    with open(f"{save_path}/{result_name}.pkl", "wb") as f:
        pickle.dump(episodic_returns_per_variant, f)


fig, ax = plt.subplots(1, 1, figsize=set_size(doc_width_pt, 0.49, (1, 1)))

aucs = []
for variant_name, returns in episodic_returns_per_variant.items():
    aucs.append(np.sum(returns))
aucs = np.array(aucs)

top_k_idxes = np.argsort(aucs)[-top_k:]

variant_names = np.array(list(episodic_returns_per_variant.keys()))

for idx, (variant_name, returns) in enumerate(episodic_returns_per_variant.items()):
    if idx not in top_k_idxes:
        continue

    num_episodes = range(len(returns))
    ax.plot(num_episodes, np.cumsum(returns), marker="x", label=variant_name)

ax.set_ylabel("Cumulative Return")
ax.set_xlabel("Training Episode")
ax.legend()

fig.tight_layout()
fig.savefig(
    f"{save_path}/{result_name}.pdf", format="pdf", bbox_inches="tight", dpi=600
)
