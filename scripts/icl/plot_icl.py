import _pickle as pickle
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from jaxl.plot_utils import set_size, pgf_with_latex



# Use the seborn style
sns.set_style("darkgrid")
sns.set_palette("colorblind")

# But with fonts from the document body
# plt.rcParams.update(pgf_with_latex)

# Using the set_size function as defined earlier
doc_width_pt = 1000.0

exp_name = "pixel_noise_0.1"
save_path = "./{}.pdf".format(exp_name)
load_path = "./plot_data"


result_paths = os.listdir(load_path)
for result_i, result_path in enumerate(result_paths):
    curr_path = os.path.join(load_path, result_path)
    data = pickle.load(open(curr_path, "rb"))
    variant_name = result_path[:-len("{}-accuracies.pkl".format(exp_name))]

    checkpoint_steps = data["checkpoint_steps"]
    accuracies = data["accuracies"]

    if result_i == 0:
        num_rows = math.ceil(len(accuracies) / 3)
        num_cols = 3
        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=set_size(doc_width_pt, 0.95, (num_rows, num_cols)),
            layout="constrained",
        )

    for eval_i, (eval_name, eval_accs) in enumerate(accuracies.items()):
        ax = axes[eval_i // num_cols, eval_i % num_cols]
        ax.plot(checkpoint_steps, eval_accs, label=variant_name if eval_i == 0 else "")

        if result_i == 0:
            ax.set_title(eval_name)
            ax.set_ylim(-1.0, 101.0)

fig.supxlabel("Number of updates")
fig.supylabel("Accuracy")
fig.legend(
    bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
    loc="lower center",
    ncols=len(result_paths),
    borderaxespad=0.0,
    frameon=True,
    fontsize="8",
)

fig.savefig(
    save_path, format="pdf", bbox_inches="tight", dpi=600
)
