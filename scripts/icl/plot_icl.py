import _pickle as pickle
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from jaxl.plot_utils import set_size, pgf_with_latex



# Use the seborn style
sns.set_style("darkgrid")
sns.set_palette("colorblind")

# But with fonts from the document body
# plt.rcParams.update(pgf_with_latex)

# Using the set_size function as defined earlier
doc_width_pt = 1000.0

save_path = "./bursty_1.0-pixel_noise_0.1-accuracies.pkl"

data = pickle.load(open(save_path, "rb"))

checkpoint_steps = data["checkpoint_steps"]
accuracies = data["accuracies"]


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
    ax.plot(checkpoint_steps, eval_accs)
    ax.set_title(eval_name)

fig.supxlabel("Number of updates")
fig.supylabel("Accuracy")

fig.savefig(
    "{}.pdf".format(save_path[:-4]), format="pdf", bbox_inches="tight", dpi=600
)
