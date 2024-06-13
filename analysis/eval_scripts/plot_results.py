import _pickle as pickle
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from jaxl.plot_utils import set_size


def main(args):
    load_dir = args.load_dir

    random_checkpoint = os.listdir(load_dir)[0].split(".pkl")[0]
    learner_name = random_checkpoint.split("-checkpoint_idx_")[0]

    checkpoints = sorted(
        [
            int(checkpoint.split(".pkl")[0].split("-checkpoint_idx_")[-1])
            for checkpoint in os.listdir(load_dir)
        ]
    )

    results = []
    for checkpoint_i in checkpoints:
        checkpoint_path = os.path.join(
            load_dir, f"{learner_name}-checkpoint_idx_{checkpoint_i}.pkl"
        )
        results.append(pickle.load(open(checkpoint_path, "rb")))

    results = np.array(results)
    fig, ax = plt.subplots(1, 1, figsize=set_size(1, 1))

    ret_mean = np.mean(results, axis=-1)
    ret_std = np.std(results, axis=-1) / np.sqrt(results.shape[-1])

    ax.plot(checkpoints, ret_mean, label="$\pi_{{learner}}$")
    ax.fill_between(checkpoints, ret_mean + ret_std, ret_mean - ret_std, alpha=0.3)
    ax.set_title("Evaluation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Return")

    baseline_mean = 203.98
    baseline_std = 147.888 / np.sqrt(50)

    ax.axhline(baseline_mean, label="$\pi_{{ref}}$", color="black", linestyle="--")
    ax.axhspan(
        baseline_mean + baseline_std,
        baseline_mean - baseline_std,
        facecolor="black",
        alpha=0.3,
    )
    fig.legend()
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_dir",
        type=str,
        required=True,
        help="The directory to load the result from",
    )
    args = parser.parse_args()
    main(args)
