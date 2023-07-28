from gymnasium.experimental.wrappers import RecordVideoV0
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager
from typing import Iterable

import _pickle as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

from jaxl.constants import *
from jaxl.envs.rollouts import EvaluationRollout
from jaxl.utils import RunningMeanStd, flatten_dict
from jaxl.plot_utils import set_size, pgf_with_latex, get_evaluation_components


# Use the seborn style
plt.style.use("seaborn")
# But with fonts from the document body
plt.rcParams.update(pgf_with_latex)

# Using the set_size function as defined earlier
doc_width_pt = 452.9679

# Variants:
# experiment_name = "single_hyperparameter_robustness"
# experiment_dir = (
#     f"/Users/chanb/research/personal/jaxl/jaxl/logs/dmc/cheetah/{experiment_name}"
# )

# Pendulum continuous expert
# experiment_name = "pendulum_cont"
# experiment_dir = (
#     "/Users/chanb/research/personal/mtil_results/data/experts/pendulum/continuous/runs/0/"
# )

# Pendulum discrete expert
# experiment_name = "pendulum_disc"
# experiment_dir = (
#     "/Users/chanb/research/personal/mtil_results/data/experts/pendulum/discrete/runs/0/"
# )

# Cheetah continuous expert
# experiment_name = "cheetah_cont"
# experiment_dir = (
#     "/Users/chanb/research/personal/mtil_results/data/experts/cheetah/continuous/runs/0/"
# )

# Cheetah discrete expert
# experiment_name = "cheetah_disc"
# experiment_dir = (
#     "/Users/chanb/research/personal/mtil_results/data/experts/cheetah/discrete/runs/0/"
# )

# Walker continuous expert
# experiment_name = "walker_cont"
# experiment_dir = (
#     "/Users/chanb/research/personal/mtil_results/data/experts/walker/continuous/runs/0/"
# )

# # Walker discrete expert
experiment_name = "walker_disc"
experiment_dir = (
    "/Users/chanb/research/personal/mtil_results/data/experts/walker/discrete/runs/0/"
)

save_path = f"./results-{experiment_name}"
os.makedirs(save_path, exist_ok=True)

num_evaluation_episodes = 10
env_seed = 9999
record_video = False

assert os.path.isdir(experiment_dir), f"{experiment_dir} is not a directory"

if os.path.isfile(f"{save_path}/returns.pkl"):
    (result_per_variant, env_configs) = pickle.load(
        open(f"{save_path}/returns.pkl", "rb")
    )
else:
    result_per_variant = {}
    env_configs = {}
    for variant_i, variant_name in enumerate(os.listdir(experiment_dir)):
        if (variant_i + 1) % 10 == 0:
            print(f"Processed {variant_i + 1} variants")
        variant_path = os.path.join(experiment_dir, variant_name)
        episodic_returns_per_variant = {}
        for agent_path, _, filenames in os.walk(variant_path):
            for filename in filenames:
                if filename != "config.json":
                    continue

                env_config_path = os.path.join(agent_path, "env_config.pkl")
                env_config = None
                if os.path.isfile(env_config_path):
                    env_config = pickle.load(open(env_config_path, "rb"))

                env, policy = get_evaluation_components(agent_path)

                checkpoint_manager = CheckpointManager(
                    os.path.join(agent_path, "models"),
                    PyTreeCheckpointer(),
                )
                for checkpoint_step in checkpoint_manager.all_steps():
                    if (
                        record_video
                        and checkpoint_step == checkpoint_manager.latest_step()
                    ):
                        env = RecordVideoV0(
                            env,
                            f"{save_path}/videos/variant_{variant_name}/model_id_{checkpoint_step}",
                            disable_logger=True,
                        )
                    params = checkpoint_manager.restore(checkpoint_step)
                    model_dict = params[CONST_MODEL_DICT]
                    agent_policy_params = model_dict[CONST_MODEL][CONST_POLICY]
                    agent_obs_rms = False
                    if CONST_OBS_RMS in params:
                        agent_obs_rms = RunningMeanStd()
                        agent_obs_rms.set_state(params[CONST_OBS_RMS])

                    agent_rollout = EvaluationRollout(env, seed=env_seed)
                    agent_rollout.rollout(
                        agent_policy_params,
                        policy,
                        agent_obs_rms,
                        num_evaluation_episodes,
                        None,
                        use_tqdm=False,
                    )

                    episodic_returns_per_variant.setdefault(checkpoint_step, [])
                    episodic_returns_per_variant[checkpoint_step].append(
                        np.mean(agent_rollout.episodic_returns)
                    )
        result_per_variant[variant_name] = episodic_returns_per_variant
        env_configs[variant_name] = env_config["modified_attributes"]

    with open(f"{save_path}/returns.pkl", "wb") as f:
        pickle.dump((result_per_variant, env_configs), f)

# Plot main return
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

ax.set_ylabel("Expected Return")
ax.set_xlabel("Iterations")
ax.legend()

fig.tight_layout()
fig.savefig(f"{save_path}/returns.pdf", format="pdf", bbox_inches="tight", dpi=600)

# Plot return based on environmental parameter
max_return_means = []
max_return_stds = []
modified_attributes = {}
for variant_name in env_configs:
    env_config = env_configs[variant_name]
    returns = result_per_variant[variant_name]

    max_return_mean = -np.inf
    max_return_std = 0.0
    for val in returns.values():
        mean = np.mean(val)
        std = np.std(val)
        if max_return_mean < mean:
            max_return_mean = mean
            max_return_std = std

    max_return_means.append(max_return_mean)
    max_return_stds.append(max_return_std)

    for attr_name, attr_val in flatten_dict(env_config):
        if isinstance(attr_val, Iterable) and len(attr_val) > 1:
            for val_i in range(len(attr_val)):
                modified_attributes.setdefault(f"{attr_name}.{val_i}", [])
                modified_attributes[f"{attr_name}.{val_i}"].append(attr_val[val_i])
        elif isinstance(attr_val, Iterable):
            modified_attributes.setdefault(attr_name, [])
            modified_attributes[attr_name].append(attr_val[0])
        else:
            modified_attributes.setdefault(attr_name, [])
            modified_attributes[attr_name].append(attr_val)


max_return_means = np.array(max_return_means)
max_return_stds = np.array(max_return_stds)

for attr_name, attr_vals in modified_attributes.items():
    modified_attributes[attr_name] = np.array(attr_vals)

from matplotlib.ticker import FormatStrFormatter

fraction = 0.95
num_cols = 2
num_rows = int(np.ceil(len(modified_attributes) / 2))

fig, axes = plt.subplots(
    num_rows, num_cols, figsize=set_size(doc_width_pt, fraction, (num_rows, num_cols))
)
fig.supylabel("Expected Return")

filter_size = 21
boundary = filter_size // 2
filter = np.ones(filter_size) / filter_size
for attr_i, (attr_name, attr_vals) in enumerate(modified_attributes.items()):
    sort_idxes = np.argsort(attr_vals)

    if num_rows == num_cols == 1:
        ax = axes
    elif num_rows == 1 or num_cols == 1:
        ax = axes[attr_i]
    else:
        ax = axes[attr_i // num_cols, attr_i % num_cols]

    means = max_return_means[sort_idxes]

    means_smoothed = np.convolve(means, filter, "valid")

    x_vals = attr_vals[sort_idxes]
    stds = max_return_stds[sort_idxes]
    if boundary > 0:
        x_vals = x_vals[boundary:-boundary]
        stds = stds[boundary:-boundary]
    ax.plot(x_vals, means_smoothed, marker="x")
    ax.fill_between(
        x_vals,
        means_smoothed + stds,
        means_smoothed - stds,
        alpha=0.3,
    )

    ax.set_xlabel(f"{attr_name}")

    if x_vals.max() - x_vals.min() <= 1.0:
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))

num_blanks = num_cols - len(modified_attributes) % num_cols
if num_cols > num_blanks > 0:
    for ax_i in range(1, num_blanks + 1):
        axes[-1, -ax_i].axis("off")

fig.tight_layout()
fig.savefig(
    f"{save_path}/returns-variants.pdf", format="pdf", bbox_inches="tight", dpi=600
)
