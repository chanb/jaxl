from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager

import _pickle as pickle
import gzip
import math
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import os

from jaxl.constants import *
from jaxl.envs.rollouts import EvaluationRollout
from jaxl.utils import RunningMeanStd
from jaxl.plot_utils import set_size, pgf_with_latex, get_evaluation_components


# Use the seborn style
plt.style.use("seaborn")
# But with fonts from the document body
plt.rcParams.update(pgf_with_latex)
colors = pl.cm.jet(np.linspace(0, 1, 10))

# Using the set_size function as defined earlier
doc_width_pt = 452.9679

experiment_name = "bc_subsampling"
save_path = f"./{experiment_name}-results"
experiment_dir = f"./logs/{experiment_name}"

num_evaluation_episodes = 30
env_seed = 9999
record_video = False

assert os.path.isdir(experiment_dir), f"{experiment_dir} is not a directory"

os.makedirs(save_path, exist_ok=True)


if os.path.isfile(f"{save_path}/results.pkl"):
    (results, env_configs) = pickle.load(open(f"{save_path}/results.pkl", "rb"))
else:
    results = {}
    env_configs = {}
    for agent_path, _, filenames in os.walk(experiment_dir):
        for filename in filenames:
            if filename != "config.json":
                continue

            print("Processing {}".format(agent_path))
            variant_name = os.path.basename(os.path.dirname(agent_path))
            variant_info = variant_name.split("-")
            (env_name, control_mode) = variant_info[0].split("_")
            buffer_size = int(variant_info[1].split("size_")[-1])
            subsampling = int(
                os.path.basename(agent_path).split("-")[0].split("subsampling_")[-1]
            )

            reference_agent_path = f"../expert_policies/{env_name}_{control_mode}"

            results.setdefault((env_name, control_mode), {})
            results[(env_name, control_mode)].setdefault(subsampling, [])
            env_config_path = os.path.join(reference_agent_path, "env_config.pkl")
            env_config = None
            if os.path.isfile(env_config_path):
                env_config = pickle.load(open(env_config_path, "rb"))
            env_configs[(env_name, control_mode)] = env_config

            env, policy = get_evaluation_components(
                agent_path, ref_agent_path=reference_agent_path
            )
            checkpoint_manager = CheckpointManager(
                os.path.join(agent_path, "models"),
                PyTreeCheckpointer(),
            )
            params = checkpoint_manager.restore(checkpoint_manager.latest_step())
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
            results[(env_name, control_mode)][subsampling].append(
                (buffer_size, np.mean(agent_rollout.episodic_returns))
            )
            env.close()

    with open(f"{save_path}/results.pkl", "wb") as f:
        pickle.dump((results, env_configs), f)

expert_returns = {}
subsampling_map = {
    "frozenlake": 200,
    "pendulum": 200,
    "cartpole": 1000,
    "cheetah": 1000,
    "walker": 1000,
}
for env_name, control_mode in results:
    buffer_path = "./logs/demonstrations/expert_buffer-default-{}_{}-num_samples_10000-subsampling_{}.gzip".format(
        env_name, control_mode, subsampling_map[env_name]
    )
    with gzip.open(buffer_path, "rb") as f:
        buffer_dict = pickle.load(f)
        expert_returns[(env_name, control_mode)] = np.sum(
            buffer_dict["rewards"]
        ) / np.sum(buffer_dict["dones"])

num_envs = len(results)
num_rows = math.ceil(num_envs / 2)
num_cols = 2

# Plot main return
fig, axes = plt.subplots(
    num_rows,
    num_cols,
    figsize=set_size(doc_width_pt, 0.95, (num_rows, num_cols)),
    layout="constrained",
)

map_env = {
    "frozenlake": "frozen lake",
    "cheetah": "cheetah",
    "walker": "walker",
    "cartpole": "cartpole",
    "pendulum": "pendulum",
}
map_control = {
    "discrete": "discrete",
    "continuous": "continuous",
}


env_names = [
    ("frozenlake", "discrete"),
    ("cartpole", "continuous"),
    # ("pendulum", "discrete"),
    # ("pendulum", "continuous"),
    ("cheetah", "discrete"),
    ("cheetah", "continuous"),
    ("walker", "discrete"),
    ("walker", "continuous"),
]

# Plot main return
for ax_i, env_name in enumerate(env_names):
    result = results[env_name]
    if axes.ndim == 2:
        ax = axes[ax_i // num_cols, ax_i % num_cols]
    else:
        ax = axes[ax_i]

    ax.axhline(
        expert_returns[env_name],
        label="expert" if ax_i == 0 else "",
        color="black",
        linestyle="--",
    )

    for plot_i, (subsampling, res) in enumerate(result.items()):
        np_res = np.array(res)
        buffer_sizes, returns = np_res[:, 0], np_res[:, 1]
        unique_buffer_sizes = np.unique(buffer_sizes)

        means = []
        stds = []

        for buffer_size in unique_buffer_sizes:
            means.append(np.mean(returns[buffer_sizes == buffer_size]))
            stds.append(np.std(returns[buffer_sizes == buffer_size]))

        means = np.array(means)
        stds = np.array(stds)

        label = ""
        if ax_i == 0:
            if subsampling == max(result.keys()):
                label = "full"
            else:
                label = subsampling
        ax.plot(unique_buffer_sizes, means, marker="x", label=label)
        ax.fill_between(
            unique_buffer_sizes,
            means + stds,
            means - stds,
            alpha=0.3,
        )

        ax.set_xlabel("{} {}".format(map_env[env_name[0]], map_control[env_name[1]]))
        ax.legend()

fig.supylabel("Expected Return")
fig.supxlabel("Amount of Transitions")
fig.savefig(f"{save_path}/returns.pdf", format="pdf", bbox_inches="tight", dpi=600)
