from gymnasium.experimental.wrappers import RecordVideoV0
from itertools import product
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager

import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os

from jaxl.buffers import get_buffer
from jaxl.constants import *
from jaxl.envs.rollouts import EvaluationRollout
from jaxl.utils import RunningMeanStd, parse_dict
from jaxl.plot_utils import set_size, pgf_with_latex, get_evaluation_components


# Use the seborn style
plt.style.use("seaborn")
# But with fonts from the document body
plt.rcParams.update(pgf_with_latex)

# Using the set_size function as defined earlier
doc_width_pt = 452.9679

task = "pendulum"
control_mode = "continuous"

save_path = f"./results_policy_robustness-pendulum"
os.makedirs(save_path, exist_ok=True)

rollout_seed = 1000
env_seed_range = 1000

default_agent_path = "/Users/chanb/research/personal/jaxl/scripts/mtil/local/policy_robustness_pendulum/logs/pendulum-default"
variant_agent_path = "/Users/chanb/research/personal/jaxl/scripts/mtil/local/policy_robustness_pendulum/logs/pendulum-env_seed_769"

env_seed = 50
# env_seed = 540
# env_seed = 35
# env_seed = 105
# env_seed = 769
num_evaluation_episodes = 50
max_episode_length = 200

all_res = {}

def get_params(agent_path):
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
    return agent_policy_params, agent_obs_rms


def get_data(agent_path, target_agent_path, env_variant_seed):
    all_env_configs = {}
    episodic_returns_per_seed = {}

    env, policy = (
        get_evaluation_components(target_agent_path, env_variant_seed, False)
    )
    
    policy_params, policy_obs_rms = get_params(agent_path)
    target_params, target_obs_rms = get_params(target_agent_path)

    buffer_size = max_episode_length * num_evaluation_episodes

    buffer_config = {
        "buffer_type": "default",
        "buffer_size": buffer_size
    }

    buffer = get_buffer(
        parse_dict(buffer_config),
        0,
        env,
        (1,),
    )

    agent_rollout = EvaluationRollout(env, seed=rollout_seed)
    agent_rollout.rollout(
        target_params,
        policy,
        target_obs_rms,
        num_evaluation_episodes,
        buffer,
        use_tqdm=False,
    )

    # TODO: Get approx KL

    env.close()
    return episodic_returns_per_seed, all_env_configs

if os.path.isfile(f"{save_path}/{task}_{control_mode}-approx_kl_{seed}.pkl"):
    results = pickle.load(
        open(f"{save_path}/{task}_{control_mode}-approx_kl_{seed}.pkl", "rb")
    )
else:
    results = get_data(
        default_agent_path,
        variant_agent_path,
        env_seed
    )

    with open(f"{save_path}/{task}_{control_mode}-approx_kl_{seed}.pkl", "wb") as f:
        pickle.dump(results, f)

# Plot main return
num_rows = 1
num_cols = 1
fig, ax = plt.subplots(
    num_rows,
    num_cols,
    figsize=set_size(doc_width_pt, 0.95, (num_rows, num_cols)),
    layout="constrained",
)

print(all_env_configs.items())
all_env_seeds = [None, 769, *env_seeds]
seeds_to_plot = np.array(all_env_seeds)

torques = np.array(
    [
        2.0 if env_seed is None else all_env_configs[env_seed]["max_torque"]
        for env_seed in all_env_seeds
    ]
)
sort_idxes = np.argsort(torques)
linestyles = ["--", "-."]

for idx, returns_per_seed in enumerate([default_episodic_returns, variant_episodic_returns]):
    means = []
    stds = []
    for val in returns_per_seed.values():
        means.append(np.mean(val))
        stds.append(np.std(val))
    means = np.array(means)
    stds = np.array(stds)

    ax.axvline(torques[idx], linestyle="--", linewidth=0.5, color="black", alpha=0.5)
    ax.plot(
        torques[sort_idxes],
        means[sort_idxes],
        marker="^",
        ms=3.0,
        linewidth=0.75,
        label="torque @ {:.2f}".format(torques[idx])
    )
    ax.fill_between(
        torques[sort_idxes],
        means[sort_idxes] + stds[sort_idxes],
        means[sort_idxes] - stds[sort_idxes],
        alpha=0.3,
    )
ax.legend()
ax.set_xlabel("Maximum Torque")
ax.set_ylabel("Expected Return")

fig.savefig(
    f"{save_path}/policy_robustness.pdf", format="pdf", bbox_inches="tight", dpi=600
)
