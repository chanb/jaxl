from gymnasium.experimental.wrappers import RecordVideoV0
from itertools import product
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager

import _pickle as pickle
import json
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

# Using the set_size function as defined earlier
doc_width_pt = 452.9679

task = "cartpole"
control_mode = "continuous"

save_path = f"./results_policy_robustness-{task}"
os.makedirs(save_path, exist_ok=True)

seed = 999

rollout_seed = 1000
env_seed_range = 1000
num_envs_to_test = 8

default_agent_path = "/Users/chanb/research/personal/jaxl/scripts/mtil/local/policy_robustness-example/logs/cartpole_continuous/ppo-08-01-23_09_01_54-2ae9e079-f3d6-4def-be16-204ce4c44442"
variant_agent_path = "/Users/chanb/research/personal/jaxl/scripts/mtil/local/policy_robustness-example/logs/cartpole_continuous/ppo-08-01-23_10_07_12-308c3889-7e91-406b-9b1d-a9cfd723defb"

variant_env_seed = json.load(
    open(os.path.join(variant_agent_path, "config.json"), "r")
)["learner_config"]["env_config"]["env_kwargs"]["seed"]

num_evaluation_episodes = 50
record_video = False

assert num_envs_to_test > 0, f"num_envs_to_test needs to be at least 1"

all_res = {}

rng = np.random.RandomState(seed)

# Ensure it is not covering the same seeds in the agents
env_seeds = rng.randint(10, env_seed_range, size=num_envs_to_test)
while len(np.unique(env_seeds)) != num_envs_to_test:
    env_seeds = rng.randint(10, env_seed_range, size=num_envs_to_test)


def get_data(agent_path, all_env_seeds):
    all_env_configs = {}
    episodic_returns_per_seed = {}

    checkpoint_manager = CheckpointManager(
        os.path.join(agent_path, "models"),
        PyTreeCheckpointer(),
    )

    for env_seed in all_env_seeds:
        env, policy = (
            get_evaluation_components(agent_path, None, True)
            if env_seed is None
            else get_evaluation_components(agent_path, env_seed, False)
        )
        if record_video:
            env = RecordVideoV0(
                env,
                f"{save_path}/videos/{task}-{control_mode}/env_seed_{env_seed}",
                disable_logger=True,
            )
        all_env_configs[env_seed] = env.get_config()["modified_attributes"]
        print(
            env_seed,
            env.get_config()["modified_attributes"],
        )
        params = checkpoint_manager.restore(checkpoint_manager.latest_step())
        model_dict = params[CONST_MODEL_DICT]
        agent_policy_params = model_dict[CONST_MODEL][CONST_POLICY]
        agent_obs_rms = False
        if CONST_OBS_RMS in params:
            agent_obs_rms = RunningMeanStd()
            agent_obs_rms.set_state(params[CONST_OBS_RMS])

        agent_rollout = EvaluationRollout(env, seed=rollout_seed)
        agent_rollout.rollout(
            agent_policy_params,
            policy,
            agent_obs_rms,
            num_evaluation_episodes,
            None,
            use_tqdm=False,
        )

        episodic_returns_per_seed.setdefault(env_seed, [])
        episodic_returns_per_seed[env_seed].append(agent_rollout.episodic_returns)
        env.close()
    return episodic_returns_per_seed, all_env_configs


if os.path.isfile(f"{save_path}/{task}_{control_mode}-returns_{seed}.pkl"):
    (
        default_episodic_returns,
        variant_episodic_returns,
        variant_env_seed,
        env_seeds,
        all_env_configs,
    ) = pickle.load(open(f"{save_path}/{task}_{control_mode}-returns_{seed}.pkl", "rb"))
else:
    all_env_seeds = [None, variant_env_seed, *env_seeds]

    default_episodic_returns, all_env_configs = get_data(
        default_agent_path, all_env_seeds
    )
    variant_episodic_returns, _ = get_data(variant_agent_path, all_env_seeds)

    with open(f"{save_path}/{task}_{control_mode}-returns_{seed}.pkl", "wb") as f:
        pickle.dump(
            (
                default_episodic_returns,
                variant_episodic_returns,
                variant_env_seed,
                env_seeds,
                all_env_configs,
            ),
            f,
        )

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
all_env_seeds = [None, variant_env_seed, *env_seeds]
seeds_to_plot = np.array(all_env_seeds)

stiffness = np.array(
    [
        0.0
        if env_seed is None
        else all_env_configs[env_seed]["joint"]["hinge_1"]["stiffness"][0]
        for env_seed in all_env_seeds
    ]
)
sort_idxes = np.argsort(stiffness)
linestyles = ["--", "-."]

for idx, returns_per_seed in enumerate(
    [default_episodic_returns, variant_episodic_returns]
):
    means = []
    stds = []
    for val in returns_per_seed.values():
        means.append(np.mean(val))
        stds.append(np.std(val))
    means = np.array(means)
    stds = np.array(stds)

    ax.axvline(stiffness[idx], linestyle="--", linewidth=0.5, color="black", alpha=0.5)
    ax.plot(
        stiffness[sort_idxes],
        means[sort_idxes],
        marker="^",
        ms=3.0,
        linewidth=0.75,
        label="stiffness @ {:.2f}".format(stiffness[idx]),
    )
    ax.fill_between(
        stiffness[sort_idxes],
        means[sort_idxes] + stds[sort_idxes],
        means[sort_idxes] - stds[sort_idxes],
        alpha=0.3,
    )
ax.legend()
ax.set_xlabel("Joint Stiffness")
ax.set_ylabel("Expected Return")

fig.savefig(
    f"{save_path}/policy_robustness.pdf", format="pdf", bbox_inches="tight", dpi=600
)
