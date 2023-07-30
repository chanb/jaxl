from gymnasium.experimental.wrappers import RecordVideoV0
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager

import _pickle as pickle
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
experiment_name = "objective_comparison"
experiment_dir = f"./logs"
save_path = f"./results-{experiment_name}"
os.makedirs(save_path, exist_ok=True)

num_evaluation_episodes = 10
env_seed = 9999
record_video = False

assert os.path.isdir(experiment_dir), f"{experiment_dir} is not a directory"

if os.path.isdir(f"{save_path}"):
    (result_per_variant, env_configs) = pickle.load(
        open(f"{save_path}/returns.pkl", "rb")
    )
    entropy_per_variant = pickle.load(open(f"{save_path}/entropies.pkl", "rb"))
else:
    result_per_variant = {}
    env_configs = {}
    entropy_per_variant = {}
    for variant_i, variant_name in enumerate(os.listdir(experiment_dir)):
        if (variant_i + 1) % 10 == 0:
            print(f"Processed {variant_i + 1} variants")
        variant_path = os.path.join(experiment_dir, variant_name)
        episodic_returns_per_variant = {}
        entropies = {}
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

                auxes = os.path.join(agent_path, "auxes")
                for checkpoint_name in os.listdir(auxes):
                    checkpoint_i = int(checkpoint_name[:-4].split("-")[-1])
                    data = pickle.load(open(os.path.join(auxes, checkpoint_name), "rb"))
                    entropies.setdefault(checkpoint_i, [])
                    entropies[checkpoint_i].append(
                        np.mean(data[CONST_POLICY][CONST_ENTROPY])
                    )

        result_per_variant[variant_name] = entropies
        result_per_variant[variant_name] = episodic_returns_per_variant
        env_configs[variant_name] = env_config["modified_attributes"]

    with open(f"{save_path}/returns.pkl", "wb") as f:
        pickle.dump((result_per_variant, env_configs), f)
    with open(f"{save_path}/entropies.pkl", "wb") as f:
        pickle.dump(entropy_per_variant, f)

# Plot main return
fig, axes = plt.subplots(
    1, 2, figsize=set_size(doc_width_pt, 0.95, (1, 2)), layout="constrained"
)

ax = axes[0]
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
ax.legend()


ax = axes[1]
for variant_name, entropies in entropy_per_variant.items():
    iteration = list(entropies.keys())
    means = []
    stds = []
    for val in entropies.values():
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

fig.supxlabel("Iterations")
fig.savefig(
    f"{save_path}/returns_and_entropy.pdf", format="pdf", bbox_inches="tight", dpi=600
)
