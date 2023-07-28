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
experiment_name = "bc_amount_data"
save_path = f"./results-{experiment_name}"
experiment_dir = f"./logs/bc_amount_data/cheetah_continuous"
reference_agent_path = "../expert_policies/cheetah_continuous"

num_evaluation_episodes = 10
env_seed = 9999
record_video = False

assert os.path.isdir(experiment_dir), f"{experiment_dir} is not a directory"

os.makedirs(save_path, exist_ok=True)
if os.path.isdir(f"{save_path}/returns.pkl"):
    (result_per_variant, env_configs) = pickle.load(
        open(f"{save_path}/returns.pkl", "rb")
    )
else:
    result_per_variant = {}

    env_config_path = os.path.join(reference_agent_path, "env_config.pkl")
    env_config = None
    if os.path.isfile(env_config_path):
        env_config = pickle.load(open(env_config_path, "rb"))

    num_variants = len(os.listdir(experiment_dir))
    for variant_i, variant_name in enumerate(os.listdir(experiment_dir)):
        print(f"Processing {variant_i + 1} / {num_variants} variants")
        variant_path = os.path.join(experiment_dir, variant_name)
        episodic_returns_per_variant = {}
        entropies = {}
        for agent_path, _, filenames in os.walk(variant_path):
            for filename in filenames:
                if filename != "config.json":
                    continue

                env, policy = get_evaluation_components(
                    agent_path, use_default=True, ref_agent_path=reference_agent_path
                )

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
                    env.close()

        result_per_variant[variant_name] = entropies
        result_per_variant[variant_name] = episodic_returns_per_variant

    with open(f"{save_path}/returns.pkl", "wb") as f:
        pickle.dump((result_per_variant, env_config), f)

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


fig.supxlabel("Iterations")
fig.savefig(f"{save_path}/returns.pdf", format="pdf", bbox_inches="tight", dpi=600)
