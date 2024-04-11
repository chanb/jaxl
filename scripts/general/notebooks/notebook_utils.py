import _pickle as pickle
import json
import os

from gymnasium.experimental.wrappers import RecordVideoV0
from orbax.checkpoint import PyTreeCheckpointer

from jaxl.constants import *
from jaxl.envs import get_environment
from jaxl.envs.rollouts import EvaluationRollout
from jaxl.models import get_model, get_policy, policy_output_dim, MultitaskPolicy
from jaxl.utils import get_dict_value, parse_dict, set_dict_value


def get_config(agent_path):
    agent_config_path = os.path.join(agent_path, "config.json")
    with open(agent_config_path, "r") as f:
        agent_config_dict = json.load(f)
        # TODO: Add code to deal with BC models
        agent_config_dict["learner_config"]["env_config"]["env_kwargs"][
            "render_mode"
        ] = "rgb_array"
        if "policy_distribution" not in agent_config_dict["learner_config"]:
            agent_config_dict["learner_config"][
                "policy_distribution"
            ] = CONST_DETERMINISTIC
        set_dict_value(agent_config_dict, "vmap_all", False)
        (multitask, num_models) = get_dict_value(agent_config_dict, "num_models")
        agent_config = parse_dict(agent_config_dict)
    return agent_config, {
        "multitask": multitask,
        "num_models": num_models,
    }


def get_agent(env, agent_config, aux):
    input_dim = env.observation_space.shape
    output_dim = policy_output_dim(env.act_dim, agent_config.learner_config)
    model = get_model(
        input_dim,
        output_dim,
        getattr(agent_config.model_config, "policy", agent_config.model_config),
    )
    policy = get_policy(model, agent_config.learner_config)
    if aux["multitask"]:
        policy = MultitaskPolicy(policy, model, aux["num_models"])

    return model, policy


def restore_agent_state(model_dict_path):
    checkpointer = PyTreeCheckpointer()
    model_dict = checkpointer.restore(model_dict_path)
    policy_params = model_dict[CONST_MODEL][CONST_POLICY]
    with open(os.path.join(model_dict_path, "learner_dict.pkl"), "rb") as f:
        learner_dict = pickle.load(f)
        obs_rms = learner_dict[CONST_OBS_RMS]

    return policy_params, obs_rms


def get_episodic_returns_per_checkpoint(
    agent_path,
    variant_name,
    env_seed,
    num_episodes,
    buffer=None,
    record_video=False,
    use_tqdm=True,
):
    episodic_returns = {}
    checkpoint_paths = os.listdir(os.path.join(agent_path, "models"))
    env_config = None
    for idx, checkpoint_path in enumerate(checkpoint_paths):
        agent_config, aux = get_config(agent_path)
        env = get_environment(agent_config.learner_config.env_config)
        _, policy = get_agent(env, agent_config, aux)

        if idx == 0:
            if hasattr(env, "get_config"):
                env_config = env.get_config()

        model_id = int(checkpoint_path.split("-")[-1])
        if record_video:
            env = RecordVideoV0(
                env,
                f"videos/variant_{variant_name}-model_id_{model_id}-videos",
                disable_logger=True,
            )

        agent_model_path = os.path.join(agent_path, "models", checkpoint_path)
        agent_policy_params, agent_obs_rms = restore_agent_state(agent_model_path)

        agent_rollout = EvaluationRollout(env, seed=env_seed)
        agent_rollout.rollout(
            agent_policy_params, policy, agent_obs_rms, num_episodes, buffer, use_tqdm
        )

        episodic_returns[model_id] = agent_rollout.episodic_returns
    return {
        CONST_EPISODIC_RETURNS: episodic_returns,
        CONST_ENV_CONFIG: env_config,
    }
