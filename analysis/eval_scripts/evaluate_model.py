import _pickle as pickle
import argparse
import numpy as np
import os


from jaxl.constants import *
from jaxl.models.utils import (
    get_model,
    load_config,
    load_params,
    get_wsrl_model,
    get_policy,
    policy_output_dim,
    get_residual_policy,
)
from jaxl.envs import get_environment
from jaxl.envs.rollouts import EvaluationRollout
from jaxl.utils import get_device, parse_dict


def main(args):
    get_device(args.device)
    total_episodes = args.total_episodes
    eval_seed = args.eval_seed
    render = args.render
    random = args.random
    learner_path = args.learner_path
    checkpoint_idx = args.checkpoint_idx
    save_dir = args.save_dir

    learner_name = os.path.basename(learner_path)

    _, config = load_config(learner_path)
    env_config = {
        "env_type": "manipulator_learning",
        "env_name": "PandaPlayInsertTrayXYZState",
        "env_kwargs": {"main_task": "stack", "dense_reward": False},
    }
    env = get_environment(parse_dict(env_config))

    include_absorbing_state = False
    if config.learner_config.task == CONST_RESIDUAL:
        backbone_act_dim = policy_output_dim(env.act_dim, config.model_config.backbone)
        residual_act_dim = policy_output_dim(env.act_dim, config.model_config.residual)

        backbone_model = get_model(
            env.observation_space.shape, backbone_act_dim, config.model_config.backbone
        )
        residual_model = get_model(
            env.observation_space.shape, residual_act_dim, config.model_config.residual
        )
        policy = get_residual_policy(
            backbone_model,
            residual_model,
            config.model_config,
        )
    else:
        model_out_dim = policy_output_dim(env.act_dim, config.learner_config)
        if config.learner_config.learner == CONST_BC:
            model = get_model(
                int(np.prod(env.observation_space.shape)) + 1,
                env.act_dim,
                config.model_config,
            )
            include_absorbing_state = True
        elif config.learner_config.task == CONST_WSRL:
            model = get_wsrl_model(
                env.observation_space.shape, model_out_dim, config.model_config.policy
            )
            include_absorbing_state = True
        else:
            model = get_model(
                env.observation_space.shape, model_out_dim, config.model_config.policy
            )
        policy = get_policy(model, config.learner_config)

    params = load_params(f"{learner_path}:{checkpoint_idx}")

    rollout = EvaluationRollout(env, eval_seed)

    rollout.rollout(
        params[CONST_MODEL_DICT][CONST_MODEL][CONST_POLICY],
        policy,
        False,
        total_episodes,
        random=random,
        render=render,
        include_absorbing_state=include_absorbing_state,
    )

    print(
        "Average return: {} +/- {}".format(
            np.mean(rollout.episodic_returns), np.std(rollout.episodic_returns)
        )
    )
    save_path = os.path.join(
        save_dir, f"{learner_name}-checkpoint_idx_{checkpoint_idx}.pkl"
    )
    pickle.dump(rollout.episodic_returns, open(save_path, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cpu", help="The CUDA device to run evaluation on"
    )
    parser.add_argument(
        "--total_episodes",
        type=int,
        default=50,
        help="Number of episodes to evaluate the model on",
    )
    parser.add_argument("--eval_seed", type=int, default=42, help="The seed for RNG")
    parser.add_argument(
        "--render", action="store_true", help="Rendering the evaluation"
    )
    parser.add_argument(
        "--random", action="store_true", help="Use stochastic policy if available"
    )
    parser.add_argument(
        "--learner_path", type=str, required=True, help="The model to evaluate"
    )
    parser.add_argument(
        "--checkpoint_idx",
        type=str,
        required=True,
        help="The model checkpoint to evaluate",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="The directory to save the result in",
    )
    args = parser.parse_args()
    main(args)
