import _pickle as pickle
import json
import os
import tqdm

from gymnasium import Env
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from typing import Any, Dict, Tuple, Union

from jaxl.buffers import get_buffer, ReplayBuffer
from jaxl.constants import *
from jaxl.envs import get_environment
from jaxl.learners import Learner
from jaxl.models import get_model, get_policy, policy_output_dim, Policy
from jaxl.utils import DummySummaryWriter, parse_dict, RunningMeanStd

import jaxl.learners as jaxl_learners


"""
Getter for the learner.
XXX: Feel free to add new components as needed.
"""


def get_learner(
    learner_config: SimpleNamespace,
    model_config: SimpleNamespace,
    optimizer_config: SimpleNamespace,
) -> Learner:
    """
    Gets a learner.

    :param learner_config: the learner configuration
    :param model_config: the model configuration
    :param optimizer_config: the optimizer configuration
    :type learner_config: SimpleNamespace
    :type model_config: SimpleNamespace
    :type optimizer_config: SimpleNamespace
    :return: a learner
    :type: Learner

    """
    assert (
        learner_config.task in VALID_TASK
    ), f"{learner_config.task} is not supported (one of {VALID_TASK})"

    if learner_config.task == CONST_RL:
        make_learner = jaxl_learners.get_rl_learner
    elif learner_config.task == CONST_IL:
        make_learner = jaxl_learners.get_il_learner
    else:
        raise NotImplementedError
    return make_learner(learner_config, model_config, optimizer_config)


"""
This function runs the training and keeps track of the progress.
Any learner will be using this function.
XXX: Try not to modify this
"""


def train(
    learner: Learner,
    config: SimpleNamespace,
    hyperparameter_str: str,
    save_path: str = None,
):
    """
    Executes the training loop.

    :param learner: the learner
    :param config: the experiment configuration
    :param hyperparameter_str: the hyperparameter setting in string format
    :param save_path: the directory to save the experiment progress
    :type learner: Learner
    :type config: SimpleNamespace
    :type hyperparameter_str: str
    :type save_path: str: (Default value = None)

    """
    logging_config = config.logging_config
    train_config = config.train_config

    true_epoch = 0
    summary_writer = DummySummaryWriter()
    try:
        if save_path:
            if logging_config.checkpoint_interval:
                os.makedirs(os.path.join(save_path, "models"), exist_ok=True)
                os.makedirs(os.path.join(save_path, "auxes"), exist_ok=True)
            summary_writer = SummaryWriter(log_dir=f"{save_path}/tensorboard")
            summary_writer.add_text(
                CONST_HYPERPARAMETERS,
                hyperparameter_str,
            )
            learner.save_env_config(os.path.join(save_path, "env_config.pkl"))
            params = learner.checkpoint(final=False)

            checkpoint_manager = CheckpointManager(
                os.path.join(save_path, "models"),
                PyTreeCheckpointer(),
            )

        for epoch in tqdm.tqdm(range(train_config.num_epochs)):
            train_aux = learner.update()
            true_epoch = epoch + 1

            if (
                save_path
                and logging_config.log_interval
                and true_epoch % logging_config.log_interval == 0
            ):
                if CONST_LOG in train_aux:
                    # NOTE: we expect the user to properly define the logging scalars in the learner
                    for key, val in train_aux[CONST_LOG].items():
                        summary_writer.add_scalar(key, val, true_epoch)

            if (
                save_path
                and logging_config.checkpoint_interval
                and (
                    true_epoch % logging_config.checkpoint_interval == 0
                    or true_epoch == 1
                )
            ):
                with open(
                    os.path.join(save_path, "auxes", f"auxes-{true_epoch}.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(train_aux, f)
                checkpoint_manager.save(true_epoch, learner.checkpoint(final=False))
    except KeyboardInterrupt:
        pass
    if save_path:
        checkpoint_manager.save(true_epoch, learner.checkpoint(final=True))


def load_evaluation_components(
    run_path: str,
    buffer_size: int,
) -> Tuple[
    Policy,
    Dict[str, Any],
    Union[RunningMeanStd, bool],
    Env,
    ReplayBuffer,
    int,
]:
    """

    Loads the latest checkpointed agent, the buffer, and the environment.

    :param run_path: the configuration file path to load the components from
    :param buffer_size: the buffer size
    :type run_path: str
    :type buffer_size: int
    :return: the latest checkpointed agent, the buffer, and the environment
    :rtype: Tuple[Policy, Dict[str, Any], Union[RunningMeanStd, bool], ReplayBuffer, Env, int,]

    """
    assert buffer_size > 0, f"buffer_size {buffer_size} needs to be at least 1."
    assert os.path.isdir(run_path), f"{run_path} is not a directory"

    agent_config_path = os.path.join(run_path, "config.json")
    with open(agent_config_path, "r") as f:
        agent_config_dict = json.load(f)

    agent_config_dict["learner_config"]["buffer_config"]["buffer_size"] = buffer_size
    agent_config_dict["learner_config"]["buffer_config"]["buffer_type"] = CONST_DEFAULT
    agent_config_dict["learner_config"]["env_config"]["env_kwargs"][
        "render_mode"
    ] = "rgb_array"
    agent_config = parse_dict(agent_config_dict)

    h_state_dim = (1,)
    if hasattr(agent_config.model_config, "h_state_dim"):
        h_state_dim = agent_config.model_config.h_state_dim
    env = get_environment(agent_config.learner_config.env_config)
    env_seed = agent_config.learner_config.seeds.env_seed

    buffer = get_buffer(
        agent_config.learner_config.buffer_config,
        agent_config.learner_config.seeds.buffer_seed,
        env,
        h_state_dim,
    )
    input_dim = buffer.input_dim
    output_dim = policy_output_dim(buffer.output_dim, agent_config.learner_config)
    model = get_model(
        input_dim,
        output_dim,
        getattr(agent_config.model_config, "policy", agent_config.model_config),
    )
    policy = get_policy(model, agent_config.learner_config)

    checkpoint_manager = CheckpointManager(
        os.path.join(run_path, "models"),
        PyTreeCheckpointer(),
    )

    params = checkpoint_manager.restore(checkpoint_manager.latest_step())
    model_dict = params[CONST_MODEL_DICT]
    policy_params = model_dict[CONST_MODEL][CONST_POLICY]
    obs_rms = False
    if CONST_OBS_RMS in params:
        obs_rms = RunningMeanStd()
        obs_rms.set_state(params[CONST_OBS_RMS])
    return policy, policy_params, obs_rms, buffer, env, env_seed
