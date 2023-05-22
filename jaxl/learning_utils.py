import _pickle as pickle
import os
import tqdm

from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace

from jaxl.constants import *
from jaxl.learners import Learner
from jaxl.utils import DummySummaryWriter

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
    save_path: str = None,
):
    logging_config = config.logging_config
    train_config = config.train_config

    summary_writer = DummySummaryWriter()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        if logging_config.checkpoint_interval:
            os.makedirs(os.path.join(save_path, "models"), exist_ok=True)
        if logging_config.log_interval:
            os.makedirs(os.path.join(save_path, "auxes"), exist_ok=True)
        with open(os.path.join(save_path, "config.pkl"), "wb") as f:
            pickle.dump(config, f)
        summary_writer = SummaryWriter(log_dir=f"{save_path}/tensorboard")

    # try:
    true_epoch = 0
    for epoch in tqdm.tqdm(range(train_config.num_epochs)):
        train_aux = learner.update()
        true_epoch = epoch + 1

        if (
            logging_config.log_interval
            and true_epoch % logging_config.log_interval == 0
        ):
            if save_path:
                with open(
                    os.path.join(save_path, "auxes", f"auxes-{true_epoch}.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(train_aux, f)

                if CONST_LOG in train_aux:
                    # NOTE: we expect the user to properly define the logging scalars in the learner
                    for key, val in train_aux[CONST_LOG].items():
                        summary_writer.add_scalar(key, val, true_epoch)

        if (
            save_path
            and logging_config.checkpoint_interval
            and true_epoch % logging_config.checkpoint_interval == 0
        ):
            learner.checkpoint(os.path.join(save_path, "models", f"model-{true_epoch}"))
    # finally:
    if save_path:
        termination_save_path = os.path.join(
            save_path, f"termination_model-epoch_{true_epoch}"
        )
        learner.checkpoint(termination_save_path)
        learner.save_buffer(f"{termination_save_path}.gzip")
