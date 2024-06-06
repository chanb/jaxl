import dill
import math
import os
import tqdm

from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace

from jaxl.constants import *
from jaxl.learners import Learner
from jaxl.plot_utils import icl_image_grid
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
    elif learner_config.task == CONST_ICL:
        make_learner = jaxl_learners.get_icl_learner
    elif learner_config.task == CONST_SUPERVISED:
        make_learner = jaxl_learners.get_supervised_learner
    elif learner_config.task == CONST_WSRL:
        make_learner = jaxl_learners.get_wsrl_learner
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

    num_digits = int(math.log10(train_config.num_epochs)) + 1

    def pad_string(s):
        s = str(s)
        return "0" * (num_digits - len(s)) + s

    true_epoch = 0
    summary_writer = DummySummaryWriter()
    try:
        if save_path:
            os.makedirs(os.path.join(save_path, "models"), exist_ok=True)
            os.makedirs(os.path.join(save_path, "auxes"), exist_ok=True)
            summary_writer = SummaryWriter(log_dir=f"{save_path}/tensorboard")
            summary_writer.add_text(
                CONST_HYPERPARAMETERS,
                hyperparameter_str,
            )
            learner.save_env_config(os.path.join(save_path, "env_config.dill"))
            dill.dump(
                learner.checkpoint(),
                open(
                    os.path.join(save_path, "models", "{}.dill".format(pad_string(0))),
                    "wb",
                ),
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
                and (true_epoch % logging_config.checkpoint_interval == 0)
            ):
                with open(
                    os.path.join(save_path, "auxes", f"auxes-{true_epoch}.dill"),
                    "wb",
                ) as f:
                    dill.dump(train_aux, f)

                if CONST_DATA in train_aux and getattr(
                    config.logging_config, "image_data", False
                ):
                    os.makedirs(os.path.join(save_path, "imgs"), exist_ok=True)
                    icl_image_grid(
                        *train_aux[CONST_DATA],
                        doc_width_pt=1000,
                        filename=os.path.join(
                            save_path, "imgs/train_{}.png".format(true_epoch)
                        ),
                    )
                dill.dump(
                    learner.checkpoint(),
                    open(
                        os.path.join(
                            save_path,
                            "models",
                            "{}.dill".format(pad_string(true_epoch)),
                        ),
                        "wb",
                    ),
                )
    except KeyboardInterrupt:
        pass
    if save_path:
        dill.dump(
            learner.checkpoint(final=True),
            open(
                os.path.join(
                    save_path, "models", "{}.dill".format(pad_string(true_epoch))
                ),
                "wb",
            ),
        )
