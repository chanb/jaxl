import io
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image

import jaxl.envs

from jaxl.constants import *
from jaxl.envs import get_environment
from jaxl.models import get_model, get_policy, policy_output_dim
from jaxl.models.policies import MultitaskPolicy
from jaxl.utils import set_dict_value, get_dict_value, parse_dict


def set_size(width_pt, fraction=1, subplots=(1, 1), use_golden_ratio=True):
    """
    Reference: https://jwalton.info/Matplotlib-latex-PGF/
    Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    if use_golden_ratio:
        # Golden ratio to set aesthetic figure height
        golden_ratio = (5**0.5 - 1) / 2

        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    else:
        fig_height_in = fig_width_in * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,  # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    "text.latex.preamble": r"\usepackage{amsmath}",
}


def get_config(agent_path, env_seed=None, ref_agent_path=None):
    agent_config_path = os.path.join(agent_path, "config.json")
    with open(agent_config_path, "r") as f:
        agent_config_dict = json.load(f)

        if ref_agent_path is not None:
            with open(os.path.join(ref_agent_path, "config.json"), "r") as f2:
                ref_agent_config_dict = json.load(f2)
            agent_config_dict["learner_config"]["env_config"] = ref_agent_config_dict[
                "learner_config"
            ]["env_config"]
            agent_config_dict["learner_config"]["policy_distribution"] = (
                ref_agent_config_dict["learner_config"]["policy_distribution"]
            )
            if (
                agent_config_dict["learner_config"]["policy_distribution"]
                == CONST_GAUSSIAN
            ):
                agent_config_dict["learner_config"][
                    "policy_distribution"
                ] = CONST_DETERMINISTIC

        agent_config_dict["learner_config"]["env_config"]["env_kwargs"][
            "render_mode"
        ] = "rgb_array"
        if "policy_distribution" not in agent_config_dict["learner_config"]:
            agent_config_dict["learner_config"][
                "policy_distribution"
            ] = CONST_DETERMINISTIC

        if env_seed is not None:
            agent_config_dict["learner_config"]["env_config"]["env_kwargs"][
                "seed"
            ] = env_seed

            agent_config_dict["learner_config"]["env_config"]["env_kwargs"][
                "use_default"
            ] = False

        if (
            "parameter_config_path"
            in agent_config_dict["learner_config"]["env_config"]["env_kwargs"]
        ):
            curr_config_path = agent_config_dict["learner_config"]["env_config"][
                "env_kwargs"
            ]["parameter_config_path"]
            agent_config_dict["learner_config"]["env_config"]["env_kwargs"][
                "parameter_config_path"
            ] = os.path.join(
                os.path.dirname(jaxl.envs.__file__),
                curr_config_path.split("/envs/")[-1],
            )

        set_dict_value(agent_config_dict, "vmap_all", False)
        (multitask, num_models) = get_dict_value(agent_config_dict, "num_models")
        agent_config = parse_dict(agent_config_dict)
    return agent_config, {
        "multitask": multitask,
        "num_models": num_models,
    }


def get_evaluation_components(agent_path, env_seed=None, ref_agent_path=None):
    agent_config, aux = get_config(agent_path, env_seed, ref_agent_path)
    env = get_environment(agent_config.learner_config.env_config)

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

    return env, policy


# NOTE: For plotting in Tensorboard
def plot_to_image(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape((3,) + fig.canvas.get_width_height()[::-1])
    return img


def icl_image_grid(
    context_inputs,
    context_outputs,
    queries,
    labels,
    preds,
    doc_width_pt=500.0,
    filename=None,
):
    num_samples, context_len = context_inputs.shape[:2]

    # Create a figure to contain the plot.
    nrows = num_samples * 2
    num_contexts_per_col = math.ceil(context_len / 2)
    ncols = num_contexts_per_col + 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=set_size(doc_width_pt, 0.95, (nrows, ncols), False),
        layout="constrained",
    )

    for sample_i, (
        context_inputs_i,
        context_outputs_i,
        query_i,
        label_i,
        pred_i,
    ) in enumerate(
        zip(
            context_inputs,
            context_outputs,
            queries,
            labels,
            preds,
        )
    ):
        for context_i in range(context_len):
            row_i = context_i // num_contexts_per_col + sample_i * 2
            col_i = context_i % num_contexts_per_col

            axes[row_i, col_i].set_xticks([])
            axes[row_i, col_i].set_yticks([])
            axes[row_i, col_i].grid(False)
            axes[row_i, col_i].set_title(
                "{}".format(np.argmax(context_outputs_i[context_i]))
            )
            axes[row_i, col_i].imshow(context_inputs_i[context_i], cmap=plt.cm.binary)
        axes[sample_i * 2, -1].imshow(query_i[0])
        axes[sample_i * 2, -1].set_title("Label: {}".format(np.argmax(label_i)))
        axes[sample_i * 2, -1].set_xticks([])
        axes[sample_i * 2, -1].set_yticks([])
        axes[sample_i * 2 + 1, -1].axis("off")
        axes[sample_i * 2 + 1, -1].set_title("Pred: {}".format(np.argmax(pred_i)))
        title = ""
        top_10 = np.argsort(pred_i)[-10:]
        for idx in top_10:
            title += "idx {}: {:.4f}\n".format(idx, pred_i[idx])
        axes[sample_i * 2 + 1, -1].text(0.25, 0.0, title, fontsize=6)

    if filename is not None:
        fig.savefig(filename, format="png")

    return fig
