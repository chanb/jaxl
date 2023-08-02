import json
import os

import jaxl.envs

from jaxl.constants import *
from jaxl.envs import get_environment
from jaxl.models import get_model, get_policy, policy_output_dim
from jaxl.models.policies import MultitaskPolicy
from jaxl.utils import set_dict_value, get_dict_value, parse_dict


def set_size(width_pt, fraction=1, subplots=(1, 1)):
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

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

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
            agent_config_dict["learner_config"][
                "policy_distribution"
            ] = ref_agent_config_dict["learner_config"]["policy_distribution"]
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
