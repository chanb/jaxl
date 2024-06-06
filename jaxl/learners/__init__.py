from types import SimpleNamespace

from jaxl.constants import *
from jaxl.learners.learner import Learner
from jaxl.learners.a2c import A2C
from jaxl.learners.bc import BC
from jaxl.learners.in_context import (
    InContextLearner,
    BinaryClassificationInContextLearner,
)
from jaxl.learners.mtbc import MTBC
from jaxl.learners.ppo import PPO
from jaxl.learners.reinforce import REINFORCE
from jaxl.learners.residual_learning.rlpd import ResidualRLPDSAC
from jaxl.learners.residual_learning.sac import ResidualSAC
from jaxl.learners.rlpd import RLPDSAC
from jaxl.learners.sac import SAC, CrossQSAC
from jaxl.learners.supervised import SupervisedLearner
from jaxl.learners.wsrl import WSRLPPO, WSRLREINFORCE, WSRLPolicyEvaluation


def get_rl_learner(
    learner_config: SimpleNamespace,
    model_config: SimpleNamespace,
    optimizer_config: SimpleNamespace,
) -> Learner:
    """
    Gets reinforcement learning learner.

    :param learner_config: the learner configuration
    :param model_config: the model configuration
    :param optimizer_config: the optimizer configuration
    :type learner_config: SimpleNamespace
    :type model_config: SimpleNamespace
    :type optimizer_config: SimpleNamespace
    :return: the reinforcement learning learner
    :rtype: Learner

    """
    assert (
        learner_config.learner in VALID_RL_LEARNER
    ), f"{learner_config.learner} is not supported (one of {VALID_RL_LEARNER})"
    if learner_config.learner == CONST_A2C:
        learner_constructor = A2C
    elif learner_config.learner == CONST_PPO:
        learner_constructor = PPO
    elif learner_config.learner == CONST_REINFORCE:
        learner_constructor = REINFORCE
    elif learner_config.learner == CONST_SAC:
        sac_variant = getattr(learner_config, "variant", CONST_DEFAULT)
        if sac_variant == CONST_DEFAULT:
            learner_constructor = SAC
        elif sac_variant == CONST_CROSS_Q:
            learner_constructor = CrossQSAC
        elif sac_variant == CONST_RLPD:
            learner_constructor = RLPDSAC
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return learner_constructor(learner_config, model_config, optimizer_config)


def get_wsrl_learner(
    learner_config: SimpleNamespace,
    model_config: SimpleNamespace,
    optimizer_config: SimpleNamespace,
) -> Learner:
    """
    Gets warm-start reinforcement learning learner.

    :param learner_config: the learner configuration
    :param model_config: the model configuration
    :param optimizer_config: the optimizer configuration
    :type learner_config: SimpleNamespace
    :type model_config: SimpleNamespace
    :type optimizer_config: SimpleNamespace
    :return: the reinforcement learning learner
    :rtype: Learner

    """
    assert (
        learner_config.learner in VALID_RL_LEARNER
    ), f"{learner_config.learner} is not supported (one of {VALID_RL_LEARNER})"
    if learner_config.learner == CONST_PPO:
        learner_constructor = WSRLPPO
    elif learner_config.learner == CONST_REINFORCE:
        learner_constructor = WSRLREINFORCE
    elif learner_config.learner == CONST_POLICY_EVALUATION:
        learner_constructor = WSRLPolicyEvaluation
    else:
        raise NotImplementedError

    return learner_constructor(learner_config, model_config, optimizer_config)


def get_residual_rl_learner(
    learner_config: SimpleNamespace,
    model_config: SimpleNamespace,
    optimizer_config: SimpleNamespace,
) -> Learner:
    """
    Gets residual reinforcement learning learner.

    :param learner_config: the learner configuration
    :param model_config: the model configuration
    :param optimizer_config: the optimizer configuration
    :type learner_config: SimpleNamespace
    :type model_config: SimpleNamespace
    :type optimizer_config: SimpleNamespace
    :return: the reinforcement learning learner
    :rtype: Learner

    """
    assert (
        learner_config.learner in VALID_RL_LEARNER
    ), f"{learner_config.learner} is not supported (one of {VALID_RL_LEARNER})"
    if learner_config.learner == CONST_SAC:
        sac_variant = getattr(learner_config, "variant", CONST_DEFAULT)
        if sac_variant == CONST_DEFAULT:
            learner_constructor = ResidualSAC
        elif sac_variant == CONST_RLPD:
            learner_constructor = ResidualRLPDSAC
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return learner_constructor(learner_config, model_config, optimizer_config)


def get_il_learner(
    learner_config: SimpleNamespace,
    model_config: SimpleNamespace,
    optimizer_config: SimpleNamespace,
) -> Learner:
    """
    Gets imitation learning learner.

    :param learner_config: the learner configuration
    :param model_config: the model configuration
    :param optimizer_config: the optimizer configuration
    :type learner_config: SimpleNamespace
    :type model_config: SimpleNamespace
    :type optimizer_config: SimpleNamespace
    :return: the imitation learning learner
    :rtype: Learner

    """
    assert (
        learner_config.learner in VALID_IL_LEARNER
    ), f"{learner_config.learner} is not supported (one of {VALID_IL_LEARNER})"
    if learner_config.learner == CONST_BC:
        learner_constructor = BC
    elif learner_config.learner == CONST_MTBC:
        learner_constructor = MTBC
    else:
        raise NotImplementedError

    return learner_constructor(learner_config, model_config, optimizer_config)


def get_icl_learner(
    learner_config: SimpleNamespace,
    model_config: SimpleNamespace,
    optimizer_config: SimpleNamespace,
) -> Learner:
    """
    Gets in-context learning learner.

    :param learner_config: the learner configuration
    :param model_config: the model configuration
    :param optimizer_config: the optimizer configuration
    :type learner_config: SimpleNamespace
    :type model_config: SimpleNamespace
    :type optimizer_config: SimpleNamespace
    :return: the in-context learning learner
    :rtype: Learner

    """
    assert (
        learner_config.learner in VALID_ICL_LEARNER
    ), f"{learner_config.learner} is not supported (one of {VALID_ICL_LEARNER})"
    if learner_config.learner == CONST_MLE:
        if learner_config.losses[0] == CONST_SIGMOID_BCE:
            learner_constructor = BinaryClassificationInContextLearner
        else:
            learner_constructor = InContextLearner
    else:
        raise NotImplementedError

    return learner_constructor(learner_config, model_config, optimizer_config)


def get_supervised_learner(
    learner_config: SimpleNamespace,
    model_config: SimpleNamespace,
    optimizer_config: SimpleNamespace,
) -> Learner:
    """
    Gets supervised learning learner.

    :param learner_config: the learner configuration
    :param model_config: the model configuration
    :param optimizer_config: the optimizer configuration
    :type learner_config: SimpleNamespace
    :type model_config: SimpleNamespace
    :type optimizer_config: SimpleNamespace
    :return: the supervised learning learner
    :rtype: Learner

    """
    assert (
        learner_config.learner in VALID_SUPERVISED_LEARNER
    ), f"{learner_config.learner} is not supported (one of {VALID_SUPERVISED_LEARNER})"
    if learner_config.learner == CONST_MLE:
        learner_constructor = SupervisedLearner
    else:
        raise NotImplementedError

    return learner_constructor(learner_config, model_config, optimizer_config)
