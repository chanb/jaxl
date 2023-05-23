from jaxl.constants.buffers import *
from jaxl.constants.envs import *
from jaxl.constants.learners import *
from jaxl.constants.losses import *
from jaxl.constants.models import *
from jaxl.constants.optimizers import *

CONST_CONFIG = "config"
CONST_HYPERPARAMETERS = "hyperparameters"
CONST_MODEL_DICT = "model_dict"
CONST_OPTIMIZER = "optimizer"
CONST_OPT_STATE = "opt_state"
CONST_PARAMS = "params"
CONST_TRAIN = "train"
CONST_VALIDATION = "validation"
CONST_VAL_PREDS = "validation_predictions"
CONST_LOG = "log"

CONST_IL = "imitation_learning"
CONST_RL = "reinforcement_learning"

VALID_TASK = [CONST_IL, CONST_RL]

CONST_LATEST_RETURN = "latest_return"
CONST_LATEST_EPISODE_LENGTH = "latest_episode_length"
CONST_EPISODE_LENGTHS = "episode_lengths"
CONST_EPISODIC_RETURNS = "episodic_returns"

CONST_IS_RATIO = "is_ratio"
