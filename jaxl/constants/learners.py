CONST_MLE = "mle"

VALID_REGRESSION_LEARNER = [CONST_MLE]

CONST_PGD = "pgd"

VALID_TRAIN_STEP_WRAPPER = [CONST_PGD]

CONST_PRE_PARAM_NORM = "pre_param_norm"
CONST_POST_PARAM_NORM = "post_param_norm"

CONST_A2C = "a2c"
CONST_PPO = "ppo"
CONST_REINFORCE = "reinforce"
CONST_SAC = "sac"

VALID_RL_LEARNER = [CONST_A2C, CONST_PPO, CONST_REINFORCE, CONST_SAC]

CONST_BC = "bc"
CONST_MTBC = "mtbc"
VALID_IL_LEARNER = [CONST_BC, CONST_MTBC]

VALID_ICL_LEARNER = [CONST_MLE]

CONST_PRE_PARAM_NORM = "pre_param_norm"
CONST_POST_PARAM_NORM = "post_param_norm"

CONST_REGULARIZATION = "regularization"
CONST_PARAM_NORM = "param_norm"
CONST_GRAD_NORM = "grad_norm"
CONST_STOP_UPDATE = "stop_update"

CONST_INPUT_RMS = "input_rms"
CONST_OBS_RMS = "obs_rms"
CONST_VALUE_RMS = "value_rms"

CONST_UPDATE_TIME = "update_time"
CONST_ROLLOUT_TIME = "rollout_time"

CONST_PI_LOSS_SETTING = "pi_loss_setting"
CONST_VF_LOSS_SETTING = "vf_loss_setting"

CONST_NUM_CLIPPED = "num_clipped"

CONST_REVERSE_KL = "reverse_kl"
CONST_CLIP = "clip"
VALID_PPO_OBJECTIVE = [CONST_CLIP, CONST_REVERSE_KL]

CONST_CONTEXT_INPUT = "context_input"
CONST_CONTEXT_OUTPUT = "context_output"
