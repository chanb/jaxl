CONST_ADAM = "adam"
CONST_FROZEN = "frozen"
CONST_SGD = "sgd"

VALID_OPTIMIZER = [CONST_ADAM, CONST_FROZEN, CONST_SGD]

CONST_MASK_NAMES = "mask_names"

CONST_CONSTANT_SCHEDULE = "constant_schedule"
CONST_EXPONENTIAL_DECAY = "exponential_decay"
CONST_LINEAR_SCHEDULE = "linear_schedule"
VALID_SCEHDULER = [
    CONST_CONSTANT_SCHEDULE,
    CONST_EXPONENTIAL_DECAY,
    CONST_LINEAR_SCHEDULE,
]
