SINGLE_SWEEP = {
    "ppo": {
        "continuous": {
            "policy_distribution": "gaussian",
            "objective": "clip",
            "hyperparameters": {"clip_param": [False, 0.1, 0.2]},
        },
        "discrete": {
            "policy_distribution": "softmax",
            "objective": "reverse_kl",
            "hyperparameters": {"beta": [0.2, 0.02, 0.002]},
        },
        "general": {
            "buffer_size": [2048],
            "max_grad_norm": [False, 0.5],
            "opt_batch_size": [64, 128, 256],
            "opt_epochs": [100, 200],
            "ent_coef": [
                {
                    "scheduler": "constant_schedule",
                    "scheduler_kwargs": {"value": 0.0},
                },
                {
                    "scheduler": "linear_schedule",
                    "scheduler_kwargs": {
                        "init_value": 0.002,
                        "end_value": 0.0,
                        "transition_begin": 0,
                        "transition_steps": 100,
                    },
                },
            ],
        },
    },
}


HYPERPARAM_SWEEP = {
    "ppo": {
        "continuous": {
            "policy_distribution": "gaussian",
            "objective": "clip",
            "hyperparameters": {"clip_param": [0.1, 0.2]},
        },
        "discrete": {
            "policy_distribution": "softmax",
            "objective": "reverse_kl",
            "hyperparameters": {"beta": [0.02, 0.002]},
        },
        "general": {
            "buffer_size": [2048],
            "max_grad_norm": [
                0.5
            ],  # clipping gradient norm is better, tested on [False, 0.5]
            "opt_batch_size": [128, 256],
            "opt_epochs": [200],  # more epochs is better, tested on [100, 200]
            "ent_coef": [
                {
                    "scheduler": "constant_schedule",
                    "scheduler_kwargs": {"value": 0.0},
                },
                {
                    "scheduler": "linear_schedule",
                    "scheduler_kwargs": {
                        "init_value": 0.002,
                        "end_value": 0.0,
                        "transition_begin": 0,
                        "transition_steps": 100,
                    },
                },
            ],
        },
    },
}

PENDULUM_CONTINUOUS = {
    "ppo": {
        "discrete": {
            "policy_distribution": "gaussian",
            "objective": "clip",
            "hyperparameters": {"clip_param": [0.2]},
        },
        "general": {
            "buffer_size": [2048],
            "max_grad_norm": [0.5],
            "opt_batch_size": [128],
            "opt_epochs": [200],
            "ent_coef": [
                {
                    "scheduler": "constant_schedule",
                    "scheduler_kwargs": {"value": 0.0},
                }
            ],
        },
    },
}

PENDULUM_DISCRETE = {
    "ppo": {
        "discrete": {
            "policy_distribution": "softmax",
            "objective": "reverse_kl",
            "hyperparameters": {"beta": [0.002]},
        },
        "general": {
            "buffer_size": [2048],
            "max_grad_norm": [0.5],
            "opt_batch_size": [256],
            "opt_epochs": [200],
            "ent_coef": [
                {
                    "scheduler": "constant_schedule",
                    "scheduler_kwargs": {"value": 0.0},
                }
            ],
        },
    },
}

CHEETAH_CONTINUOUS = {
    "ppo": {
        "continuous": {
            "policy_distribution": "gaussian",
            "objective": "clip",
            "hyperparameters": {"clip_param": [0.2]},
        },
        "general": {
            "buffer_size": [2048],
            "max_grad_norm": [0.5],
            "opt_batch_size": [128],
            "opt_epochs": [200],
            "ent_coef": [
                {
                    "scheduler": "constant_schedule",
                    "scheduler_kwargs": {"value": 0.0},
                }
            ],
        },
    },
}

CHEETAH_DISCRETE = {
    "ppo": {
        "discrete": {
            "policy_distribution": "softmax",
            "objective": "reverse_kl",
            "hyperparameters": {"beta": [0.02]},
        },
        "general": {
            "buffer_size": [2048],
            "max_grad_norm": [0.5],
            "opt_batch_size": [128],
            "opt_epochs": [200],
            "ent_coef": [
                {
                    "scheduler": "constant_schedule",
                    "scheduler_kwargs": {"value": 0.0},
                }
            ],
        },
    },
}

WALKER_CONTINUOUS = {
    "ppo": {
        "continuous": {
            "policy_distribution": "gaussian",
            "objective": "clip",
            "hyperparameters": {"clip_param": [0.1]},
        },
        "general": {
            "buffer_size": [2048],
            "max_grad_norm": [0.5],
            "opt_batch_size": [128],
            "opt_epochs": [200],
            "ent_coef": [
                {
                    "scheduler": "constant_schedule",
                    "scheduler_kwargs": {"value": 0.0},
                }
            ],
        },
    },
}

WALKER_DISCRETE = {
    "ppo": {
        "discrete": {
            "policy_distribution": "softmax",
            "objective": "reverse_kl",
            "hyperparameters": {"beta": [0.002]},
        },
        "general": {
            "buffer_size": [2048],
            "max_grad_norm": [0.5],
            "opt_batch_size": [128],
            "opt_epochs": [200],
            "ent_coef": [
                {
                    "scheduler": "constant_schedule",
                    "scheduler_kwargs": {"value": 0.0},
                }
            ],
        },
    },
}

HYPERPARAM_SETS = {
    "single_sweep": SINGLE_SWEEP,
    "hyperparam_sweep": HYPERPARAM_SWEEP,
    "pendulum_cont": PENDULUM_CONTINUOUS,
    "pendulum_disc": PENDULUM_DISCRETE,
    "cheetah_cont": CHEETAH_CONTINUOUS,
    "cheetah_disc": CHEETAH_DISCRETE,
    "walker_cont": WALKER_CONTINUOUS,
    "walker_disc": WALKER_DISCRETE,
}
