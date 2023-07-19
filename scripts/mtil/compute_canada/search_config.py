POLICY_CONFIG = {
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
    },
}

HYPERPARAMETERS_CONFIG = {
    "ppo": {
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
}


HYPERPARAMETER_ROBUSTNESS_POLICY_CONFIG = {
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
    },
}


HYPERPARAMETER_ROBUSTNESS_CONFIG = {
    "ppo": {
        "buffer_size": [2048],
        "max_grad_norm": [False, 0.5],
        "opt_batch_size": [128, 256],
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
}
