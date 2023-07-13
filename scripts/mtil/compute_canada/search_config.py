HYPERPARAMETERS_CONFIG = {
    "ParameterizedPendulum-v0": {
        "model": [[64, 64], [128, 128]],
        "lr": [3e-4, 1e-3],
        "max_grad_norm": [False, 0.5],
        "obs_rms": [False, True],
        "opt_batch_size": [64, 128, 256],
        "opt_epochs": [4, 10],
        "vf_clip_param": [False, 0.2],
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
    "ParameterizedHopper-v0": {
        "model": [[128, 128]],
        "lr": [3e-4, 1e-3],
        "max_grad_norm": [False, 0.5],
        "obs_rms": [True],
        "opt_batch_size": [64, 128, 256, 512],
        "opt_epochs": [4, 10],
        "vf_clip_param": [False, 0.2],
        "value_rms": [True],
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
    "ParameterizedHalfCheetah-v0": {
        "model": [[128, 128]],
        "lr": [3e-4, 1e-3],
        "max_grad_norm": [False, 0.5],
        "obs_rms": [True],
        "opt_batch_size": [64, 128, 256, 512],
        "opt_epochs": [4, 10],
        "vf_clip_param": [False, 0.2],
        "buffer_size": [1024, 2048],
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
