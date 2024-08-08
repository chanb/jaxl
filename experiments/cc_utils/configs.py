EXPERIMENTS = {
    "linearly_separable-default": {
        "run_time": "00:45:00",
        "num_seeds": 10,
        "variant": "high_prob",
        "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99],
    },
    # Margin
    "linearly_separable-margin_0.0": {
        "run_time": "00:45:00",
        "num_seeds": 10,
        "variant": "high_prob",
        "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99],
    },
    "linearly_separable-margin_0.2": {
        "run_time": "00:45:00",
        "num_seeds": 10,
        "variant": "high_prob",
        "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99],
    },
    "linearly_separable-margin_0.02": {
        "run_time": "00:45:00",
        "num_seeds": 10,
        "variant": "high_prob",
        "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99],
    },
    # Input noise
    "linearly_separable-noise_std_0.1": {
        "run_time": "00:45:00",
        "num_seeds": 10,
        "variant": "high_prob",
        "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99],
    },
    "linearly_separable-noise_std_0.5": {
        "run_time": "00:45:00",
        "num_seeds": 10,
        "variant": "high_prob",
        "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99],
    },
    "linearly_separable-noise_std_1.0": {
        "run_time": "00:45:00",
        "num_seeds": 10,
        "variant": "high_prob",
        "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99],
    },
    # Number of high (and low) frequency classes
    "linearly_separable-num_frequency_classes_100": {
        "run_time": "00:45:00",
        "num_seeds": 10,
        "variant": "high_prob",
        "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99],
    },
    "linearly_separable-num_frequency_classes_10000": {
        "run_time": "01:30:00",
        "num_seeds": 10,
        "variant": "high_prob",
        "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99],
    },
    "linearly_separable-num_frequency_classes_100000": {
        "run_time": "07:00:00",
        "num_seeds": 10,
        "variant": "high_prob",
        "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99],
    },
    # Non-linear decision boundary
    "non_linearly_separable-default": {
        "run_time": "01:00:00",
        "num_seeds": 10,
        "variant": "high_prob",
        "values": [0.5, 0.67, 0.75, 0.8, 0.9, 0.99],
    },
}

EVALUATION = {
    "num_eval_samples": 1000,
    "batch_size": 100,
}
