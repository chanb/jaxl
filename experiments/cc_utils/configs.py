EXPERIMENTS = {
    "random_high_prob-linearly_separable": {
        "run_time": "03:00:00",
        "num_seeds": 10,
        "variant": "high_prob",
        "values": [0.5, 0.75, 0.9, 0.99, 0.999],
    },
    "random_high_prob-linearly_separable-more_samples": {
        "run_time": "06:00:00",
        "num_seeds": 10,
        "variant": "high_prob",
        "values": [0.5, 0.75, 0.9, 0.99, 0.999],
    },
    "random_high_prob-linearly_separable-one_sample": {
        "run_time": "03:00:00",
        "num_seeds": 10,
        "variant": "high_prob",
        "values": [0.5, 0.75, 0.9, 0.99, 0.999],
    },
    "random_high_prob-linearly_separable-one_sample-noise_0.1": {
        "run_time": "03:00:00",
        "num_seeds": 10,
        "variant": "high_prob",
        "values": [0.5, 0.75, 0.9, 0.99, 0.999],
    }
}

EVALUATION = {
    "num_eval_samples": 1000,
    "batch_size": 100,
}
