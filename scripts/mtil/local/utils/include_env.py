import json
import os

env_map = {
    "pendulum": "ParameterizedPendulum-v0",
    "cheetah": "DMCCheetah-v0",
    "walker": "DMCWalker-v0",
}

runs_dir = "/Users/chanb/research/personal/mtil_results/data_from_pretrain/bc_main"
dirs_to_runs_dir = runs_dir.split("/")

for run_path, _, filenames in os.walk(runs_dir):
    for filename in filenames:
        if filename != "config.json":
            continue

        config_path = os.path.join(run_path, filename)
        with open(config_path, "r") as f:
            config = json.load(f)

        dirs_to_config = config_path.split("/")[len(dirs_to_runs_dir) :]
        env_name = env_map[dirs_to_config[0]]
        control_mode = dirs_to_config[1]

        # Construct env config for easier evaluation
        config["learner_config"]["env_config"] = {
            "env_type": "gym",
            "env_name": env_name,
            "env_kwargs": {
                "seed": int(
                    config["learner_config"]["buffer_config"]["load_buffer"].split("-")[
                        1
                    ]
                ),
                "use_default": False,
                "control_mode": control_mode,
            },
        }

        with open(config_path, "w+") as f:
            json.dump(config, f)
