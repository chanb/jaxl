from itertools import product

import json
import numpy as np
import os

run_seed = 0
rng = np.random.RandomState(run_seed)

envs = {
    # "frozenlake_discrete": {
    #     "subsamplings": [1, 20, 200],
    # },
    # "cheetah_discrete": {
    #     "subsamplings": [1, 20, 1000],
    # },
    # "cheetah_continuous": {
    #     "subsamplings": [1, 20, 1000],
    # },
    # "cartpole_continuous": {
    #     "subsamplings": [1, 20, 1000],
    # },
    "walker_discrete": {
        "subsamplings": [1, 20, 1000],
    },
    "walker_continuous": {
        "subsamplings": [1, 20, 1000],
    },
}

data_dir = "./logs/demonstrations"
config_template = "/Users/chanb/research/personal/jaxl/scripts/mtil/experiments/configs/bc_subsampling/bc_template.json"
buffer_sizes = [100, 500, 1000, 2500, 5000, 7500]

num_runs = 5
seeds = rng.randint(0, 1000, num_runs)

with open(config_template, "r") as f:
    template = json.load(f)

os.makedirs("./configs", exist_ok=True)
sh_content = ""
sh_content += "#!/bin/bash\n"
sh_content += "source /Users/chanb/research/personal/jaxl/.venv/bin/activate\n"

for env_name, env_config in envs.items():
    print(f"Processing {env_name}")

    for buffer_size in buffer_sizes:
        template["logging_config"][
            "save_path"
        ] = "./logs/bc_subsampling/{}-size_{}".format(env_name, buffer_size)
        if env_name.split("_")[1] == "discrete":
            template["learner_config"]["losses"][0] = "categorical"
        elif env_name.split("_")[1] == "continuous":
            template["learner_config"]["losses"][0] = "gaussian"

        for subsampling, seed in product(env_config["subsamplings"], seeds):
            template["logging_config"]["experiment_name"] = f"subsampling_{subsampling}"
            template["learner_config"]["buffer_config"][
                "load_buffer"
            ] = "{}/expert_buffer-default-{}-num_samples_10000-subsampling_{}.gzip".format(
                data_dir, env_name, subsampling
            )
            template["learner_config"]["buffer_config"]["set_size"] = buffer_size
            template["learner_config"]["seeds"] = {
                "model_seed": int(seed),
                "buffer_seed": int(seed),
            }

            out_path = "configs/{}-subsampling_{}-size_{}-seed_{}".format(
                env_name, subsampling, buffer_size, seed
            )
            with open(f"{out_path}.json", "w+") as f:
                json.dump(template, f)

            sh_content += "python /Users/chanb/research/personal/jaxl/jaxl/main.py --config_path={}.json\n".format(
                out_path
            )

with open(
    os.path.join(f"./run_experiment.sh"),
    "w+",
) as f:
    f.writelines(sh_content)
