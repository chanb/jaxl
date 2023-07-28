from itertools import product

import json
import numpy as np
import os

run_seed = 0
rng = np.random.RandomState(run_seed)

envs = {
    "pendulum_continuous": {
        "config": "/Users/chanb/research/personal/jaxl/scripts/mtil/experiments/configs/bc_subsampling/pendulum_cont.json",
        "expert_data_prefixes": "/Users/chanb/research/personal/jaxl/scripts/mtil/local/bc_subsampling/logs/expert_buffer-default-pendulum_continuous",
        "subsamplings": [1, 20, 200],
    },
    "cheetah_discrete": {
        "config": "/Users/chanb/research/personal/jaxl/scripts/mtil/experiments/configs/bc_subsampling/cheetah_disc.json",
        "expert_data_prefixes": "/Users/chanb/research/personal/jaxl/scripts/mtil/local/bc_subsampling/logs/expert_buffer-default-cheetah_discrete",
        "subsamplings": [1, 20, 1000],
    },
}

num_runs = 5

seeds = rng.randint(0, 1000, num_runs)

os.makedirs("./configs")
sh_content = ""
sh_content += "#!/bin/bash\n"
sh_content += "source /Users/chanb/research/personal/jaxl/.venv/bin/activate\n"
for env_name, env_config in envs.items():
    print(f"Processing {env_name}")

    with open(env_config["config"], "r") as f:
        template = json.load(f)

    for subsampling, seed in product(env_config["subsamplings"], seeds):
        template["logging_config"]["experiment_name"] = f"subsampling_{subsampling}"
        template["learner_config"]["buffer_config"][
            "load_buffer"
        ] = "{}-subsampling_{}.gzip".format(
            env_config["expert_data_prefixes"], subsampling
        )
        template["learner_config"]["seeds"] = {
            "model_seed": int(seed),
            "buffer_seed": int(seed),
        }

        out_path = "configs/{}-subsampling_{}-seed_{}".format(
            env_name, subsampling, seed
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
