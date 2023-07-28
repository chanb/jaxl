from itertools import product

import json
import numpy as np
import os

run_seed = 0
rng = np.random.RandomState(run_seed)

envs = {
    "pendulum_continuous": {
        "config": "/Users/chanb/research/personal/jaxl/scripts/mtil/experiments/configs/bc_amount_data/pendulum_cont.json",
        "expert_data_prefixes": "/Users/chanb/research/personal/jaxl/scripts/mtil/local/demonstrations/expert_buffer-default-pendulum_continuous",
        "buffer_sizes": [2500, 5000, 7500, 10000],
    },
}

num_runs = 5
seeds = rng.randint(0, 1000, num_runs)

os.makedirs("./configs", exist_ok=True)
sh_content = ""
sh_content += "#!/bin/bash\n"
sh_content += "source /Users/chanb/research/personal/jaxl/.venv/bin/activate\n"
for env_name, env_config in envs.items():
    print(f"Processing {env_name}")

    with open(env_config["config"], "r") as f:
        template = json.load(f)

    for buffer_size, seed in product(env_config["buffer_sizes"], seeds):
        template["logging_config"]["experiment_name"] = f"buffer_size_{buffer_size}"
        template["learner_config"]["buffer_config"][
            "load_buffer"
        ] = "{}-subsampling_{}.gzip".format(
            env_config["expert_data_prefixes"], 200
        )
        template["learner_config"]["buffer_config"][
            "set_size"
        ] = buffer_size

        template["learner_config"]["seeds"] = {
            "model_seed": int(seed),
            "buffer_seed": int(seed),
        }

        out_path = "configs/{}-buffer_size_{}-seed_{}".format(
            env_name, buffer_size, seed
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
