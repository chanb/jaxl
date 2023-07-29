from itertools import product

import json
import numpy as np
import os

run_seed = 0
rng = np.random.RandomState(run_seed)

variants = {
    "clipped-tanh": {
        "config": "/Users/chanb/research/personal/jaxl/scripts/mtil/experiments/configs/objective_comparison/clipped.json",
    },
    "reverse_kl-tanh": {
        "config": "/Users/chanb/research/personal/jaxl/scripts/mtil/experiments/configs/objective_comparison/reverse_kl.json",
    },
}

num_runs = 5

seeds = rng.randint(0, 1000, num_runs)

os.makedirs("./configs", exist_ok=True)
sh_content = ""
sh_content += "#!/bin/bash\n"
sh_content += "source /Users/chanb/research/personal/jaxl/.venv/bin/activate\n"
for variant_name, variant_config in variants.items():
    print(f"Processing {variant_name}")

    with open(variant_config["config"], "r") as f:
        template = json.load(f)

    for seed in seeds:
        template["learner_config"]["seeds"] = {
            "model_seed": int(seed),
            "buffer_seed": int(seed),
        }

        out_path = "configs/{}-seed_{}".format(variant_name, seed)
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
