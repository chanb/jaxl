from itertools import product

import json
import os

template_path = "/Users/chanb/research/personal/jaxl/experiments/stream_block_biuniform/configs/ablation_template.json"
script_dir = "/Users/chanb/research/personal/jaxl/experiments/stream_block_biuniform/ablation/"
log_dir = "/Users/chanb/research/personal/jaxl/experiments/stream_block_biuniform/logs"

ablations = {
    "high_prob": [0.6, 0.75, 0.8, 0.9],
    "num_high_prob_classes": [2, 16, 64],
    "num_low_prob_classes": [16, 64, 128, 512],
    "abstraxt_class": [0, 1],
    "iid_context": [0, 1, "single_tower"],
}

script_paths = []
for ablation_name, values in ablations.items():
    ablation_dir = os.path.join(script_dir, ablation_name)
    curr_script_path = os.path.join(script_dir, f"{ablation_name}.sh")
    os.makedirs(ablation_dir, exist_ok=True)

    sh_content = "#!/bin/bash\n"
    sh_content = "source /Users/chanb/research/personal/jaxl/.venv/bin/activate\n\n"
    for value in values:
        curr_exp_name = f"{ablation_name}_{value}"
        curr_config_path = os.path.join(ablation_dir, f"{curr_exp_name}.json")

        # Write experiment
        config_dict = json.load(open(template_path, "r"))

        if ablation_name == "iid_context" and value == "single_tower":
            config_dict["model_config"]["single_tower"] = True
        else:
            config_dict["learner_config"]["dataset_config"]["dataset_kwargs"][
                ablation_name
            ] = value
        config_dict["logging_config"]["save_path"] = os.path.join(
            log_dir, ablation_name
        )
        config_dict["logging_config"]["experiment_name"] = curr_exp_name

        json.dump(
            config_dict,
            open(curr_config_path, "w"),
        )

        sh_content += "python /Users/chanb/research/personal/jaxl/main.py --config_path={}\n".format(
            curr_config_path
        )

    with open(curr_script_path, "w") as f:
        f.write(sh_content)

    script_paths.append(curr_script_path)


run_all_content = "#!/bin/bash\n"
for curr_script_path in script_paths:
    run_all_content += "chmod +x {}\n".format(curr_script_path)

run_all_content += " && ".join(script_paths)
with open(os.path.join(script_dir, "run_all.sh"), "w") as f:
    f.write(run_all_content)
