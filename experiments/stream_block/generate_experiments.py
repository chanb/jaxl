from itertools import product

import json
import os

template_path = "/Users/chanb/research/personal/jaxl/experiments/stream_block/configs/ablation_template.json"
script_dir = "/Users/chanb/research/personal/jaxl/experiments/stream_block/ablation/"
log_dir = "/Users/chanb/research/personal/jaxl/experiments/stream_block/logs"

ablations = {
    "num_base_classes": [10240, 102400, 1024000],
    "num_clusters": [1024, 16, 128, 512],
    "num_abstract_classes": [32, 2, 128, 512],
    "input_noise_std": [0.0, 0.1, 0.5],
    "zipf_exp": [0.0, 1.0, 2.0],
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
