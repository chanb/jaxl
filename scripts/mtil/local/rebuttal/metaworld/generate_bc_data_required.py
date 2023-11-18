import json
import os


num_data_to_check = [100, 250, 500, 1000, 5000, 10000]
expert_dataset = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/expert_data/test_metaworld-seed_0.gzip"
bc_template = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/configs/bc_template.json"

save_path = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/results"
experiment_name = "bc_data_ablation-smaller_network"

config_out_path = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/configs/bc_data_ablation-smaller_network"
script_out_path = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/bc_data_ablation-smaller_network.sh"
main_path = "/home/bryanpu1/projects/jaxl/jaxl/main.py"
device = "gpu:1"

os.makedirs(config_out_path, exist_ok=True)
assert os.path.isfile(bc_template), f"{bc_template} is not a file"
with open(bc_template, "r") as f:
    template = json.load(f)


sh_content = "#!/bin/bash\n"
sh_content += "conda activate jaxl\n"
for num_data in num_data_to_check:
    template["learner_config"]["buffer_config"]["load_buffer"] = expert_dataset
    template["learner_config"]["buffer_config"]["set_size"] = num_data
    template["learner_config"]["losses"][0] = "gaussian"
    template["logging_config"]["save_path"] = save_path
    template["logging_config"]["experiment_name"] = experiment_name

    curr_config_path = os.path.join(config_out_path, "bc_data_ablation-{}.json".format(num_data))
    with open(curr_config_path, "w+") as f:
        json.dump(template, f)

    sh_content += "python {} --config_path={} --device={} \n".format(main_path, curr_config_path, device)

with open(script_out_path, "w+") as f:
    f.writelines(sh_content)
