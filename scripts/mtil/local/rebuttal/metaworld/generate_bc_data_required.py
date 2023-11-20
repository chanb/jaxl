import json
import os


num_data_to_check = [100, 250, 500, 1000, 5000, 10000, 15000, 20000, 25000, 50000]
expert_dataset = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/expert_data/test_metaworld-seed_0.gzip"
bc_template = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/configs/bc_template.json"

save_path = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/results/bc"
arch_name = "large_network"

config_out_path = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/configs/bc_data_ablation-{}".format(arch_name)
script_out_path = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/bc_data_ablation-{}.sh".format(arch_name)
main_path = "/home/bryanpu1/projects/jaxl/jaxl/main.py"
device = "gpu:1"

os.makedirs(config_out_path, exist_ok=True)
assert os.path.isfile(bc_template), f"{bc_template} is not a file"
with open(bc_template, "r") as f:
    template = json.load(f)

def modify_network(template, arch_name):
    if arch_name == "default_network":
        template["model_config"]["features"] = [32, 32]
        template["model_config"]["kernel_sizes"] = [[3, 3], [3, 3]]
        template["model_config"]["layers"] = [32, 32]
    elif arch_name == "small_network":
        template["model_config"]["features"] = [16, 16]
        template["model_config"]["kernel_sizes"] = [[3, 3], [3, 3]]
        template["model_config"]["layers"] = [32]
    elif arch_name == "medium_network":
        template["model_config"]["features"] = [32, 32]
        template["model_config"]["kernel_sizes"] = [[3, 3], [3, 3]]
        template["model_config"]["layers"] = [32]
    elif arch_name == "large_network":
        template["model_config"]["features"] = [64, 64]
        template["model_config"]["kernel_sizes"] = [[3, 3], [3, 3]]
        template["model_config"]["layers"] = [32, 32]
    else:
        assert 0


sh_content = "#!/bin/bash\n"
sh_content += "conda activate jaxl\n"
for num_data in num_data_to_check:
    template["learner_config"]["buffer_config"]["load_buffer"] = expert_dataset
    template["learner_config"]["buffer_config"]["set_size"] = num_data
    template["learner_config"]["losses"][0] = "gaussian"
    template["logging_config"]["save_path"] = os.path.join(save_path, arch_name)
    template["logging_config"]["experiment_name"] = f"num_data-{num_data}"

    modify_network(template, arch_name)

    curr_config_path = os.path.join(config_out_path, "bc_data_ablation-{}.json".format(num_data))
    with open(curr_config_path, "w+") as f:
        json.dump(template, f)

    sh_content += "python {} --config_path={} --device={} \n".format(main_path, curr_config_path, device)

with open(script_out_path, "w+") as f:
    f.writelines(sh_content)
