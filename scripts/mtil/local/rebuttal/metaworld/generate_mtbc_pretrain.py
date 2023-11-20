import json
import os


num_runs = 5
num_taskss = [1, 2, 4, 8, 16]
num_data = 500
expert_datasets_dir = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/expert_data/"
mtbc_template = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/configs/pretrain_mtbc_template.json"

save_path = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/results/mtbc_pretrain"
arch_name = "large_network"

config_out_path = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/configs/pretrain_mtbc-{}".format(arch_name)
script_out_dir = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/"
main_path = "/home/bryanpu1/projects/jaxl/jaxl/main.py"
num_devices = 4
device_range = list(range(num_devices))

os.makedirs(config_out_path, exist_ok=True)
assert os.path.isfile(mtbc_template), f"{mtbc_template} is not a file"
with open(mtbc_template, "r") as f:
    template = json.load(f)

def modify_network(template, arch_name):
    if arch_name == "default_network":
        template["model_config"]["encoder"]["features"] = [32, 32]
        template["model_config"]["encoder"]["kernel_sizes"] = [[3, 3], [3, 3]]
        template["model_config"]["encoder"]["layers"] = [32]
        template["model_config"]["encoder_dim"] = [32]
    elif arch_name == "large_network":
        template["model_config"]["encoder"]["features"] = [64, 64]
        template["model_config"]["encoder"]["kernel_sizes"] = [[3, 3], [3, 3]]
        template["model_config"]["encoder"]["layers"] = [32]
        template["model_config"]["encoder_dim"] = [32]
    else:
        assert 0

for task_i, num_tasks in enumerate(num_taskss):
    script_out_path = os.path.join(script_out_dir, "pretrain_mtbc-{}-num_tasks_{}.sh".format(arch_name, num_tasks))
    device = device_range[task_i % num_devices]
    sh_content = "#!/bin/bash\n"
    sh_content += "conda activate jaxl\n"
    for model_seed in range(num_runs):
        template["learner_config"]["buffer_configs"] = []
        for buffer_seed in range(num_tasks):
            expert_dataset = os.path.join(expert_datasets_dir, "test_metaworld-seed_{}.gzip".format(buffer_seed))
            curr_buffer_config = {
                "load_buffer": expert_dataset,
                "set_size": num_data,
                "buffer_type": "memory_efficient",
            }
        template["learner_config"]["losses"][0] = "gaussian"
        template["logging_config"]["save_path"] = os.path.join(save_path, arch_name, f"num_tasks_{num_tasks}")
        template["logging_config"]["experiment_name"] = f"model_seed_{model_seed}"

        modify_network(template, arch_name)

        curr_config_path = os.path.join(config_out_path, "mtbc_data_ablation-model_seed_{}-num_tasks_{}.json".format(model_seed, num_tasks))
        with open(curr_config_path, "w+") as f:
            json.dump(template, f)

        sh_content += "python {} --config_path={} --device=gpu:{} \n".format(main_path, curr_config_path, device)

    with open(script_out_path, "w+") as f:
        f.writelines(sh_content)
