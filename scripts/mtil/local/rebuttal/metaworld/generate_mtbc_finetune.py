import json
import os


eval_seeds = list(range(20, 30))
num_data = 500
expert_datasets_dir = (
    "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/expert_data/"
)
mtbc_template = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/configs/finetune_mtbc_template.json"

save_path = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/results/mtbc_finetune"
arch_name = "large_network"

mtbc_pretrain_dir = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/results/mtbc_pretrain/{}".format(
    arch_name
)
config_out_path = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/configs/finetune_mtbc-{}".format(
    arch_name
)
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


pretrained_configs = {}
for pretrain_path, _, filenames in os.walk(mtbc_pretrain_dir):
    for filename in filenames:
        if not filename.endswith("config.json"):
            continue

        path_info = pretrain_path.split("/")
        num_tasks = int(path_info[-2].split("num_tasks_")[-1])
        pretrain_model_seed = int(path_info[-1].split("-")[0].split("model_seed_")[-1])

        pretrained_configs.setdefault(num_tasks, {})
        pretrained_configs[num_tasks] = (pretrain_model_seed, pretrain_path)

for task_i, num_tasks in enumerate(pretrained_configs):
    device = device_range[task_i % num_devices]
    for eval_seed in eval_seeds:
        script_out_path = os.path.join(
            script_out_dir,
            "pretrain_mtbc-{}-eval_seed_{}-num_tasks_{}-pretrain_model_seed_{}.sh".format(
                arch_name, eval_seed, num_tasks, pretrain_model_seed
            ),
        )
        expert_dataset = os.path.join(
            expert_datasets_dir, "test_metaworld-seed_{}.gzip".format(eval_seed)
        )

        sh_content = "#!/bin/bash\n"
        sh_content += "conda activate jaxl\n"

        template["learner_config"]["buffer_configs"][0]["load_buffer"] = expert_dataset
        template["learner_config"]["buffer_configs"][0]["set_size"] = num_data
        template["learner_config"]["losses"][0] = "gaussian"
        template["logging_config"]["save_path"] = os.path.join(
            save_path,
            arch_name,
            f"eval_seed_{eval_seed}/num_tasks_{num_tasks}/pretrained_model_seed_{pretrain_model_seed}",
        )
        template["logging_config"]["experiment_name"] = f"model_seed_42"
        modify_network(template, arch_name)

        curr_config_path = os.path.join(
            config_out_path,
            f"eval_seed_{eval_seed}" f"num_tasks_{num_tasks}",
            "finetune_mtbc-pretrain_model_seed_{}.json".format(pretrain_model_seed),
        )
        with open(curr_config_path, "w+") as f:
            json.dump(template, f)

        sh_content += "python {} --config_path={} --device=gpu:{} \n".format(
            main_path, curr_config_path, device
        )

    with open(script_out_path, "w+") as f:
        f.writelines(sh_content)
