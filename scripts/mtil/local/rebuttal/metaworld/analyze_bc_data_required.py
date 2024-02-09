import os

record_video = True
device = "gpu:1"
arch_name = "default_network"
expert_data = "scrambling_10-dataset_seed_17"
runs_dir = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/results/bc/{}/{}".format(expert_data, arch_name)
assert os.path.isdir(runs_dir), f"{runs_dir} does not exist."

eval_path = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/evaluate-open_drawer.py"
out_dir = "eval_results"
os.makedirs(out_dir, exist_ok=True)

script_out_path = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/evaluate-{}.sh".format(arch_name)

required_min_data = 0
scrambling_step = 10
env_seed = 0
run_seed = 1
num_episodes = 30
img_res = 84

sh_content = "#!/bin/bash\n"
sh_content += "conda activate jaxl\n"

run_dirs = {}
for run_dir in os.listdir(runs_dir):
    num_data = int(run_dir.split("-")[0].split("num_data_")[-1])
    run_dirs.setdefault(num_data, [])
    run_dirs[num_data].append(run_dir)

for num_data in sorted(list(run_dirs.keys())):
    if num_data < required_min_data:
        continue
    runs_per_num_data = run_dirs[num_data]
    for run_i, run_dir in enumerate(runs_per_num_data):
        print(run_dir)
        sh_content += "python {} --run_path={} --env_seed={} --run_seed={} --scrambling_step={} --num_episodes={} --save_stats={} --img_res={} --device={}{}\n".format(
            eval_path,
            os.path.join(runs_dir, run_dir),
            env_seed,
            run_seed,
            scrambling_step,
            num_episodes,
            os.path.join(out_dir, expert_data, runs_dir.split("/")[-1], "{}-{}".format(run_dir.split("-")[0], run_dir.split("-")[1]), "result.pkl"),
            img_res,
            device,
            " --record_video" if record_video and run_i == 0 else "",
        )

with open(script_out_path, "w+") as f:
    f.writelines(sh_content)
