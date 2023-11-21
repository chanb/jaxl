import os

record_video = True
device = "gpu:2"
arch_name = "large_network"
runs_dir = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/results/bc/{}".format(arch_name)
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
run_dirs = {
    int(run_dir.split("-")[1]): run_dir for run_dir in os.listdir(runs_dir)
}
for num_data in sorted(list(run_dirs.keys())):
    if num_data < required_min_data:
        continue
    run_dir = run_dirs[num_data]
    print(run_dir)
    sh_content += "python {} --run_path={} --env_seed={} --run_seed={} --scrambling_step={} --num_episodes={} --save_stats={} --img_res={} --device={}{}\n".format(
        eval_path,
        os.path.join(runs_dir, run_dir),
        env_seed,
        run_seed,
        scrambling_step,
        num_episodes,
        os.path.join(out_dir, runs_dir.split("/")[-1], "num_data_{}".format(run_dir.split("-")[1]), "result.pkl"),
        img_res,
        device,
        " --record_video" if record_video else "",
    )

with open(script_out_path, "w+") as f:
    f.writelines(sh_content)
