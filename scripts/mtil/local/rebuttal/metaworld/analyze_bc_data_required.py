import os

arch_name = "default_network"
runs_dir = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/results/{}".format(arch_name)
assert os.path.isdir(runs_dir), f"{runs_dir} does not exist."

eval_path = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/evaluate-open_drawer.py"
out_dir = "eval_results"
os.makedirs(out_dir, exist_ok=True)

script_out_path = "/home/bryanpu1/projects/jaxl/scripts/mtil/local/rebuttal/metaworld/evaluate-{}.sh".format(arch_name)

scrambling_step = 10
env_seed = 0
num_episodes = 30

sh_content = "#!/bin/bash\n"
sh_content += "conda activate jaxl\n"
for run_dir in os.listdir(runs_dir):
    print(run_dir)
    sh_content += "python {} --run_path={} --env_seed={} --scrambling_step={} --num_episodes={} --save_stats={}\n".format(
        eval_path,
        os.path.join(runs_dir, run_dir),
        env_seed,
        scrambling_step,
        num_episodes,
        os.path.join(out_dir, runs_dir.split("/")[-1], "num_data_{}".format(run_dir.split("-")[0]))
    )

with open(script_out_path, "w+") as f:
    f.writelines(sh_content)
