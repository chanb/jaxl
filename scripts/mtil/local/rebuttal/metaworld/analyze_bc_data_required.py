import os

runs_dir = "results"
assert os.path.isdir(runs_dir), f"{runs_dir} does not exist."

scrambling_step = 10
env_seed = 0
num_episodes = 30

sh_content = "#!/bin/bash\n"
sh_content += "conda activate jaxl\n"
for run_dir in os.listdir(runs_dir):
    
