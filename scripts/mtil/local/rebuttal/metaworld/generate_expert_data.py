import os

num_seeds = 30

# python gather_expert_data-open_drawer.py --env_seed=1 --num_samples=100000 --subsampling_length=500 --record_video --save_buffer=./test_metaworld-seed_1.gzip

scrambling_step = 5
img_res = 84
subsampling_length = 250
save_path = "./expert_data/test_metaworld-img_res_{}-subsampling_{}-scrambling_{}".format(img_res, subsampling_length, scrambling_step)


sh_content = ""
sh_content += "#!/bin/bash\n"
sh_content += "conda activate jaxl\n"
sh_content += "mkdir -p expert_data\n"
for seed in range(num_seeds):
    sh_content += "python gather_expert_data-open_drawer.py --env_seed={} --num_samples=100000 --subsampling_length={} --scrambling_step={} --save_buffer={}-seed_{}.gzip --img_res={}\n".format(
        seed, subsampling_length, scrambling_step, save_path, seed, img_res
    )

with open(
    os.path.join(f"./gather_expert_data.sh"),
    "w+",
) as f:
    f.writelines(sh_content)
