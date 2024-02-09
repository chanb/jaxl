import os

num_seeds = 30

# python gather_expert_data-open_drawer.py --env_seed=1 --num_samples=100000 --subsampling_length=500 --record_video --save_buffer=./test_metaworld-seed_1.gzip

scrambling_step = 10
img_res = 84
subsampling_length = 490
num_samples = 50000
save_path = "./expert_data/test_metaworld-img_res_{}-subsampling_{}-scrambling_{}-num_samples_{}".format(img_res, subsampling_length, scrambling_step, num_samples)

split = 15
for seed in range(num_seeds):
    if seed % split == 0:
        sh_content = ""
        sh_content += "#!/bin/bash\n"
        sh_content += "conda activate jaxl\n"
        sh_content += "mkdir -p expert_data\n"
    sh_content += "python gather_expert_data-open_drawer.py --env_seed={} --num_samples={} --subsampling_length={} --scrambling_step={} --save_buffer={}-seed_{}.gzip --img_res={}\n".format(
        seed, num_samples, subsampling_length, scrambling_step, save_path, seed, img_res
    )

    if (seed + 1) % split == 0:
        with open(
            os.path.join("./gather_expert_data-img_res_{}-scrambling_{}-subsampling_{}-{}.sh".format(img_res, scrambling_step, subsampling_length, int(seed // split))),
            "w+",
        ) as f:
            f.writelines(sh_content)
