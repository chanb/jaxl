#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate

python ../generate_metaworld_data.py \
    --main_path=${JAXL_PATH}/jaxl/scripts/mtil/compute_canada/rebuttal_ablation/gather_metaworld_data.py \
    --out_dir=${HOME}/scratch/data/expert_data \
    --run_seed=0 \
    --num_envs=9999 \
    --num_samples=100000 \
    --subsampling_lengths=500 \
    --run_time=00:30:00 \
    --max_episode_length=500 \
    --exp_name=open_drawer

chmod +x run_all-*.sh
sbatch run_all-generate_metaworld_data-open_drawer.sh
