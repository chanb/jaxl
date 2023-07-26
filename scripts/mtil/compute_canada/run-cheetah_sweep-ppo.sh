#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate

python sweep_rl.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/cheetah_sweep \
    --run_seed=0 \
    --num_envs=6 \
    --num_runs=3 \
    --hyperparam_set=cheetah_sweep \
    --use_default_env \
    --discrete_control \
    --env_name=DMCCheetah-v0 \
    --exp_name=cheetah

chmod +x run_all-*.sh
sbatch run_all-cheetah_sweep-cheetah_discrete.sh
