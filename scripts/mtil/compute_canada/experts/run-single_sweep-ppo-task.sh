#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate

exp_name=bring_ball
env_name=DMCBringBall-v0

python ../sweep_rl.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/single_sweep \
    --run_seed=0 \
    --num_envs=1 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --use_default_env \
    --env_name=${env_name} \
    --exp_name=${exp_name}

python ../sweep_rl.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/single_sweep \
    --run_seed=0 \
    --num_envs=1 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --use_default_env \
    --discrete_control \
    --env_name=${env_name} \
    --exp_name=${exp_name}

chmod +x run_all-*.sh
sbatch run_all-rl-single_sweep-${exp_name}_continuous.sh
sbatch run_all-rl-single_sweep-${exp_name}_discrete.sh
