#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate

python ../sweep_rl.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/single_sweep \
    --run_seed=0 \
    --num_envs=1 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --use_default_env \
    --env_name=ParameterizedPendulum-v0 \
    --exp_name=pendulum

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
    --env_name=ParameterizedPendulum-v0 \
    --exp_name=pendulum

python ../sweep_rl.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/single_sweep \
    --run_seed=0 \
    --num_envs=1 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --use_default_env \
    --env_name=DMCCheetah-v0 \
    --exp_name=cheetah

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
    --env_name=DMCCheetah-v0 \
    --exp_name=cheetah

python ../sweep_rl.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/single_sweep \
    --run_seed=0 \
    --num_envs=1 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --use_default_env \
    --env_name=DMCWalker-v0 \
    --exp_name=walker

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
    --env_name=DMCWalker-v0 \
    --exp_name=walker

chmod +x run_all-*.sh
sbatch run_all-rl-single_sweep-cheetah_continuous.sh
sbatch run_all-rl-single_sweep-cheetah_discrete.sh
sbatch run_all-rl-single_sweep-walker_continuous.sh
sbatch run_all-rl-single_sweep-walker_discrete.sh
sbatch run_all-rl-single_sweep-pendulum_continuous.sh
sbatch run_all-rl-single_sweep-pendulum_discrete.sh
