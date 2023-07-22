#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate

python sweep_rl.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/experts \
    --run_seed=0 \
    --env_seed=42 \
    --hyperparam_set=pendulum_cont \
    --env_name=ParameterizedPendulum-v0 \
    --exp_name=pendulum

python sweep_rl.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/experts \
    --run_seed=0 \
    --env_seed=42 \
    --hyperparam_set=pendulum_disc \
    --discrete_control \
    --env_name=ParameterizedPendulum-v0 \
    --exp_name=pendulum

python sweep_rl.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/experts \
    --run_seed=0 \
    --env_seed=42 \
    --hyperparam_set=cheetah_cont \
    --env_name=DMCCheetah-v0 \
    --exp_name=cheetah

python sweep_rl.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/experts \
    --run_seed=0 \
    --env_seed=42 \
    --hyperparam_set=cheetah_disc \
    --discrete_control \
    --env_name=DMCCheetah-v0 \
    --exp_name=cheetah

python sweep_rl.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/experts \
    --run_seed=0 \
    --env_seed=42 \
    --hyperparam_set=walker_cont \
    --env_name=DMCWalker-v0 \
    --exp_name=walker

python sweep_rl.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/experts \
    --run_seed=0 \
    --env_seed=42 \
    --hyperparam_set=walker_disc \
    --discrete_control \
    --env_name=DMCWalker-v0 \
    --exp_name=walker

chmod +x run_all-*.sh
sbatch run_all-cheetah_cont-cheetah_continuous.sh
sbatch run_all-cheetah_disc-cheetah_discrete.sh
sbatch run_all-walker_cont-walker_continuous.sh
sbatch run_all-walker_disc-walker_discrete.sh
sbatch run_all-pendulum_cont-pendulum_continuous.sh
sbatch run_all-pendulum_disc-pendulum_discrete.sh
