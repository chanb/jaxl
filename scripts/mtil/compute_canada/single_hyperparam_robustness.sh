#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate

python single_hyperparam_robustness.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/single_hyperparam_robustness \
    --run_seed=0 \
    --env_name=ParameterizedPendulum-v0 \
    --exp_name=pendulum_cont \
    --num_envs=6 \
    --num_runs=3

python single_hyperparam_robustness.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/single_hyperparam_robustness \
    --run_seed=0 \
    --env_name=ParameterizedPendulum-v0 \
    --exp_name=pendulum_disc \
    --discrete_control \
    --num_envs=6 \
    --num_runs=3

python single_hyperparam_robustness.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/single_hyperparam_robustness \
    --run_seed=0 \
    --env_name=DMCCheetah-v0 \
    --exp_name=cheetah_cont \
    --num_envs=6 \
    --num_runs=3

python single_hyperparam_robustness.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/single_hyperparam_robustness \
    --run_seed=0 \
    --env_name=DMCCheetah-v0 \
    --exp_name=cheetah_disc \
    --discrete_control \
    --num_envs=6 \
    --num_runs=3

python single_hyperparam_robustness.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/single_hyperparam_robustness \
    --run_seed=0 \
    --env_name=DMCWalker-v0 \
    --exp_name=walker_cont \
    --num_envs=6 \
    --num_runs=3

python single_hyperparam_robustness.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/single_hyperparam_robustness \
    --run_seed=0 \
    --env_name=DMCWalker-v0 \
    --exp_name=walker_disc \
    --discrete_control \
    --num_envs=6 \
    --num_runs=3

chmod +x run_all-*.sh
sbatch run_all-single_hyperparam_robustness-cheetah_disc.sh
sbatch run_all-single_hyperparam_robustness-cheetah_cont.sh
sbatch run_all-single_hyperparam_robustness-walker_disc.sh
sbatch run_all-single_hyperparam_robustness-walker_cont.sh
sbatch run_all-single_hyperparam_robustness-pendulum_disc.sh
sbatch run_all-single_hyperparam_robustness-pendulum_cont.sh
