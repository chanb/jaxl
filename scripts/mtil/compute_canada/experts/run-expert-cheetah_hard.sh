#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate

python ../sweep_rl.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/experts \
    --run_seed=0 \
    --num_envs=100 \
    --num_runs=1 \
    --checkpoint_interval=250 \
    --num_epochs=500 \
    --run_time=02:30:00 \
    --hyperparam_set=cheetah_cont \
    --env_name=DMCCheetah-v0 \
    --exp_name=cheetah_hard \
    --env_file=/home/chanb/src/jaxl/jaxl/envs/dmc/configs/cheetah_hard.json

python ../sweep_rl.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/experts \
    --run_seed=0 \
    --num_envs=100 \
    --num_runs=1 \
    --checkpoint_interval=250 \
    --num_epochs=500 \
    --run_time=02:30:00 \
    --hyperparam_set=cheetah_disc \
    --discrete_control \
    --env_name=DMCCheetah-v0 \
    --exp_name=cheetah_hard \
    --env_file=/home/chanb/src/jaxl/jaxl/envs/dmc/configs/cheetah_hard.json

chmod +x run_all-*.sh
sbatch run_all-rl-cheetah_cont-cheetah_hard_continuous.sh
sbatch run_all-rl-cheetah_disc-cheetah_hard_discrete.sh
