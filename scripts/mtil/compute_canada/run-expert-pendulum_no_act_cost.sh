#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate

python sweep_rl.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/experts \
    --run_seed=0 \
    --num_envs=100 \
    --num_runs=1 \
    --checkpoint_interval=250 \
    --num_epochs=500 \
    --run_time=02:30:00 \
    --hyperparam_set=pendulum_cont \
    --env_name=ParameterizedPendulum-v0 \
    --exp_name=pendulum_no_act_cost

python sweep_rl.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/ppo.json \
    --out_dir=${HOME}/scratch/data/experts \
    --run_seed=0 \
    --num_envs=100 \
    --num_runs=1 \
    --checkpoint_interval=250 \
    --num_epochs=500 \
    --run_time=02:30:00 \
    --hyperparam_set=pendulum_disc \
    --discrete_control \
    --env_name=ParameterizedPendulum-v0 \
    --exp_name=pendulum_no_act_cost

chmod +x run_all-*.sh
sbatch run_all-pendulum_cont-pendulum_no_act_cost_continuous.sh
sbatch run_all-pendulum_disc-pendulum_no_act_cost_discrete.sh
