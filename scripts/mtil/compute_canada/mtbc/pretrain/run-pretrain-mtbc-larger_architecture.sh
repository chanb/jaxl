#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate


python ../../sweep_pretrain_mtbc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/mtbc_architecture/pretrain_mtbc.json \
    --out_dir=${HOME}/scratch/data/pretrain_mtbc-larger_architecture \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/pendulum_cont \
    --num_heldouts=10 \
    --num_tasks_variants=1,2,4,8,16 \
    --num_samples=1000 \
    --samples_per_task \
    --exp_name=pendulum-larger_architecture \
    --num_epochs=5000 \
    --run_time=02:30:00


python ../../sweep_pretrain_mtbc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/mtbc_architecture/pretrain_mtbc.json \
    --out_dir=${HOME}/scratch/data/pretrain_mtbc-larger_architecture \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/cartpole_cont \
    --num_heldouts=10 \
    --num_tasks_variants=1,2,4,8,16 \
    --num_samples=10000 \
    --samples_per_task \
    --exp_name=cartpole-larger_architecture \
    --num_epochs=5000 \
    --run_time=02:30:00


python ../../sweep_pretrain_mtbc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/mtbc_architecture/pretrain_mtbc.json \
    --out_dir=${HOME}/scratch/data/pretrain_mtbc-larger_architecture \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/walker_disc \
    --num_heldouts=10 \
    --num_tasks_variants=1,2,4,8,16 \
    --num_samples=500 \
    --samples_per_task \
    --exp_name=walker-larger_architecture \
    --discrete_control \
    --num_epochs=5000 \
    --run_time=02:30:00

chmod +x run_all-*.sh
sbatch run_all-pretrain-mtbc-single_sweep-pendulum-larger_architecture_continuous.sh
sbatch run_all-pretrain-mtbc-single_sweep-cartpole-larger_architecture_continuous.sh
sbatch run_all-pretrain-mtbc-single_sweep-walker-larger_architecture_discrete.sh