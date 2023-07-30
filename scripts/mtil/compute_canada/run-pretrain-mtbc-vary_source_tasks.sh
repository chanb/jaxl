#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate


python sweep_pretrain_mtbc-vary_source_tasks.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/pretrain_mtbc.json \
    --out_dir=${HOME}/scratch/data/pretrain_mtbc-vary_source_tasks \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/pendulum_cont \
    --num_heldouts=10 \
    --num_tasks_variants=1,2,4,8,16 \
    --num_variations=5 \
    --num_samples=1600 \
    --exp_name=pendulum


python sweep_pretrain_mtbc-vary_source_tasks.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/pretrain_mtbc.json \
    --out_dir=${HOME}/scratch/data/pretrain_mtbc-vary_source_tasks \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/pendulum_disc \
    --num_heldouts=10 \
    --num_tasks_variants=1,2,4,8,16 \
    --num_variations=5 \
    --num_samples=4800 \
    --discrete_control \
    --exp_name=pendulum

chmod +x run_all-*.sh
sbatch run_all-pretrain-mtbc-vary_source_tasks-single_sweep-pendulum_continuous.sh
sbatch run_all-pretrain-mtbc-vary_source_tasks-single_sweep-pendulum_discrete.sh
