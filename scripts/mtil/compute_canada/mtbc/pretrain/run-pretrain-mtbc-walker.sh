#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate


python ../../sweep_pretrain_mtbc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/pretrain_mtbc.json \
    --out_dir=${HOME}/scratch/data/pretrain_mtbc_main \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/walker_cont \
    --num_heldouts=10 \
    --num_tasks_variants=1,2,4,8,16 \
    --num_samples=2000 \
    --exp_name=walker \
    --samples_per_task \
    --run_time=01:30:00 \
    --num_epochs=5000


python ../../sweep_pretrain_mtbc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/pretrain_mtbc.json \
    --out_dir=${HOME}/scratch/data/pretrain_mtbc_main \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/walker_disc \
    --num_heldouts=10 \
    --num_tasks_variants=1,2,4,8,16 \
    --num_samples=500 \
    --discrete_control \
    --exp_name=walker \
    --samples_per_task \
    --run_time=01:30:00 \
    --num_epochs=5000

chmod +x run_all-*.sh
sbatch run_all-pretrain-mtbc-single_sweep-walker_continuous.sh
sbatch run_all-pretrain-mtbc-single_sweep-walker_discrete.sh
