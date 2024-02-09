#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate


python ../../../../sweep_finetune_mtbc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/finetune_mtbc.json \
    --out_dir=${HOME}/scratch/data/finetune_mtbc_main-8x_source_data-1.5x_target_data \
    --run_seed=0 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/walker_cont \
    --pretrain_dir=${HOME}/scratch/data/pretrain_mtbc_main-8x_source_data/walker-8x_source_data/continuous \
    --num_heldouts=10 \
    --num_samples=3000 \
    --exp_name=walker-8x_source_data-1.5x_target_data \
    --run_time=00:30:00


python ../../../../sweep_finetune_mtbc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/finetune_mtbc.json \
    --out_dir=${HOME}/scratch/data/finetune_mtbc_main-8x_source_data-1.5x_target_data \
    --run_seed=0 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/walker_disc \
    --pretrain_dir=${HOME}/scratch/data/pretrain_mtbc_main-8x_source_data/walker-8x_source_data/discrete \
    --num_heldouts=10 \
    --num_samples=750 \
    --discrete_control \
    --exp_name=walker-8x_source_data-1.5x_target_data \
    --run_time=00:30:00

chmod +x run_all-*.sh
sbatch run_all-finetune-mtbc-single_sweep-walker-8x_source_data-1.5x_target_data_continuous.sh
sbatch run_all-finetune-mtbc-single_sweep-walker-8x_source_data-1.5x_target_data_discrete.sh
