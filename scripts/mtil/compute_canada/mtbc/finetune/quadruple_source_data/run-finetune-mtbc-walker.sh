#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate


python ../../../sweep_finetune_mtbc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/finetune_mtbc.json \
    --out_dir=${HOME}/scratch/data/finetune_mtbc_main-quadruple_source_data \
    --run_seed=0 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/walker_cont \
    --pretrain_dir=${HOME}/scratch/data/pretrain_mtbc_main-quadruple_source_data/walker-quadruple_source_data/continuous \
    --num_heldouts=10 \
    --num_samples=2000 \
    --exp_name=walker-quadruple_source_data


python ../../../sweep_finetune_mtbc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/finetune_mtbc.json \
    --out_dir=${HOME}/scratch/data/finetune_mtbc_main-quadruple_source_data \
    --run_seed=0 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/walker_disc \
    --pretrain_dir=${HOME}/scratch/data/pretrain_mtbc_main-quadruple_source_data/walker-quadruple_source_data/discrete \
    --num_heldouts=10 \
    --num_samples=500 \
    --discrete_control \
    --exp_name=walker-quadruple_source_data

chmod +x run_all-*.sh
sbatch run_all-finetune-mtbc-single_sweep-walker-quadruple_source_data_continuous.sh
sbatch run_all-finetune-mtbc-single_sweep-walker-quadruple_source_data_discrete.sh
