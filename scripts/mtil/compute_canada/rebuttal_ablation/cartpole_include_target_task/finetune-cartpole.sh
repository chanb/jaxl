#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate


python ../sweep_finetune_mtbc-include_pretrain_seed.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/finetune_mtbc.json \
    --out_dir=${HOME}/scratch/data/finetune_mtbc_main-include_representation \
    --run_seed=0 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/cartpole_cont \
    --pretrain_dir=${HOME}/scratch/data/pretrain_mtbc_main-include_representation/cartpole/continuous \
    --num_samples=500 \
    --exp_name=cartpole

chmod +x run_all-*.sh
sbatch run_all-finetune-mtbc-single_sweep-cartpole_continuous.sh
