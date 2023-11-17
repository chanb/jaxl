#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate


python ../sweep_pretrain_mtbc-include_pretrain_seed.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/pretrain_mtbc.json \
    --out_dir=${HOME}/scratch/data/pretrain_mtbc_main-include_target_task \
    --num_heldouts=0 \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/cartpole_cont \
    --num_tasks_variants=1,2,4,8,16 \
    --offset=4 \
    --num_samples=500 \
    --samples_per_task \
    --exp_name=cartpole \
    --num_epochs=5000 \
    --run_time=01:30:00

chmod +x run_all-*.sh
sbatch run_all-pretrain-mtbc-single_sweep-cartpole_continuous-offset_4.sh
