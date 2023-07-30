#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate


python sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/bc.json \
    --out_dir=${HOME}/scratch/data/bc_main \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/walker_cont \
    --dataset_variant=0 \
    --dataset_variant=1 \
    --dataset_variant=2 \
    --dataset_variant=3 \
    --dataset_variant=4 \
    --dataset_variant=5 \
    --dataset_variant=6 \
    --dataset_variant=7 \
    --dataset_variant=8 \
    --dataset_variant=9 \
    --num_samples=7500 \
    --exp_name=walker


python sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/bc.json \
    --out_dir=${HOME}/scratch/data/bc_main \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/walker_disc \
    --dataset_variant=0 \
    --dataset_variant=1 \
    --dataset_variant=2 \
    --dataset_variant=3 \
    --dataset_variant=4 \
    --dataset_variant=5 \
    --dataset_variant=6 \
    --dataset_variant=7 \
    --dataset_variant=8 \
    --dataset_variant=9 \
    --num_samples=2500 \
    --discrete_control \
    --exp_name=walker

chmod +x run_all-*.sh
sbatch run_all-single_sweep-walker_continuous.sh
sbatch run_all-single_sweep-walker_discrete.sh
