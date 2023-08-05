#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate


python ../../sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/bc.json \
    --out_dir=${HOME}/scratch/data/bc_less_data \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/frozenlake_disc \
    --num_heldouts=10 \
    --num_samples=500 \
    --discrete_control \
    --exp_name=frozenlake

chmod +x run_all-*.sh
sbatch run_all-bc-single_sweep-frozenlake_discrete.sh
