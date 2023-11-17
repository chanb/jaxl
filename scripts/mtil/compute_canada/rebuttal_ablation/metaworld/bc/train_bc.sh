#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate


python ../../sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/metaworld/bc.json \
    --out_dir=${HOME}/scratch/data/bc_less_data \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/open_drawer_cont \
    --num_heldouts=10 \
    --num_samples=500 \
    --exp_name=open_drawer

chmod +x run_all-*.sh
sbatch run_all-bc-single_sweep-open_drawer_continuous.sh
