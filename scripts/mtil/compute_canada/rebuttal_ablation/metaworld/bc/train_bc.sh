#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate


python ../../../sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/metaworld/bc.json \
    --out_dir=${HOME}/scratch/data/bc-open_drawer-1x_target_data \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/open_drawer \
    --num_heldouts=10 \
    --run_time=07:00:00 \
    --num_samples=250 \
    --exp_name=open_drawer-1x_target_data

python ../../../sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/metaworld/bc.json \
    --out_dir=${HOME}/scratch/data/bc-open_drawer-2x_target_data \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/open_drawer \
    --num_heldouts=10 \
    --run_time=07:00:00 \
    --num_samples=500 \
    --exp_name=open_drawer-2x_target_data

python ../../../sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/metaworld/bc.json \
    --out_dir=${HOME}/scratch/data/bc-open_drawer-4x_target_data \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/open_drawer \
    --num_heldouts=10 \
    --run_time=07:00:00 \
    --num_samples=1000 \
    --exp_name=open_drawer-4x_target_data

python ../../../sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/metaworld/bc.json \
    --out_dir=${HOME}/scratch/data/bc-open_drawer-8x_target_data \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/open_drawer \
    --num_heldouts=10 \
    --run_time=07:00:00 \
    --num_samples=2000 \
    --exp_name=open_drawer-8x_target_data

python ../../../sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/metaworld/bc.json \
    --out_dir=${HOME}/scratch/data/bc-open_drawer-16x_target_data \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/open_drawer \
    --num_heldouts=10 \
    --run_time=07:00:00 \
    --num_samples=4000 \
    --exp_name=open_drawer-16x_target_data

chmod +x run_all-*.sh
sbatch run_all-bc-single_sweep-open_drawer-1x_target_data_continuous.sh
sbatch run_all-bc-single_sweep-open_drawer-2x_target_data_continuous.sh
sbatch run_all-bc-single_sweep-open_drawer-4x_target_data_continuous.sh
sbatch run_all-bc-single_sweep-open_drawer-8x_target_data_continuous.sh
sbatch run_all-bc-single_sweep-open_drawer-16x_target_data_continuous.sh
