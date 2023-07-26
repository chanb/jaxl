#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate

python sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/bc.json \
    --out_dir=${HOME}/scratch/data/single_sweep_bc \
    --run_seed=0 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/pendulum/continuous \
    --dataset_variant=0 \
    --dataset_variant=1 \
    --dataset_variant=2 \
    --exp_name=pendulum

python sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/bc.json \
    --out_dir=${HOME}/scratch/data/single_sweep_bc \
    --run_seed=0 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/pendulum/discrete \
    --dataset_variant=0 \
    --dataset_variant=1 \
    --dataset_variant=2 \
    --discrete_control \
    --exp_name=pendulum

python sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/bc.json \
    --out_dir=${HOME}/scratch/data/single_sweep_bc \
    --run_seed=0 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/cheetah/continuous \
    --dataset_variant=0 \
    --dataset_variant=1 \
    --dataset_variant=2 \
    --exp_name=cheetah

python sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/bc.json \
    --out_dir=${HOME}/scratch/data/single_sweep_bc \
    --run_seed=0 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/cheetah/discrete \
    --dataset_variant=0 \
    --dataset_variant=1 \
    --dataset_variant=2 \
    --discrete_control \
    --exp_name=cheetah

python sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/bc.json \
    --out_dir=${HOME}/scratch/data/single_sweep_bc \
    --run_seed=0 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/walker/continuous \
    --dataset_variant=0 \
    --dataset_variant=1 \
    --dataset_variant=2 \
    --exp_name=walker

python sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/bc.json \
    --out_dir=${HOME}/scratch/data/single_sweep_bc \
    --run_seed=0 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/walker/discrete \
    --dataset_variant=0 \
    --dataset_variant=1 \
    --dataset_variant=2 \
    --discrete_control \
    --exp_name=walker

chmod +x run_all-*.sh
sbatch run_all-single_sweep-cheetah_continuous.sh
sbatch run_all-single_sweep-cheetah_discrete.sh
sbatch run_all-single_sweep-walker_continuous.sh
sbatch run_all-single_sweep-walker_discrete.sh
sbatch run_all-single_sweep-pendulum_continuous.sh
sbatch run_all-single_sweep-pendulum_discrete.sh
