#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate


python ../../sweep_finetune_mtbc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/mtbc_architecture/finetune_mtbc.json \
    --out_dir=${HOME}/scratch/data/finetune_mtbc-larger_architecture \
    --run_seed=0 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/pendulum_cont \
    --num_heldouts=10 \
    --pretrain_dir=${HOME}/scratch/data/pretrain_mtbc-larger_architecture/pendulum-larger_architecture/continuous \
    --num_samples=1000 \
    --exp_name=pendulum


python ../../sweep_finetune_mtbc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/mtbc_architecture/finetune_mtbc.json \
    --out_dir=${HOME}/scratch/data/finetune_mtbc-larger_architecture \
    --run_seed=0 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/cartpole_cont \
    --num_heldouts=10 \
    --pretrain_dir=${HOME}/scratch/data/pretrain_mtbc-larger_architecture/cartpole-larger_architecture/continuous \
    --num_samples=10000 \
    --exp_name=cartpole


python ../../sweep_finetune_mtbc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/mtbc_architecture/finetune_mtbc.json \
    --out_dir=${HOME}/scratch/data/finetune_mtbc-larger_architecture \
    --run_seed=0 \
    --num_runs=1 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/walker_disc \
    --num_heldouts=10 \
    --pretrain_dir=${HOME}/scratch/data/pretrain_mtbc-larger_architecture/walker-larger_architecture/discrete \
    --discrete_control \
    --num_samples=500 \
    --exp_name=walker

chmod +x run_all-*.sh
sbatch run_all-finetune-mtbc-single_sweep-pendulum-larger_architecture_continuous.sh
sbatch run_all-finetune-mtbc-single_sweep-cartpole-larger_architecture_continuous.sh
sbatch run_all-finetune-mtbc-single_sweep-walker-larger_architecture_discrete.sh
