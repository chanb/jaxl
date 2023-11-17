#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate


python ../../../sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/bc.json \
    --out_dir=${HOME}/scratch/data/bc-8x_target_data \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/cartpole_cont \
    --num_heldouts=10 \
    --num_samples=4000 \
    --exp_name=cartpole-8x_target_data


python ../../../sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/bc.json \
    --out_dir=${HOME}/scratch/data/bc-8x_target_data \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/frozenlake_disc \
    --num_heldouts=10 \
    --num_samples=4000 \
    --discrete_control \
    --exp_name=frozenlake-8x_target_data


python ../../../sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/bc.json \
    --out_dir=${HOME}/scratch/data/bc-8x_target_data \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/cheetah_cont \
    --num_heldouts=10 \
    --num_samples=8000 \
    --exp_name=cheetah-8x_target_data


python ../../../sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/bc.json \
    --out_dir=${HOME}/scratch/data/bc-8x_target_data \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/cheetah_disc \
    --num_heldouts=10 \
    --num_samples=8000 \
    --discrete_control \
    --exp_name=cheetah-8x_target_data


python ../../../sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/bc.json \
    --out_dir=${HOME}/scratch/data/bc-8x_target_data \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/pendulum_cont \
    --num_heldouts=10 \
    --num_samples=8000 \
    --exp_name=pendulum-8x_target_data


python ../../../sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/bc.json \
    --out_dir=${HOME}/scratch/data/bc-8x_target_data \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/pendulum_disc \
    --num_heldouts=10 \
    --num_samples=8000 \
    --discrete_control \
    --exp_name=pendulum-8x_target_data

python ../../../sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/bc.json \
    --out_dir=${HOME}/scratch/data/bc-8x_target_data \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/walker_cont \
    --num_heldouts=10 \
    --num_samples=16000 \
    --exp_name=walker-8x_target_data


python ../../../sweep_bc.py \
    --main_path=${JAXL_PATH}/jaxl/main.py \
    --config_template=${JAXL_PATH}/scripts/mtil/experiments/configs/main/bc.json \
    --out_dir=${HOME}/scratch/data/bc-8x_target_data \
    --run_seed=0 \
    --num_runs=5 \
    --hyperparam_set=single_sweep \
    --data_dir=${HOME}/scratch/data/expert_data/walker_disc \
    --num_heldouts=10 \
    --num_samples=4000 \
    --discrete_control \
    --exp_name=walker-8x_target_data

chmod +x run_all-*.sh
sbatch run_all-bc-single_sweep-cartpole-8x_target_data_continuous.sh
sbatch run_all-bc-single_sweep-frozenlake-8x_target_data_discrete.sh
sbatch run_all-bc-single_sweep-cheetah-8x_target_data_continuous.sh
sbatch run_all-bc-single_sweep-cheetah-8x_target_data_discrete.sh
sbatch run_all-bc-single_sweep-pendulum-8x_target_data_continuous.sh
sbatch run_all-bc-single_sweep-pendulum-8x_target_data_discrete.sh
sbatch run_all-bc-single_sweep-walker-8x_target_data_continuous.sh
sbatch run_all-bc-single_sweep-walker-8x_target_data_discrete.sh
