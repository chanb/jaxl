#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate

python generate_expert_data.py \
    --main_path=${JAXL_PATH}/jaxl/gather_expert_data.py \
    --out_dir=${HOME}/scratch/data/expert_data \
    --run_seed=0 \
    --env_seed=9999 \
    --num_samples=10000 \
    --subsampling_lengths=20 \
    --subsampling_lengths=1 \
    --run_time=05:00:00 \
    --max_episode_length=200 \
    --runs_dir=${HOME}/scratch/data/experts/pendulum/continuous/runs \
    --exp_name=pendulum_cont

python generate_expert_data.py \
    --main_path=${JAXL_PATH}/jaxl/gather_expert_data.py \
    --out_dir=${HOME}/scratch/data/expert_data \
    --run_seed=0 \
    --env_seed=9999 \
    --num_samples=10000 \
    --subsampling_lengths=20 \
    --subsampling_lengths=1 \
    --run_time=05:00:00 \
    --max_episode_length=200 \
    --runs_dir=${HOME}/scratch/data/experts/pendulum/discrete/runs \
    --exp_name=pendulum_disc

python generate_expert_data.py \
    --main_path=${JAXL_PATH}/jaxl/gather_expert_data.py \
    --out_dir=${HOME}/scratch/data/expert_data \
    --run_seed=0 \
    --env_seed=9999 \
    --num_samples=10000 \
    --subsampling_lengths=20 \
    --subsampling_lengths=1 \
    --run_time=05:00:00 \
    --max_episode_length=1000 \
    --runs_dir=${HOME}/scratch/data/experts/cheetah/continuous/runs \
    --exp_name=cheetah_cont

python generate_expert_data.py \
    --main_path=${JAXL_PATH}/jaxl/gather_expert_data.py \
    --out_dir=${HOME}/scratch/data/expert_data \
    --run_seed=0 \
    --env_seed=9999 \
    --num_samples=10000 \
    --subsampling_lengths=20 \
    --subsampling_lengths=1 \
    --run_time=05:00:00 \
    --max_episode_length=1000 \
    --runs_dir=${HOME}/scratch/data/experts/cheetah/discrete/runs \
    --exp_name=cheetah_disc

chmod +x run_all-*.sh
sbatch run_all-generate_expert_data-cheetah_cont.sh
sbatch run_all-generate_expert_data-cheetah_disc.sh
sbatch run_all-generate_expert_data-pendulum_cont.sh
sbatch run_all-generate_expert_data-pendulum_disc.sh

