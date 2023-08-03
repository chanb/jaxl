#!/bin/bash

base_path=/Users/chanb/research/personal/jaxl/scripts/mtil/local
source /Users/chanb/research/personal/jaxl/.venv/bin/activate

mkdir -p ./logs/demonstrations

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/frozenlake_discrete/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-frozenlake_discrete-num_samples_10000-subsampling_1.gzip \
#     --env_seed=1000 \
#     --subsampling_length=1 \
#     --max_episode_length=200

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/frozenlake_discrete/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-frozenlake_discrete-num_samples_10000-subsampling_20.gzip \
#     --env_seed=1000 \
#     --subsampling_length=20 \
#     --max_episode_length=200

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/frozenlake_discrete/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-frozenlake_discrete-num_samples_10000-subsampling_200.gzip \
#     --env_seed=1000 \
#     --subsampling_length=200 \
#     --max_episode_length=200

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/cartpole_continuous/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-cartpole_continuous-num_samples_10000-subsampling_1.gzip \
#     --env_seed=1000 \
#     --subsampling_length=1 \
#     --max_episode_length=1000

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/cartpole_continuous/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-cartpole_continuous-num_samples_10000-subsampling_20.gzip \
#     --env_seed=1000 \
#     --subsampling_length=20 \
#     --max_episode_length=1000

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/cartpole_continuous/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-cartpole_continuous-num_samples_10000-subsampling_1000.gzip \
#     --env_seed=1000 \
#     --subsampling_length=1000 \
#     --max_episode_length=1000

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/cheetah_discrete/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-cheetah_discrete-num_samples_10000-subsampling_1.gzip \
#     --env_seed=1000 \
#     --subsampling_length=1 \
#     --max_episode_length=1000

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/cheetah_discrete/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-cheetah_discrete-num_samples_10000-subsampling_20.gzip \
#     --env_seed=1000 \
#     --subsampling_length=20 \
#     --max_episode_length=1000

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/cheetah_discrete/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-cheetah_discrete-num_samples_10000-subsampling_1000.gzip \
#     --env_seed=1000 \
#     --subsampling_length=1000 \
#     --max_episode_length=1000

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/cheetah_continuous/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-cheetah_continuous-num_samples_10000-subsampling_1.gzip \
#     --env_seed=1000 \
#     --subsampling_length=1 \
#     --max_episode_length=1000

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/cheetah_continuous/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-cheetah_continuous-num_samples_10000-subsampling_20.gzip \
#     --env_seed=1000 \
#     --subsampling_length=20 \
#     --max_episode_length=1000

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/cheetah_continuous/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-cheetah_continuous-num_samples_10000-subsampling_1000.gzip \
#     --env_seed=1000 \
#     --subsampling_length=1000 \
#     --max_episode_length=1000

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/pendulum_discrete/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-pendulum_discrete-num_samples_10000-subsampling_1.gzip \
#     --env_seed=1000 \
#     --subsampling_length=1 \
#     --max_episode_length=200

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/pendulum_discrete/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-pendulum_discrete-num_samples_10000-subsampling_20.gzip \
#     --env_seed=1000 \
#     --subsampling_length=20 \
#     --max_episode_length=200

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/pendulum_discrete/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-pendulum_discrete-num_samples_10000-subsampling_200.gzip \
#     --env_seed=1000 \
#     --subsampling_length=200 \
#     --max_episode_length=200

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/pendulum_continuous/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-pendulum_continuous-num_samples_10000-subsampling_1.gzip \
#     --env_seed=1000 \
#     --subsampling_length=1 \
#     --max_episode_length=200

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/pendulum_continuous/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-pendulum_continuous-num_samples_10000-subsampling_20.gzip \
#     --env_seed=1000 \
#     --subsampling_length=20 \
#     --max_episode_length=200

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=${base_path}/expert_policies/pendulum_continuous/ \
#     --num_samples=10000 \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-pendulum_continuous-num_samples_10000-subsampling_200.gzip \
#     --env_seed=1000 \
#     --subsampling_length=200 \
#     --max_episode_length=200

python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
    --run_path=${base_path}/expert_policies/walker_discrete/ \
    --num_samples=10000 \
    --save_buffer=./logs/demonstrations/expert_buffer-default-walker_discrete-num_samples_10000-subsampling_1.gzip \
    --env_seed=1000 \
    --subsampling_length=1 \
    --max_episode_length=1000

python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
    --run_path=${base_path}/expert_policies/walker_discrete/ \
    --num_samples=10000 \
    --save_buffer=./logs/demonstrations/expert_buffer-default-walker_discrete-num_samples_10000-subsampling_20.gzip \
    --env_seed=1000 \
    --subsampling_length=20 \
    --max_episode_length=1000

python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
    --run_path=${base_path}/expert_policies/walker_discrete/ \
    --num_samples=10000 \
    --save_buffer=./logs/demonstrations/expert_buffer-default-walker_discrete-num_samples_10000-subsampling_1000.gzip \
    --env_seed=1000 \
    --subsampling_length=1000 \
    --max_episode_length=1000

python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
    --run_path=${base_path}/expert_policies/walker_continuous/ \
    --num_samples=10000 \
    --save_buffer=./logs/demonstrations/expert_buffer-default-walker_continuous-num_samples_10000-subsampling_1.gzip \
    --env_seed=1000 \
    --subsampling_length=1 \
    --max_episode_length=1000

python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
    --run_path=${base_path}/expert_policies/walker_continuous/ \
    --num_samples=10000 \
    --save_buffer=./logs/demonstrations/expert_buffer-default-walker_continuous-num_samples_10000-subsampling_20.gzip \
    --env_seed=1000 \
    --subsampling_length=20 \
    --max_episode_length=1000

python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
    --run_path=${base_path}/expert_policies/walker_continuous/ \
    --num_samples=10000 \
    --save_buffer=./logs/demonstrations/expert_buffer-default-walker_continuous-num_samples_10000-subsampling_1000.gzip \
    --env_seed=1000 \
    --subsampling_length=1000 \
    --max_episode_length=1000