source /Users/chanb/research/personal/jaxl/.venv/bin/activate
mkdir -p logs/demonstrations

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=/Users/chanb/research/personal/jaxl/scripts/mtil/local/expert_policies/cheetah_continuous/ \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-cheetah_continuous-num_samples_100000-subsampling_1000.gzip \
#     --num_samples=100000 \
#     --env_seed=1000 \
#     --subsampling_length=1000 \
#     --max_episode_length=1000

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=/Users/chanb/research/personal/jaxl/scripts/mtil/local/expert_policies/cheetah_discrete/ \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-cheetah_discrete-num_samples_100000-subsampling_1000.gzip \
#     --num_samples=100000 \
#     --env_seed=1000 \
#     --subsampling_length=1000 \
#     --max_episode_length=1000

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=/Users/chanb/research/personal/jaxl/scripts/mtil/local/expert_policies/pendulum_continuous/ \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-pendulum_continuous-num_samples_100000-subsampling_200.gzip \
#     --num_samples=100000 \
#     --env_seed=1000 \
#     --subsampling_length=200 \
#     --max_episode_length=200

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=/Users/chanb/research/personal/jaxl/scripts/mtil/local/expert_policies/pendulum_discrete/ \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-pendulum_discrete-num_samples_100000-subsampling_200.gzip \
#     --num_samples=100000 \
#     --env_seed=1000 \
#     --subsampling_length=200 \
#     --max_episode_length=200

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=/Users/chanb/research/personal/jaxl/scripts/mtil/local/expert_policies/walker_continuous/ \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-walker_continuous-num_samples_100000-subsampling_1000.gzip \
#     --num_samples=100000 \
#     --env_seed=1000 \
#     --subsampling_length=1000 \
#     --max_episode_length=1000

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=/Users/chanb/research/personal/mtil_results/data_from_pretrain/pretrained_ppo/walker_discrete/ \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-walker_discrete-num_samples_100000-subsampling_1000.gzip \
#     --num_samples=100000 \
#     --env_seed=1000 \
#     --subsampling_length=1000 \
#     --max_episode_length=1000


python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
    --run_path=/Users/chanb/research/personal/jaxl/scripts/mtil/local/expert_policies/cartpole_continuous/ \
    --save_buffer=./logs/demonstrations/expert_buffer-default-cartpole_continuous-num_samples_5000000-subsampling_1000.gzip \
    --num_samples=5000000 \
    --env_seed=1000 \
    --subsampling_length=1000 \
    --max_episode_length=1000

# python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
#     --run_path=/Users/chanb/research/personal/jaxl/scripts/mtil/local/expert_policies/frozenlake_discrete/ \
#     --save_buffer=./logs/demonstrations/expert_buffer-default-frozenlake_discrete-num_samples_100000-subsampling_200.gzip \
#     --num_samples=100000 \
#     --env_seed=1000 \
#     --subsampling_length=200 \
#     --max_episode_length=200
