python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
    --run_path=/Users/chanb/research/personal/mtil_results/data_from_pretrain/pretrained_ppo/cheetah_discrete/ \
    --num_samples=100000 \
    --save_buffer=./logs/demonstrations/expert_buffer-default-cheetah_discrete-num_samples_100000-subsampling_1.gzip \
    --env_seed=1000 \
    --subsampling_length=1 \
    --max_episode_length=1000

python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
    --run_path=/Users/chanb/research/personal/mtil_results/data_from_pretrain/pretrained_ppo/cheetah_discrete/ \
    --num_samples=100000 \
    --save_buffer=./logs/demonstrations/expert_buffer-default-cheetah_discrete-num_samples_100000-subsampling_20.gzip \
    --env_seed=1000 \
    --subsampling_length=20 \
    --max_episode_length=1000

python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
    --run_path=/Users/chanb/research/personal/mtil_results/data_from_pretrain/pretrained_ppo/cheetah_discrete/ \
    --num_samples=100000 \
    --save_buffer=./logs/demonstrations/expert_buffer-default-cheetah_discrete-num_samples_100000-subsampling_1000.gzip \
    --env_seed=1000 \
    --subsampling_length=1000 \
    --max_episode_length=1000


python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
    --run_path=/Users/chanb/research/personal/mtil_results/data_from_pretrain/pretrained_ppo/pendulum_continuous/ \
    --num_samples=100000 \
    --save_buffer=./logs/demonstrations/expert_buffer-default-pendulum_continuous-num_samples_100000-subsampling_1.gzip \
    --env_seed=1000 \
    --subsampling_length=1 \
    --max_episode_length=200

python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
    --run_path=/Users/chanb/research/personal/mtil_results/data_from_pretrain/pretrained_ppo/pendulum_continuous/ \
    --num_samples=100000 \
    --save_buffer=./logs/demonstrations/expert_buffer-default-pendulum_continuous-num_samples_100000-subsampling_20.gzip \
    --env_seed=1000 \
    --subsampling_length=20 \
    --max_episode_length=200

python /Users/chanb/research/personal/jaxl/jaxl/gather_expert_data.py \
    --run_path=/Users/chanb/research/personal/mtil_results/data_from_pretrain/pretrained_ppo/pendulum_continuous/ \
    --num_samples=100000 \
    --save_buffer=./logs/demonstrations/expert_buffer-default-pendulum_continuous-num_samples_100000-subsampling_200.gzip \
    --env_seed=1000 \
    --subsampling_length=200 \
    --max_episode_length=200
