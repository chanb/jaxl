#!/bin/bash
conda activate jaxl
mkdir -p expert_data
python gather_expert_data-open_drawer.py --env_seed=0 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_0.gzip
python gather_expert_data-open_drawer.py --env_seed=1 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_1.gzip
python gather_expert_data-open_drawer.py --env_seed=2 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_2.gzip
python gather_expert_data-open_drawer.py --env_seed=3 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_3.gzip
python gather_expert_data-open_drawer.py --env_seed=4 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_4.gzip
python gather_expert_data-open_drawer.py --env_seed=5 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_5.gzip
python gather_expert_data-open_drawer.py --env_seed=6 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_6.gzip
python gather_expert_data-open_drawer.py --env_seed=7 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_7.gzip
python gather_expert_data-open_drawer.py --env_seed=8 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_8.gzip
python gather_expert_data-open_drawer.py --env_seed=9 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_9.gzip
python gather_expert_data-open_drawer.py --env_seed=10 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_10.gzip
python gather_expert_data-open_drawer.py --env_seed=11 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_11.gzip
python gather_expert_data-open_drawer.py --env_seed=12 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_12.gzip
python gather_expert_data-open_drawer.py --env_seed=13 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_13.gzip
python gather_expert_data-open_drawer.py --env_seed=14 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_14.gzip
python gather_expert_data-open_drawer.py --env_seed=15 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_15.gzip
python gather_expert_data-open_drawer.py --env_seed=16 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_16.gzip
python gather_expert_data-open_drawer.py --env_seed=17 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_17.gzip
python gather_expert_data-open_drawer.py --env_seed=18 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_18.gzip
python gather_expert_data-open_drawer.py --env_seed=19 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_19.gzip
python gather_expert_data-open_drawer.py --env_seed=20 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_20.gzip
python gather_expert_data-open_drawer.py --env_seed=21 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_21.gzip
python gather_expert_data-open_drawer.py --env_seed=22 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_22.gzip
python gather_expert_data-open_drawer.py --env_seed=23 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_23.gzip
python gather_expert_data-open_drawer.py --env_seed=24 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_24.gzip
python gather_expert_data-open_drawer.py --env_seed=25 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_25.gzip
python gather_expert_data-open_drawer.py --env_seed=26 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_26.gzip
python gather_expert_data-open_drawer.py --env_seed=27 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_27.gzip
python gather_expert_data-open_drawer.py --env_seed=28 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_28.gzip
python gather_expert_data-open_drawer.py --env_seed=29 --num_samples=100000 --subsampling_length=500 --scrambling_step=10 --save_buffer=./expert_data/test_metaworld-seed_29.gzip
