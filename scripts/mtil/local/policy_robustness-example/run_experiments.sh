#!/bin/bash
source /Users/chanb/research/personal/jaxl/.venv/bin/activate

python /Users/chanb/research/personal/jaxl/jaxl/main.py --config_path=/Users/chanb/research/personal/jaxl/scripts/mtil/experiments/configs/policy_robustness-pendulum/default_env.json

# Should update the config with the pretrained model
python /Users/chanb/research/personal/jaxl/jaxl/main.py --config_path=/Users/chanb/research/personal/jaxl/scripts/mtil/experiments/configs/policy_robustness-pendulum/extreme_env.json
