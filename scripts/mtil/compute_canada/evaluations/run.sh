#!/bin/bash

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate

python generate-evaluate_all.py