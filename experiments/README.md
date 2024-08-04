# Installation
```
module load python/3.10

python -m venv ~/icl_env
source ~/icl_env/bin/activate

pip install jax --no-index
pip install optax flax --no-index
pip install chex dill matplotlib --no-index
pip install gymnasium --no-index
pip install torch torchvision --no-index
pip install tensorflow_datasets --no-index
```