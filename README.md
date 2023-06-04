# JAXL (JAX Learning)
## Prerequisite:
- Python 3.9
- MuJoCo (See [here](https://github.com/openai/mujoco-py#install-mujoco))
  - Remember to set `LD_LIBRARY_PATH` (e..g. `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<PATH_TO_MUJOCO>/.mujoco/mujoco210/bin`)
  - For troubleshooting, see [here](https://github.com/openai/mujoco-py#ubuntu-installtion-troubleshooting)

## Installation:
You may simply `pip install`:
```
pip install -r requirements/all.txt
pip install -e .
```

When install Jax on GPU, note that we need to do the following instead (see [here](https://github.com/google/jax#installation)):
```
# CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

### Compute Canada
You may install this code on Compute Canada by simply running `jaxl/installation/compute_canada/initial_setup.sh`.

## Experiments
### Locally
To run a single experiment, you may execute:
```
python jaxl/main.py --config_path=${config_path} --run_seed=${run_seed}
```
Examples of configuration file (i.e. `config_path`) are located under `jaxl/configs`.

To run multiple experiments, you may refer to `scripts/mtil/local/inverted_pendulum`.
In particular, we first run `generate_experts.py` to construct a bash script, say `run_all-inverted_pendulum.sh`.
Then, running said bash script will execute mutliple experiments.

### Compute Canada
To run on Compute Canada, you may refer to `scripts/mtil/compute_canada`.
In partciular, first run `generate_expert_variants.py` to construct `dat` file, consisting of different hyperparameters.
Then, run `sbatch generate_experts.sh` to run each variant in paralllel.

## Design Pattern
This codebase aims to combine both imitation learning and reinforcement learning into a single design pattern.
To this end, we note that imitation learning consists of supervised learning, which is purely offline, whereas other approaches are more online.
In particular, offline means that we are given a fixed batch of data, whereas online means that we receive a stream of data as the learner interacts with the environment.
We emphasize that learner-environment interactions do not necessarily mean that the learner has influence over the distribution (e.g. online supervised learning).

Consequently, we have few main components:
- `main.py` is the main entrypoint to running experiments, where it defines the save paths, models, buffers, learners, and other necessary components for running an experiment.
- `learning_utils.py` orchestrates the learning progress.
That is, it will instruct the learner to perform updates, keep track of learning progress, and checkpointing the models.
It also provides utilities for constructing models, learners, and optimizers.
- `buffers` consists of various types of buffers.
We currently support the standard replay buffer (`TransitionNumPyBuffer`) and trajectory-based replay buffer (`TrajectoryNumPyBuffer`).
If we wish to add functionality such as prioritized experience replay and hindsight experience replay, we should consider using wrappers/decorators to extend the functionalities.
- `learners` consists of various types of learners.
In particular, there are two types of learners: `OfflineLearner` and `OnlineLearner`, that are realizations of the abstract `Learner` class.
By default, the `Learner` class enforces all learners have a way to perform checkpointing, initializing the parameters, and performing learning updates (if any).
  - `OfflineLearner` assumes that a batch of **offline** data is given, as a `ReplayBuffer`.
  The corresponding attribute is `_dataset`, following the standard offline learning vocabulary.
  - On the other hand, `OnlineLearner` assumes that it has access to a **buffer**, which is populated sample by sample.
  By varying the buffer size, we can obtain "streaming" online learners that assume to have no *memory*.
  The online learner can also interact with the environment, which is implemented via the `gymnasium` API.


## Styling
- We use [`black`](https://github.com/psf/black/blob/main/README.md) to format code and [`pyment`](https://github.com/dadadel/pyment/blob/master/README.rst) to generate docstrings with `reST` style.

### Generating Docstrings
To generate the docstrings in HTML format, we use [`sphinx`](https://github.com/sphinx-doc/sphinx).
```
cd docs
sphinx-apidoc -f -o . ..
make html​
```

## TODO
- Update README as more implementation comes
- Implement visualization code to compare policies
  - Check cosine similarity between policies' mean action
  - Check state-conditioned KL divergence between two policies

## Long-term Modification
- JAX implementation of the replay buffers
- Revisit A2C, PPO, and MTBC for better loss construction.
It seems to be using a different design pattern.
- Revisit how we construct models and optimizers.
There should be a more elegant way.
