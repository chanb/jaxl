import _pickle as pickle
import chex
import jax.random as jrandom
import numpy as np
import os

from torch.utils.data import Dataset
from typing import Callable, Tuple


class MultitaskRandomClassificationND(Dataset):
    """
    The dataset contains multiple ND (fixed) linear classification problems.
    """

    def __init__(
        self,
        num_sequences: int,
        sequence_length: int,
        input_dim: int,
        seed: int = 0,
        inputs_range: Tuple[float, float] = (-1.0, 1.0),
        save_path: str = None,
    ):
        self._sequence_length = sequence_length
        self._input_dim = input_dim
        self._inputs_range = inputs_range
        self._params = (1, 0.5)

        if save_path is None or not os.path.isfile(save_path):
            self._inputs, self._targets = self._generate_data(
                num_sequences=num_sequences,
                seed=seed,
            )
            if save_path is not None:
                print("Saving to {}".format(save_path))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                pickle.dump(
                    (self._inputs, self._targets, self._params),
                    open(save_path, "wb"),
                )
        else:
            print("Loading from {}".format(save_path))
            self._inputs, self._targets, self._params = pickle.load(
                open(save_path, "rb")
            )

    def _generate_data(
        self,
        num_sequences: int,
        seed: int,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        model_seed, data_gen_seed = jrandom.split(jrandom.PRNGKey(seed), 2)
        data_gen_rng = np.random.RandomState(seed=data_gen_seed)

        done_generation = False
        while not done_generation:
            inputs = data_gen_rng.uniform(
                self._inputs_range[0],
                self._inputs_range[1],
                (num_sequences, self._sequence_length, self._input_dim),
            )
            targets = data_gen_rng.binomial(1, 0.5, size=(num_sequences, self._sequence_length))

            if np.all(
                np.logical_and(
                    0 < np.sum(targets, axis=-1),
                    np.sum(targets, axis=-1) < self._sequence_length,
                )
            ):
                done_generation = True

        targets = np.eye(2)[targets][:, :]

        return (
            inputs,
            targets,
        )

    @property
    def input_dim(self) -> chex.Array:
        return (self._input_dim,)

    @property
    def output_dim(self) -> chex.Array:
        return (2,)

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    def __len__(self):
        return len(self._inputs)

    def __getitem__(self, idx):
        return (self._inputs[idx], self._targets[idx])

    def plot(self):
        raise NotImplementedError

    @property
    def params(self) -> chex.Array:
        return self._params

    @property
    def ls_estimators(self) -> chex.Array:
        inputs_T = self._inputs.transpose((0, 2, 1))
        ls_estimator = (
            np.linalg.pinv(inputs_T @ self._inputs) @ inputs_T @ self._targets
        )
        return ls_estimator
