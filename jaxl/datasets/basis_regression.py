import chex
import jax.random as jrandom
import numpy as np

from torch.utils.data import Dataset
from typing import Callable, Tuple


class MultitaskFixedBasisRegression1D(Dataset):
    """
    The dataset contains multiple 1D (fixed) basis regression problems.
    """

    def __init__(
        self,
        num_sequences: int,
        sequence_length: int,
        basis: Callable[..., chex.Array],
        seed: int = 0,
        noise: float = 1.0,
        params_bound: Tuple[float, float] = (-0.5, 0.5),
        inputs_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        self._basis = basis
        self._noise = noise
        self._sequence_length = sequence_length
        self._inputs_range = inputs_range
        self._inputs, self._targets, self._params = self._generate_data(
            num_sequences=num_sequences, seed=seed, params_bound=params_bound
        )

    def _generate_data(
        self, num_sequences: int, seed: int, params_bound: Tuple[float, float]
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        model_seed, data_gen_seed = jrandom.split(jrandom.PRNGKey(seed), 2)
        model_rng = np.random.RandomState(seed=model_seed)
        data_gen_rng = np.random.RandomState(seed=data_gen_seed)
        inputs = data_gen_rng.uniform(
            self._inputs_range[0], self._inputs_range[1], (num_sequences * self._sequence_length, 1)
        )

        transformed_inputs = self._basis(inputs)
        transformed_inputs = transformed_inputs.reshape(
            (num_sequences, self._sequence_length, -1)
        )
        params = model_rng.uniform(
            low=params_bound[0],
            high=params_bound[1],
            size=(num_sequences, transformed_inputs.shape[2], 1),
        )
        targets = (
            transformed_inputs @ params
            + data_gen_rng.randn(num_sequences, self._sequence_length, 1) * self._noise
        )

        return (
            inputs.reshape((num_sequences, self._sequence_length, 1)),
            targets,
            params,
        )

    @property
    def input_dim(self) -> chex.Array:
        return (1,)

    @property
    def output_dim(self) -> chex.Array:
        return (1,)

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
    def basis(self) -> Callable[..., chex.Array]:
        return self._basis

    @property
    def params(self) -> chex.Array:
        return self._params

    @property
    def ls_estimators(self) -> chex.Array:
        basis_mat = self._basis(self._inputs).reshape(
            (len(self), self._sequence_length, -1)
        )
        basis_mat_T = basis_mat.transpose((0, 2, 1))
        ls_estimator = (
            np.linalg.pinv(basis_mat_T @ basis_mat) @ basis_mat_T @ self._targets
        )
        return ls_estimator
