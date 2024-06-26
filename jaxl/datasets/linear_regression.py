import chex
import jax.random as jrandom
import numpy as np

from torch.utils.data import Dataset
from typing import Callable, Tuple


class MultitaskLinearRegressionND(Dataset):
    """
    The dataset contains multiple ND (fixed) linear regression problems.
    """

    def __init__(
        self,
        num_sequences: int,
        sequence_length: int,
        input_dim: int,
        seed: int = 0,
        noise: float = 1.0,
        params_bound: Tuple[float, float] = (-0.5, 0.5),
        inputs_range: Tuple[float, float] = (-1.0, 1.0),
        num_active_params: int = None,
        bias: bool = False,
    ):
        assert num_active_params is None or num_active_params >= 0
        self._noise = noise
        self._sequence_length = sequence_length
        self._input_dim = input_dim
        self._inputs_range = inputs_range
        self._inputs, self._targets, self._params = self._generate_data(
            num_sequences=num_sequences,
            seed=seed,
            params_bound=params_bound,
            num_active_params=num_active_params,
            bias=bias,
        )

    def _generate_data(
        self,
        num_sequences: int,
        seed: int,
        params_bound: Tuple[float, float],
        num_active_params: int,
        bias: bool,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        model_seed, data_gen_seed = jrandom.split(jrandom.PRNGKey(seed), 2)
        model_rng = np.random.RandomState(seed=model_seed)
        data_gen_rng = np.random.RandomState(seed=data_gen_seed)
        inputs = data_gen_rng.uniform(
            self._inputs_range[0],
            self._inputs_range[1],
            (num_sequences, self._sequence_length, self._input_dim),
        )

        params = model_rng.uniform(
            low=params_bound[0],
            high=params_bound[1],
            size=(num_sequences, self._input_dim + 1, 1),
        )

        params[:, 0] = params[:, 0] * int(bias)

        if num_active_params is not None:
            params[:, -self._input_dim - num_active_params :] = 0

        targets = (
            inputs @ params[:, 1:]
            + params[:, :1]
            + data_gen_rng.randn(num_sequences, self._sequence_length, 1) * self._noise
        )

        return (
            inputs,
            targets,
            params,
        )

    @property
    def input_dim(self) -> chex.Array:
        return (self._input_dim,)

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
    def params(self) -> chex.Array:
        return self._params

    @property
    def ls_estimators(self) -> chex.Array:
        inputs_T = self._inputs.transpose((0, 2, 1))
        ls_estimator = (
            np.linalg.pinv(inputs_T @ self._inputs) @ inputs_T @ self._targets
        )
        return ls_estimator
