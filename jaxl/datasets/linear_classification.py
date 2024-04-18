import chex
import jax.random as jrandom
import numpy as np

from torch.utils.data import Dataset
from typing import Tuple

from jaxl.datasets.utils import maybe_load_dataset, maybe_save_dataset


class MultitaskLinearClassificationND(Dataset):
    """
    The dataset contains multiple ND (fixed) linear classification problems.
    """

    def __init__(
        self,
        num_sequences: int,
        sequence_length: int,
        input_dim: int,
        seed: int = 0,
        noise: float = 0.0,
        params_bound: Tuple[float, float] = (-0.5, 0.5),
        inputs_range: Tuple[float, float] = (-1.0, 1.0),
        num_active_params: int = None,
        bias: bool = False,
        margin: float = 0.0,
        save_dir: str = None,
    ):
        assert num_active_params is None or num_active_params >= 0
        self._noise = noise
        self._sequence_length = sequence_length
        self._input_dim = input_dim
        self._inputs_range = inputs_range

        dataset_name = "multitask_linear_classification-num_sequences_{}-sequence_length_{}-input_dim_{}-noise_{}-num_active_params_{}-margin_{}-bias_{}-seed_{}.pkl".format(
            num_sequences,
            sequence_length,
            input_dim,
            noise,
            num_active_params,
            margin,
            bias,
            seed,
        )
        loaded, data = maybe_load_dataset(save_dir, dataset_name)

        if not loaded:
            inputs, targets, params = self._generate_data(
                num_sequences=num_sequences,
                sequence_length=sequence_length,
                input_dim=input_dim,
                seed=seed,
                noise=noise,
                params_bound=params_bound,
                inputs_range=inputs_range,
                num_active_params=num_active_params,
                bias=bias,
                margin=margin,
            )
            data = {
                "inputs": inputs,
                "targets": targets,
                "params": params,
                "num_sequences": num_sequences,
                "sequence_length": sequence_length,
                "input_dim": input_dim,
                "inputs_range": inputs_range,
                "params_bound": params_bound,
                "noise": noise,
                "num_active_params": num_active_params,
                "margin": margin,
                "bias": bias,
                "seed": seed,
            }
            maybe_save_dataset(
                data,
                save_dir,
                dataset_name,
            )
        self._data = data

    def _generate_data(
        self,
        num_sequences: int,
        sequence_length: int,
        input_dim: int,
        seed: int = 0,
        noise: float = 0.0,
        params_bound: Tuple[float, float] = (-0.5, 0.5),
        inputs_range: Tuple[float, float] = (-1.0, 1.0),
        num_active_params: int = None,
        bias: bool = False,
        margin: float = 0.0,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        model_seed, data_gen_seed = jrandom.split(jrandom.PRNGKey(seed), 2)
        model_rng = np.random.RandomState(seed=model_seed)
        data_gen_rng = np.random.RandomState(seed=data_gen_seed)

        params = model_rng.uniform(
            low=params_bound[0],
            high=params_bound[1],
            size=(num_sequences, input_dim + 1, 1),
        )

        params[:, 0] = params[:, 0] * int(bias)

        done_generation = False
        while not done_generation:
            num_valid_pts = 0
            inputs = np.zeros(
                (num_sequences, sequence_length, input_dim),
            )
            replace_mask = inputs == 0
            while num_valid_pts != num_sequences * sequence_length:
                samples = data_gen_rng.uniform(
                    inputs_range[0],
                    inputs_range[1],
                    (num_sequences, sequence_length, input_dim),
                )
                inputs = inputs * (1 - replace_mask) + samples * replace_mask
                dists = np.abs(inputs @ params[:, 1:] + params[:, :1])[
                    ..., 0
                ] / np.sqrt(np.sum(params[:, 1:] ** 2, axis=1))
                replace_mask = (dists < margin)[..., None]
                replace_mask = np.concatenate((replace_mask, replace_mask), axis=-1)
                num_valid_pts = np.sum(dists >= margin)

            if num_active_params is not None:
                params[:, -input_dim - num_active_params :] = 0

            targets = (
                (
                    inputs @ params[:, 1:]
                    + params[:, :1]
                    + data_gen_rng.randn(num_sequences, sequence_length, 1) * noise
                )
                >= 0
            ).astype(int)[..., 0]
            if np.all(
                np.logical_and(
                    0 < np.sum(targets, axis=-1),
                    np.sum(targets, axis=-1) < sequence_length,
                )
            ):
                done_generation = True

        targets = np.eye(2)[targets][:, :]

        return (
            inputs,
            targets,
            params,
        )

    @property
    def input_dim(self) -> chex.Array:
        return (self._data["input_dim"],)

    @property
    def output_dim(self) -> chex.Array:
        return (2,)

    @property
    def sequence_length(self) -> int:
        return self._data["sequence_length"]

    def __len__(self):
        return self._data["num_sequences"]

    def __getitem__(self, idx):
        return (self._data["inputs"][idx], self._data["targets"][idx])

    def plot(self):
        raise NotImplementedError

    @property
    def params(self) -> chex.Array:
        return self._data["params"]

    @property
    def ls_estimators(self) -> chex.Array:
        inputs_T = self._data["inputs"].transpose((0, 2, 1))
        ls_estimator = (
            np.linalg.pinv(inputs_T @ self._data["inputs"])
            @ inputs_T
            @ self._data["targets"]
        )
        return ls_estimator
