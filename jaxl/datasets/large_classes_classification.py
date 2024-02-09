import _pickle as pickle
import chex
import jax.random as jrandom
import numpy as np
import os

from torch.utils.data import Dataset
from typing import Callable, Tuple

from jaxl.constants import VALID_SPLIT, CONST_TRAIN
from jaxl.datasets.utils import maybe_save_dataset, maybe_load_dataset


class OneHotClassification(Dataset):
    """
    The dataset contains multiple ND (fixed) linear classification problems.
    """

    def __init__(
        self,
        num_sequences: int,
        sequence_length: int,
        num_classes: int,
        num_holdout: int,
        split: str,
        remap: bool = False,
        seed: int = 0,
        inputs_range: Tuple[float, float] = (0.0, 0.5),
        save_dir: str = None,
    ):
        assert (
            len(inputs_range) == 2
        ), "inputs_range should be a 2-tuple, got {}".format(len(inputs_range))
        assert (
            0 <= inputs_range[0] < inputs_range[1]
        ), "first element {} should be less than second element {} and non-negative".format(
            inputs_range[0],
            inputs_range[1],
        )
        assert split in VALID_SPLIT, "support one of {}".format(VALID_SPLIT)
        assert (
            0 <= num_holdout < num_classes
        ), "num_holdout {} must be less than num_classes {} and non-negative".format(
            num_holdout, num_classes
        )

        dataset_name = "one_hot_classification-num_sequences_{}-sequence_length_{}-num_classes_{}-num_holdout_{}-split_{}-remap_{}-seed_{}.pkl".format(
            num_sequences,
            sequence_length,
            num_classes,
            num_holdout,
            split,
            remap,
            seed,
        )
        loaded, data = maybe_load_dataset(save_dir, dataset_name)
        if not loaded:
            inputs, targets = self._generate_data(
                num_sequences=num_sequences,
                sequence_length=sequence_length,
                num_classes=num_classes,
                num_holdout=num_holdout,
                split=split,
                remap=remap,
                inputs_range=inputs_range,
                seed=seed,
            )
            data = {
                "inputs": inputs,
                "targets": targets,
                "num_sequences": num_sequences,
                "sequence_length": sequence_length,
                "num_classes": num_classes,
                "num_holdout": num_holdout,
                "inputs_range": inputs_range,
                "split": split,
                "remap": remap,
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
        num_classes: int,
        num_holdout: int,
        split: str,
        remap: bool,
        inputs_range: Tuple[float, float],
        seed: int,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        is_train = int(split == CONST_TRAIN)
        data_gen_seed = jrandom.split(jrandom.PRNGKey(seed), 2)[is_train]
        data_gen_rng = np.random.RandomState(seed=data_gen_seed)

        targets = data_gen_rng.choice(
            num_classes - num_holdout if is_train else num_holdout,
            size=(num_sequences, sequence_length),
        )
        if not is_train and not remap:
                targets += num_classes - num_holdout

        inputs = np.eye(num_classes)[targets]
        noise = inputs_range[1] - data_gen_rng.uniform(
            inputs_range[0],
            inputs_range[1],
            (num_sequences, sequence_length, 1),
        )
        inputs = inputs * noise

        if remap:
            targets = targets % 2

        return inputs, targets

    @property
    def input_dim(self) -> chex.Array:
        return (self._data["num_classes"],)

    @property
    def output_dim(self) -> chex.Array:
        return (self._data["num_classes"],)

    @property
    def sequence_length(self) -> int:
        return self._data["sequence_length"]

    def __len__(self):
        return self._data["num_sequences"]

    def __getitem__(self, idx):
        inputs = self._data["inputs"][idx]
        targets = np.eye(self._data["num_classes"])[self._data["targets"][idx]]
        return (inputs, targets)
