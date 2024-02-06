from torch.utils.data import Dataset
from types import SimpleNamespace
from typing import Tuple

from jaxl.constants import (
    VALID_OMNIGLOT_TASKS,
    CONST_MULTITASK_OMNIGLOT_FINEGRAIN,
)
from jaxl.datasets.utils import (
    maybe_save_dataset,
    maybe_load_dataset,
)

import _pickle as pickle
import chex
import jax.random as jrandom
import numpy as np
import os
import torchvision.datasets as torch_datasets

import jaxl.transforms as jaxl_transforms


def construct_omniglot(
    save_path: str,
    task_name: str = None,
    task_config: SimpleNamespace = None,
    train: bool = True,
) -> Dataset:
    """
    Constructs a customized Omniglot dataset.

    :param save_path: the path to store the Omniglot dataset
    :param task_name: the task to construct
    :param task_config: the task configuration
    :param train: the train split of the dataset
    :type save_path: str
    :type task_name: str:  (Default value = None)
    :type task_config: SimpleNamespace:  (Default value = None)
    :type train: bool:  (Default value = True)
    :return: Customized Omniglot dataset
    :rtype: Dataset

    """
    assert (
        task_name is None or task_name in VALID_OMNIGLOT_TASKS
    ), f"{task_name} is not supported (one of {VALID_OMNIGLOT_TASKS})"
    target_transform = None

    if task_name is None:
        # By default, the Omniglot task will be normalized to be between 0 to 1.
        return torch_datasets.Omniglot(
            save_path,
            background=train,
            download=True,
            transform=jaxl_transforms.DefaultPILToImageTransform(),
            target_transform=target_transform,
        )
    elif task_name == CONST_MULTITASK_OMNIGLOT_FINEGRAIN:
        return MultitaskOmniglotFineGrain(
            dataset=torch_datasets.Omniglot(
                save_path,
                train=train,
                download=True,
                transform=jaxl_transforms.StandardImageTransform(),
                target_transform=target_transform,
            ),
            num_sequences=task_config.num_sequences,
            sequence_length=task_config.sequence_length,
            random_label=getattr(task_config, "random_label", False),
            save_dir=task_config.save_dir,
        )
    else:
        raise ValueError(f"{task_name} is invalid (one of {VALID_OMNIGLOT_TASKS})")


class MultitaskOmniglotFineGrain(Dataset):
    """
    The dataset contains multiple ND (fixed) linear classification problems.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_sequences: int,
        sequence_length: int,
        seed: int = 0,
        random_label: bool = False,
        save_dir: str = None,
    ):
        dataset_name = "omniglot_finegrain-background_{}-num_sequences_{}-sequence_length_{}-random_label_{}-seed_{}.pkl".format(
            dataset.background,
            num_sequences,
            sequence_length,
            random_label,
            seed,
        )
        loaded, data = maybe_load_dataset(save_dir, dataset_name)

        if not loaded:
            sample_idxes, label_map = self._generate_data(
                dataset=dataset,
                num_sequences=num_sequences,
                random_label=random_label,
                seed=seed,
            )
            data = {
                "sample_idxes": sample_idxes,
                "label_map": label_map,
                "num_sequences": num_sequences,
                "sequence_length": sequence_length,
                "random_label": random_label,
                "background": dataset.background,
                "input_shape": [*self._dataset.data[0].shape],
                "num_classes": int(len(dataset) / 20),
                "seed": seed,
            }
            maybe_save_dataset(
                data,
                save_dir,
                dataset_name,
            )

        self._dataset = dataset
        self._data = data

    def _generate_data(
        self,
        dataset: Dataset,
        num_sequences: int,
        random_label: bool,
        seed: int,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        print("Generating Data")
        sample_key, label_key = jrandom.split(jrandom.PRNGKey(seed))
        sample_rng = np.random.RandomState(sample_key)
        label_rng = np.random.RandomState(label_key)

        sample_idxes = sample_rng.choice(
            np.arange(len(dataset)), size=(num_sequences, self._sequence_length)
        )

        label_map = np.tile(np.arange(self.output_dim[0]), reps=(num_sequences, 1))
        if random_label:
            label_map = np.apply_along_axis(
                label_rng.permutation, axis=1, arr=label_map
            )

        return sample_idxes, label_map

    @property
    def input_dim(self) -> chex.Array:
        return self._data["input_shape"]

    @property
    def output_dim(self) -> chex.Array:
        return (self._data["num_classes"],)

    @property
    def sequence_length(self) -> int:
        return self._data["sequence_length"]

    def __len__(self):
        return self._data["num_sequences"]

    def __getitem__(self, idx):
        sample_idxes = self._data["sample_idxes"][idx].tolist()
        inputs = self._dataset.transform(self._dataset.data[sample_idxes])
        labels = self._data["label_map"][idx][self._dataset.targets[sample_idxes]]
        outputs = np.eye(self._data["num_classes"])[labels]
        return (inputs, outputs)
