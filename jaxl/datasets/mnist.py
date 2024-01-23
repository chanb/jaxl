from torch.utils.data import Dataset
from types import SimpleNamespace
from typing import Tuple

from jaxl.constants import (
    VALID_MNIST_TASKS,
    CONST_MULTITASK_MNIST_FINEGRAIN,
)

import _pickle as pickle
import chex
import jax
import jax.random as jrandom
import numpy as np
import os
import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms
import torch

import jaxl.transforms as jaxl_transforms


def construct_mnist(
    save_path: str,
    task_name: str = None,
    task_config: SimpleNamespace = None,
    train: bool = True,
) -> Dataset:
    """
    Constructs a customized MNIST dataset.

    :param save_path: the path to store the MNIST dataset
    :param task_name: the task to construct
    :param task_config: the task configuration
    :param train: the train split of the dataset
    :type save_path: str
    :type task_name: str:  (Default value = None)
    :type task_config: SimpleNamespace:  (Default value = None)
    :type train: bool:  (Default value = True)
    :return: Customized MNIST dataset
    :rtype: Dataset

    """
    assert (
        task_name is None or task_name in VALID_MNIST_TASKS
    ), f"{task_name} is not supported (one of {VALID_MNIST_TASKS})"
    target_transform = None

    if task_name is None:
        # By default, the MNIST task will be normalized to be between 0 to 1.
        return torch_datasets.MNIST(
            save_path,
            train=train,
            download=True,
            transform=jaxl_transforms.DefaultPILToImageTransform(),
            target_transform=target_transform,
        )
    elif task_name == CONST_MULTITASK_MNIST_FINEGRAIN:
        return MultitaskMNISTFineGrain(
            dataset=torch_datasets.MNIST(
                save_path,
                train=train,
                download=True,
                transform=jaxl_transforms.StandardImageTransform(),
                target_transform=target_transform,
            ),
            num_sequences=task_config.num_sequences,
            sequence_length=task_config.sequence_length,
            save_path=task_config.save_path,
        )
    else:
        raise ValueError(f"{task_name} is invalid (one of {VALID_MNIST_TASKS})")


class MultitaskMNISTFineGrain(Dataset):
    """
    The dataset contains multiple ND (fixed) linear classification problems.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_sequences: int,
        sequence_length: int,
        seed: int = 0,
        save_path: str = None,
    ):
        self._dataset = dataset
        self._sequence_length = sequence_length

        if save_path is None or not os.path.isfile(save_path):
            self.sample_idxes = self._generate_data(
                dataset=dataset,
                num_sequences=num_sequences,
                seed=seed,
            )
            if save_path is not None:
                print("Saving to {}".format(save_path))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                pickle.dump(
                    self.sample_idxes,
                    open(save_path, "wb"),
                )
        else:
            print("Loading from {}".format(save_path))
            self.sample_idxes = pickle.load(
                open(save_path, "rb")
            )

    def _generate_data(
        self,
        dataset: Dataset,
        num_sequences: int,
        seed: int,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        keys = jrandom.split(jrandom.PRNGKey(seed), num=num_sequences)
        return jax.vmap(
            lambda key: jrandom.choice(key, np.arange(len(dataset)), shape=(self._sequence_length,))
        )(keys)

    @property
    def input_dim(self) -> chex.Array:
        return [*self._dataset.data[0].shape]

    @property
    def output_dim(self) -> chex.Array:
        return (10,)

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    def __len__(self):
        return len(self.sample_idxes)

    def __getitem__(self, idx):
        sample_idxes = self.sample_idxes[idx].tolist()
        inputs = self._dataset.transform(self._dataset.data[sample_idxes])
        outputs = np.eye(self.output_dim[0])[self._dataset.targets[sample_idxes]]
        return (inputs, outputs)
