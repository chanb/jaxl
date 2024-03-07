from torch.utils.data import Dataset
from types import SimpleNamespace
from typing import Tuple

from jaxl.constants import (
    VALID_OMNIGLOT_TASKS,
    CONST_MULTITASK_OMNIGLOT_FINEGRAIN,
    CONST_MULTITASK_OMNIGLOT_BURSTY,
)
from jaxl.datasets.utils import (
    maybe_save_dataset,
    maybe_load_dataset,
)

import chex
import jax.random as jrandom
import numpy as np
import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

import jaxl.transforms as jaxl_transforms


def construct_omniglot(
    save_path: str,
    task_name: str = None,
    task_config: SimpleNamespace = None,
    seed: int = 0,
    train: bool = True,
    remap: bool = False,
) -> Dataset:
    """
    Constructs a customized Omniglot dataset.

    :param save_path: the path to store the Omniglot dataset
    :param task_name: the task to construct
    :param task_config: the task configuration
    :param seed: the seed to generate the dataset
    :param train: the train split of the dataset
    :param remap: whether to do binary classification instead
    :type save_path: str
    :type task_name: str:  (Default value = None)
    :type task_config: SimpleNamespace:  (Default value = None)
    :type seed: int:  (Default value = 0)
    :type train: bool:  (Default value = True)
    :type remap: bool:  (Default value = False)
    :return: Customized Omniglot dataset
    :rtype: Dataset

    """
    assert (
        task_name is None or task_name in VALID_OMNIGLOT_TASKS
    ), f"{task_name} is not supported (one of {VALID_OMNIGLOT_TASKS})"
    target_transform = None

    transforms = [
        jaxl_transforms.DefaultPILToImageTransform(),
        jaxl_transforms.Transpose(axes=(1, 2, 0)),
    ]
    if task_config and getattr(task_config, "augmentation", False):
        transforms = [
            jaxl_transforms.DefaultPILToImageTransform(scale=1.0),
            jaxl_transforms.Transpose(axes=(1, 2, 0)),
            jaxl_transforms.GaussianNoise(0.0, task_config.noise_scale),
            torch_transforms.Normalize(0, 255.0),
        ]
    input_transform = torch_transforms.Compose(transforms)

    if task_name is None:
        # By default, the Omniglot task will be normalized to be between 0 to 1.
        return torch_datasets.Omniglot(
            save_path,
            background=train,
            download=True,
            transform=input_transform,
            target_transform=target_transform,
        )
    elif task_name == CONST_MULTITASK_OMNIGLOT_FINEGRAIN:
        return MultitaskOmniglotFineGrain(
            dataset=torch_datasets.Omniglot(
                save_path,
                background=train,
                download=True,
                transform=input_transform,
                target_transform=target_transform,
            ),
            num_sequences=task_config.num_sequences,
            sequence_length=task_config.sequence_length,
            seed=seed,
            remap=remap,
            random_label=getattr(task_config, "random_label", False),
            save_dir=task_config.save_dir,
        )
    elif task_name == CONST_MULTITASK_OMNIGLOT_BURSTY:
        return MultitaskOmniglotBursty(
            dataset=torch_datasets.Omniglot(
                save_path,
                background=train,
                download=True,
                transform=input_transform,
                target_transform=target_transform,
            ),
            num_sequences=task_config.num_sequences,
            sequence_length=task_config.sequence_length,
            p_bursty=task_config.p_bursty,
            seed=seed,
            remap=remap,
            random_label=getattr(task_config, "random_label", False),
            save_dir=task_config.save_dir,
        )
    else:
        raise ValueError(f"{task_name} is invalid (one of {VALID_OMNIGLOT_TASKS})")


class MultitaskOmniglotFineGrain(Dataset):
    """
    The dataset contains a sequence-input Omniglot problem.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_sequences: int,
        sequence_length: int,
        seed: int = 0,
        remap: bool = False,
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
            max_num_classes = 964
            num_classes = 964 if dataset.background else 659
            sample_idxes, label_map = self._generate_data(
                dataset=dataset,
                num_sequences=num_sequences,
                sequence_length=sequence_length,
                random_label=random_label,
                num_classes=num_classes,
                seed=seed,
            )

            data = {
                "sample_idxes": sample_idxes,
                "label_map": label_map,
                "num_sequences": num_sequences,
                "sequence_length": sequence_length,
                "random_label": random_label,
                "background": dataset.background,
                "input_shape": [*dataset[0][0].shape],
                "num_classes": num_classes,
                "max_num_classes": max_num_classes,
                "seed": seed,
            }
            maybe_save_dataset(
                data,
                save_dir,
                dataset_name,
            )

        self._dataset = dataset
        self._data = data
        self._remap = remap

    def _generate_data(
        self,
        dataset: Dataset,
        num_sequences: int,
        sequence_length: int,
        num_classes: int,
        random_label: bool,
        seed: int,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        print("Generating Data")
        sample_key, label_key = jrandom.split(jrandom.PRNGKey(seed))
        sample_rng = np.random.RandomState(sample_key)
        label_rng = np.random.RandomState(label_key)

        sample_idxes = sample_rng.choice(
            np.arange(len(dataset)), size=(num_sequences, sequence_length)
        )

        label_map = np.tile(np.arange(num_classes), reps=(num_sequences, 1))
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
        return (self._data["max_num_classes"],)

    @property
    def sequence_length(self) -> int:
        return self._data["sequence_length"]

    def __len__(self):
        return self._data["num_sequences"]

    def __getitem__(self, idx):
        sample_idxes = self._data["sample_idxes"][idx].tolist()
        inputs, labels = zip(*list(map(lambda ii: self._dataset[ii], sample_idxes)))
        inputs = np.concatenate([input[None] for input in inputs])
        labels = np.array(labels)
        labels = self._data["label_map"][idx][labels]
        if self._remap:
            labels = labels % 2
        outputs = np.eye(self._data["max_num_classes"])[labels]
        return (inputs, outputs)


class MultitaskOmniglotBursty(Dataset):
    """
    The dataset contains a sequence-input Omniglot problem, following Chan et al. 2022.
    The query class is repeated 3 times, and one of the remaining classes is also repeated 3 times.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_sequences: int,
        sequence_length: int,
        seed: int = 0,
        p_bursty: float = 1,
        remap: bool = False,
        random_label: bool = False,
        save_dir: str = None,
    ):
        dataset_name = "omniglot_bursty-p_bursty_{}-background_{}-num_sequences_{}-sequence_length_{}-random_label_{}-seed_{}.pkl".format(
            p_bursty,
            dataset.background,
            num_sequences,
            sequence_length,
            random_label,
            seed,
        )
        loaded, data = maybe_load_dataset(save_dir, dataset_name)

        if not loaded:
            max_num_classes = 964
            num_classes = 964 if dataset.background else 659
            min_num_per_class = 20
            context_len = sequence_length - 1
            label_to_idx = np.arange(num_classes * min_num_per_class).reshape(
                (num_classes, min_num_per_class)
            )

            (
                context_idxes,
                query_idxes,
                is_bursty,
            ) = self._generate_data(
                dataset=dataset,
                num_sequences=num_sequences,
                context_len=context_len,
                p_bursty=p_bursty,
                min_num_per_class=min_num_per_class,
                seed=seed,
            )

            data = {
                "context_idxes": context_idxes,
                "query_idxes": query_idxes,
                "num_sequences": num_sequences,
                "sequence_length": sequence_length,
                "context_len": context_len,
                "random_label": random_label,
                "background": dataset.background,
                "input_shape": [*dataset[0][0].shape],
                "num_classes": num_classes,
                "max_num_classes": max_num_classes,
                "min_num_per_class": min_num_per_class,
                "label_to_idx": label_to_idx,
                "seed": seed,
                "p_bursty": p_bursty,
                "is_bursty": is_bursty,
            }
            maybe_save_dataset(
                data,
                save_dir,
                dataset_name,
            )

        self._dataset = dataset
        self._data = data
        self._remap = remap

    def _generate_data(
        self,
        dataset: Dataset,
        num_sequences: int,
        context_len: int,
        min_num_per_class: int,
        p_bursty: float,
        seed: int,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        print("Generating Data")
        sample_key, _ = jrandom.split(jrandom.PRNGKey(seed))
        sample_rng = np.random.RandomState(sample_key)

        is_bursty = sample_rng.rand(num_sequences) < p_bursty

        query_idxes = sample_rng.choice(np.arange(len(dataset)), size=(num_sequences,))

        context_idxes = sample_rng.choice(
            np.arange(min_num_per_class), size=(num_sequences, context_len)
        )

        return context_idxes, query_idxes, is_bursty

    @property
    def input_dim(self) -> chex.Array:
        return self._data["input_shape"]

    @property
    def output_dim(self) -> chex.Array:
        return (self._data["max_num_classes"],)

    @property
    def sequence_length(self) -> int:
        return self._data["sequence_length"]

    def __len__(self):
        return self._data["num_sequences"]

    def __getitem__(self, idx):
        is_bursty = self._data["is_bursty"][idx]
        sample_rng = np.random.RandomState(idx)

        query_idx = self._data["query_idxes"][idx]
        query, label = self._dataset[query_idx]

        if is_bursty:
            label_idxes = []
            min_tokens = 6
            if self._data["sequence_length"] > min_tokens:
                label_idxes = sample_rng.choice(
                    self._data["num_classes"],
                    size=(self._data["context_len"] - min_tokens),
                )
            repeated_distractor_label = sample_rng.choice(self._data["num_classes"])
            label_idxes = sample_rng.permutation(
                np.concatenate(
                    [
                        [label] * 3,
                        [repeated_distractor_label] * 3,
                        label_idxes,
                    ]
                )[: self._data["context_len"]]
            )
        else:
            label_idxes = sample_rng.choice(
                self._data["num_classes"], size=(self._data["context_len"])
            )

        context_idxes = self._data["context_idxes"][idx]
        context_idxes = np.take_along_axis(
            self._data["label_to_idx"][label_idxes], context_idxes[:, None], axis=1
        ).flatten()
        inputs, _ = zip(*list(map(lambda ii: self._dataset[ii], context_idxes)))
        inputs = np.concatenate(
            (*[context_input[None] for context_input in inputs], query[None])
        )
        labels = np.concatenate([label_idxes, [label]])

        if self._data["random_label"]:
            label_map = sample_rng.permutation(
                self._data["num_classes"],
            )
            labels = label_map[labels]

        if self._remap:
            labels = labels % 2
        outputs = np.eye(self._data["max_num_classes"])[labels]

        return (inputs, outputs)


class MultitaskOmniglotNWayKShot(Dataset):
    """
    The dataset contains a sequence-input Omniglot N-shot K-way problem, following Chan et al. 2022.
    The query class is repeated N times, and K - 1 of the remaining classes are also repeated N times.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_sequences: int,
        sequence_length: int,
        k_way: int,
        seed: int = 0,
        save_dir: str = None,
    ):
        dataset_name = "omniglot_n_shot_k_way-k_way_{}-background_{}-num_sequences_{}-sequence_length_{}-seed_{}.pkl".format(
            k_way,
            dataset.background,
            num_sequences,
            sequence_length,
            seed,
        )
        loaded, data = maybe_load_dataset(save_dir, dataset_name)

        if not loaded:
            max_num_classes = 964
            num_classes = 964 if dataset.background else 659
            min_num_per_class = 20
            context_len = sequence_length - 1
            assert (
                context_len % k_way == 0
            ), "context_len {} must be divisible by k_way {}".format(context_len, k_way)
            label_to_idx = np.arange(num_classes * min_num_per_class).reshape(
                (num_classes, min_num_per_class)
            )

            (
                context_idxes,
                query_idxes,
            ) = self._generate_data(
                dataset=dataset,
                num_sequences=num_sequences,
                context_len=context_len,
                min_num_per_class=min_num_per_class,
                seed=seed,
            )

            data = {
                "context_idxes": context_idxes,
                "query_idxes": query_idxes,
                "num_sequences": num_sequences,
                "sequence_length": sequence_length,
                "context_len": context_len,
                "background": dataset.background,
                "input_shape": [*dataset[0][0].shape],
                "num_classes": num_classes,
                "max_num_classes": max_num_classes,
                "min_num_per_class": min_num_per_class,
                "label_to_idx": label_to_idx,
                "seed": seed,
                "k_way": k_way,
                "n_shot": context_len // k_way,
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
        context_len: int,
        min_num_per_class: int,
        seed: int,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        print("Generating Data")
        sample_key, _ = jrandom.split(jrandom.PRNGKey(seed))
        sample_rng = np.random.RandomState(sample_key)

        query_idxes = sample_rng.choice(np.arange(len(dataset)), size=(num_sequences,))

        context_idxes = sample_rng.choice(
            np.arange(min_num_per_class), size=(num_sequences, context_len)
        )

        return context_idxes, query_idxes

    @property
    def input_dim(self) -> chex.Array:
        return self._data["input_shape"]

    @property
    def output_dim(self) -> chex.Array:
        return (self._data["max_num_classes"],)

    @property
    def sequence_length(self) -> int:
        return self._data["sequence_length"]

    def __len__(self):
        return self._data["num_sequences"]

    def __getitem__(self, idx):
        sample_rng = np.random.RandomState(idx)

        query_idx = self._data["query_idxes"][idx]
        query, label = self._dataset[query_idx]

        while True:
            repeated_distractor_labels = sample_rng.choice(
                self._data["num_classes"], size=self._data["n_shot"] - 1, replace=True
            )
            if label not in repeated_distractor_labels:
                break

        label_idxes = sample_rng.permutation(
            np.tile(
                np.concatenate(
                    [
                        [label],
                        [*repeated_distractor_labels],
                    ]
                ),
                reps=(self._data["k_ways"]),
            )
        )

        context_idxes = self._data["context_idxes"][idx]
        context_idxes = np.take_along_axis(
            self._data["label_to_idx"][label_idxes], context_idxes[:, None], axis=1
        ).flatten()
        inputs, _ = zip(*list(map(lambda ii: self._dataset[ii], context_idxes)))
        inputs = np.concatenate(
            (*[context_input[None] for context_input in inputs], query[None])
        )
        labels = np.concatenate([label_idxes, [label]])
        label_to_k_way = sample_rng.permutation(np.unique(labels))
        labels = np.array([np.argmax(label_to_k_way == label) for label in labels])

        outputs = np.eye(self._data["max_num_classes"])[labels]

        return (inputs, outputs)
