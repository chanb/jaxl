from torch.utils.data import Dataset
from types import SimpleNamespace
from typing import Tuple

from jaxl.constants import (
    VALID_MNIST_TASKS,
    CONST_MULTITASK_MNIST_FINEGRAIN,
    CONST_STRATIFIED_MULTITASK_MNIST_FINEGRAIN,
    CONST_MULTITASK_MNIST_RANDOM_BINARY,
    CONST_MULTITASK_MNIST_BURSTY,
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

    input_transform = jaxl_transforms.DefaultPILToImageTransform()
    if task_config and getattr(task_config, "augmentation", False):
        import torchvision.transforms as torch_transforms

        transforms = [
            jaxl_transforms.DefaultPILToImageTransform(scale=1.0),
            jaxl_transforms.GaussianNoise(0.0, task_config.noise_scale),
            torch_transforms.Normalize(0, 255.0),
        ]
        input_transform = torch_transforms.Compose(transforms)

    if task_name is None:
        # By default, the MNIST task will be normalized to be between 0 to 1.
        return torch_datasets.MNIST(
            save_path,
            train=train,
            download=True,
            transform=input_transform,
            target_transform=target_transform,
        )
    elif task_name == CONST_MULTITASK_MNIST_FINEGRAIN:
        return MultitaskMNISTFineGrain(
            dataset=torch_datasets.MNIST(
                save_path,
                train=train,
                download=True,
                transform=input_transform,
                target_transform=target_transform,
            ),
            num_sequences=task_config.num_sequences,
            sequence_length=task_config.sequence_length,
            random_label=getattr(task_config, "random_label", False),
            save_dir=task_config.save_dir,
        )
    elif task_name == CONST_STRATIFIED_MULTITASK_MNIST_FINEGRAIN:
        return StratifiedMultitaskMNISTFineGrain(
            dataset=torch_datasets.MNIST(
                save_path,
                train=train,
                download=True,
                transform=input_transform,
                target_transform=target_transform,
            ),
            num_sequences=task_config.num_sequences,
            num_queries=task_config.num_queries,
            random_label=getattr(task_config, "random_label", False),
            save_dir=task_config.save_dir,
        )
    elif task_name == CONST_MULTITASK_MNIST_RANDOM_BINARY:
        return MultitaskMNISTRandomBinary(
            dataset=torch_datasets.MNIST(
                save_path,
                train=train,
                download=True,
                transform=input_transform,
                target_transform=target_transform,
            ),
            num_sequences=task_config.num_sequences,
            sequence_length=task_config.sequence_length,
            save_dir=task_config.save_dir,
        )
    elif task_name == CONST_MULTITASK_MNIST_BURSTY:
        return MultitaskMNISTBursty(
            dataset=torch_datasets.MNIST(
                save_path,
                train=train,
                download=True,
                transform=input_transform,
                target_transform=target_transform,
            ),
            num_sequences=task_config.num_sequences,
            sequence_length=task_config.sequence_length,
            save_dir=task_config.save_dir,
        )
    else:
        raise ValueError(f"{task_name} is invalid (one of {VALID_MNIST_TASKS})")


class MultitaskMNISTFineGrain(Dataset):
    """
    The dataset contains a sequence-input MNIST problem.
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
        dataset_name = "mnist_finegrain-train_{}-num_sequences_{}-sequence_length_{}-random_label_{}-seed_{}.pkl".format(
            dataset.train,
            num_sequences,
            sequence_length,
            random_label,
            seed,
        )
        loaded, data = maybe_load_dataset(save_dir, dataset_name)

        if not loaded:
            num_classes = 10
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
                "train": dataset.train,
                "input_shape": [*dataset[0][0].shape],
                "num_classes": num_classes,
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
        return (self._data["num_classes"],)

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
        outputs = np.eye(self._data["num_classes"])[labels]
        return (inputs, outputs)


class StratifiedMultitaskMNISTFineGrain(Dataset):
    """
    The dataset contains a sequence-input MNIST problem with stratified sampling,
    such that each class is included in the sequence
    """

    def __init__(
        self,
        dataset: Dataset,
        num_sequences: int,
        num_queries: int,
        seed: int = 0,
        random_label: bool = False,
        save_dir: str = None,
    ):
        dataset_name = "mnist_stratified_finegrain-train_{}-num_sequences_{}-num_queries_{}-random_label_{}-seed_{}.pkl".format(
            dataset.train,
            num_sequences,
            num_queries,
            random_label,
            seed,
        )
        loaded, data = maybe_load_dataset(save_dir, dataset_name)

        if not loaded:
            _, counts = np.unique(dataset.targets, return_counts=True)
            min_num_per_class = np.min(counts)
            num_classes = 10
            label_to_idx = np.vstack(
                [
                    np.where(dataset.targets == class_i)[0][:min_num_per_class]
                    for class_i in range(num_classes)
                ]
            )
            (
                context_idxes,
                query_idxes,
                swap_idxes,
                label_map,
            ) = self._generate_data(
                dataset=dataset,
                num_sequences=num_sequences,
                num_queries=num_queries,
                min_num_per_class=min_num_per_class,
                num_classes=num_classes,
                random_label=random_label,
                seed=seed,
            )

            data = {
                "context_idxes": context_idxes,
                "query_idxes": query_idxes,
                "swap_idxes": swap_idxes,
                "label_map": label_map,
                "num_sequences": num_sequences,
                "num_queries": num_queries,
                "random_label": random_label,
                "train": dataset.train,
                "input_shape": [*dataset[0][0].shape],
                "num_classes": num_classes,
                "min_num_per_class": min_num_per_class,
                "label_to_idx": label_to_idx,
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
        num_queries: int,
        min_num_per_class: int,
        num_classes: int,
        random_label: bool,
        seed: int,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        print("Generating Data")
        sample_key, label_key = jrandom.split(jrandom.PRNGKey(seed))
        sample_rng = np.random.RandomState(sample_key)
        label_rng = np.random.RandomState(label_key)

        query_idxes = sample_rng.choice(
            np.arange(len(dataset)), size=(num_sequences, num_queries)
        )

        context_idxes = sample_rng.choice(
            np.arange(min_num_per_class), size=(num_sequences, num_classes)
        )

        label_map = np.tile(np.arange(num_classes), reps=(num_sequences, 1))

        swap_idxes = np.apply_along_axis(sample_rng.permutation, axis=1, arr=label_map)

        if random_label:
            label_map = np.apply_along_axis(
                label_rng.permutation, axis=1, arr=label_map
            )

        return context_idxes, query_idxes, swap_idxes, label_map

    @property
    def input_dim(self) -> chex.Array:
        return self._data["input_shape"]

    @property
    def output_dim(self) -> chex.Array:
        return (self._data["num_classes"],)

    @property
    def num_queries(self) -> int:
        return self._data["num_queries"]

    def __len__(self):
        return self._data["num_sequences"]

    def __getitem__(self, idx):
        context_idxes = self._data["context_idxes"][idx]
        context_idxes = np.take_along_axis(
            self._data["label_to_idx"], context_idxes[:, None], axis=1
        ).flatten()
        context_inputs, context_outputs = zip(
            *list(map(lambda ii: self._dataset[ii], context_idxes))
        )
        context_inputs = np.concatenate(
            [context_input[None] for context_input in context_inputs]
        )
        context_outputs = np.array(context_outputs)
        context_outputs = self._data["label_map"][idx][context_outputs]

        context_inputs = context_inputs[self._data["swap_idxes"][idx]]
        context_outputs = context_outputs[self._data["swap_idxes"][idx]]
        context_outputs = np.eye(self._data["num_classes"])[context_outputs]

        query_idxes = self._data["query_idxes"][idx].tolist()
        queries, labels = zip(*list(map(lambda ii: self._dataset[ii], query_idxes)))
        queries = np.concatenate([query[None] for query in queries])
        labels = np.array(labels)
        labels = self._data["label_map"][idx][labels]
        outputs = np.eye(self._data["num_classes"])[labels]

        return (context_inputs, context_outputs, queries, outputs)


class MultitaskMNISTRandomBinary(Dataset):
    """
    The dataset contains a sequence-input MNIST, with labels relabelled to {0, 1}.
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
            self.sample_idxes, self.label_map = self._generate_data(
                dataset=dataset,
                num_sequences=num_sequences,
                seed=seed,
            )
            if save_path is not None:
                print("Saving to {}".format(save_path))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                pickle.dump(
                    (self.sample_idxes, self.label_map),
                    open(save_path, "wb"),
                )
        else:
            print("Loading from {}".format(save_path))
            (self.sample_idxes, self.label_map) = pickle.load(open(save_path, "rb"))

    def _generate_data(
        self,
        dataset: Dataset,
        num_sequences: int,
        seed: int,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        sample_key, label_key = jrandom.split(jrandom.PRNGKey(seed))
        sample_rng = np.random.RandomState(sample_key)
        label_rng = np.random.RandomState(label_key)

        sample_idxes = sample_rng.choice(
            np.arange(len(dataset)), size=(num_sequences, self._sequence_length)
        )

        label_map = np.zeros((10,), dtype=np.int32)
        ones = label_rng.choice(
            np.arange(10),
            replace=False,
            size=(5,),
        )
        label_map[ones] = 1
        return sample_idxes, label_map

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
        outputs = np.eye(self.output_dim[0])[
            self.label_map[self._dataset.targets[sample_idxes]]
        ]
        return (inputs, outputs)


class MultitaskMNISTBursty(Dataset):
    """
    The dataset contains a sequence-input MNIST problem, following Chan et al. 2022.
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
        dataset_name = "omniglot_bursty-p_bursty_{}-train_{}-num_sequences_{}-sequence_length_{}-random_label_{}-seed_{}.pkl".format(
            p_bursty,
            dataset.train,
            num_sequences,
            sequence_length,
            random_label,
            seed,
        )
        loaded, data = maybe_load_dataset(save_dir, dataset_name)

        if not loaded:
            num_classes = 10
            _, counts = np.unique(dataset.targets, return_counts=True)
            min_num_per_class = np.min(counts)
            context_len = sequence_length - 1
            label_to_idx = np.vstack(
                [
                    np.where(dataset.targets == class_i)[0][:min_num_per_class]
                    for class_i in range(num_classes)
                ]
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
                "train": dataset.train,
                "input_shape": [*dataset[0][0].shape],
                "num_classes": num_classes,
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
        return (self._data["num_classes"],)

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
            label_map = sample_rng.choice(
                self._data["num_classes"],
                size=(self._data["sequence_length"],),
                replace=False,
            )
            labels = label_map[labels]

        if self._remap:
            labels = labels % 2
        outputs = np.eye(self._data["num_classes"])[labels]

        return (inputs, outputs)
