import _pickle as pickle
import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import os

from torch.utils.data import Dataset
from typing import Any

from jaxl.constants import *


def make_tight_frame(
    save_path: str,
    num_classes: int = 1000,
    hidden_dim: int = 64,
    num_epochs: int = 20000,
    lr: float = 1e-4,
    seed: int = 0,
):
    """
    Generates an approximately tight frame.
    """
    frames = np.random.RandomState(seed).randn(num_classes, hidden_dim)

    @jax.jit
    def loss_fn(v):
        return jnp.sum(
            (jnp.sum(v[..., None] @ v[:, None], axis=0) - jnp.eye(hidden_dim)) ** 2
        )

    optimizer = optax.sgd(lr)
    opt_state = optimizer.init(frames)

    losses = []
    for ii in range(num_epochs):
        loss, grads = jax.value_and_grad(loss_fn)(frames)

        updates, opt_state = optimizer.update(grads, opt_state)
        frames = optax.apply_updates(frames, updates)
        if (ii + 1) % 1000 == 0:
            losses.append(loss)
            print(ii + 1, loss)
    pickle.dump(frames, open(save_path, "wb"))


def hierarchical_clustering(
    load_path: str,
) -> Any:
    """
    Perform hierarchical clustering to the tight frame.
    This algorithm forms cluster based on the cluster center similarity,
    where the similarity is defined to be the Euclidean distance between two "points".

    NOTE: Assumes the data is with shape (num_samples x dim)
    """
    assert os.path.isfile(load_path)

    data = pickle.load(open(load_path, "rb"))

    # NOTE: Use cluster center to compare cluster similarity
    def _hierarhical_clustering(
        data: chex.Array,
    ):
        if len(data) <= 1:
            return data
        
        pairwise_diff = data[:, None] - data[None, :] + np.diag




class TightFrameClassification(Dataset):
    def __init__(
        self,
        tight_frame_path: str,
        num_sequences: int,
        sequence_length: int,
        num_holdout: int,
        split: str,
        p_bursty: float,
        unique_classes: bool = False,
        random_label: bool = False,
        seed: int = 0,
    ):
        assert os.path.isfile(tight_frame_path)
        self.tight_frame = pickle.load(open(tight_frame_path, "rb"))
        is_train = int(split == CONST_TRAIN)
        data_gen_seed = jrandom.split(jrandom.PRNGKey(seed), 2)[is_train]
        data_gen_rng = np.random.RandomState(seed=data_gen_seed)

        offset = self.tight_frame.shape[0] - num_holdout
        if is_train:
            targets = data_gen_rng.choice(
                offset,
                size=num_sequences,
            )
        else:
            targets = data_gen_rng.choice(
                num_holdout,
                size=num_sequences,
            ) + offset

        self._data = {
            "targets": targets,
            "num_sequences": num_sequences,
            "sequence_length": sequence_length,
            "num_holdout": num_holdout,
            "num_classes": self.tight_frame.shape[0],
            "input_dim": self.tight_frame.shape[1],
            "split": split,
            "seed": seed,
            "is_bursty": data_gen_rng.rand(num_sequences) < p_bursty,
            "context_len": sequence_length - 1,
            "random_label": random_label,
        }

        self._unique_classes = unique_classes
        if is_train:
            self._num_classes = self._data["num_classes"] - num_holdout
            self._classes = np.arange(self._num_classes)
            self._offset = 0
        else:
            self._num_classes = num_holdout
            self._classes = np.arange(
                self._data["num_classes"] - num_holdout,
                self._data["num_classes"],
            )
            self._offset = offset

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
        is_bursty = self._data["is_bursty"][idx]
        sample_rng = np.random.RandomState(idx)
        label = self._data["targets"][idx]
        query = self.tight_frame[label]

        if is_bursty:
            label_idxes = []
            min_tokens = 6
            if self._data["sequence_length"] > min_tokens:
                label_idxes = sample_rng.choice(
                    self._classes,
                    size=(self._data["context_len"] - min_tokens),
                )
            repeated_distractor_label = sample_rng.choice(self._classes)
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
            if self._unique_classes:
                done = False
                while not done:
                    label_idxes = sample_rng.choice(
                        self._classes,
                        size=(self._data["context_len"]),
                        replace=False,
                    )
                    done = label not in label_idxes
            else:
                label_idxes = sample_rng.choice(
                    self._classes, size=(self._data["context_len"])
                )

        inputs = list(map(lambda ii: self.tight_frame[ii], label_idxes))
        inputs = np.concatenate(
            (*[context_input[None] for context_input in inputs], query[None])
        )

        labels = np.concatenate([label_idxes, [label]]) - self._offset

        if self._data["random_label"]:
            label_map = sample_rng.permutation(
                self._num_classes,
            )
            labels = label_map[labels]

        outputs = np.eye(self._data["num_classes"])[labels]
        return inputs, outputs


class TightFrameClassificationNShotKWay(Dataset):
    def __init__(
        self,
        tight_frame_path: str,
        num_sequences: int,
        sequence_length: int,
        num_holdout: int,
        split: str,
        k_way: int,
        seed: int = 0,
    ):
        assert os.path.isfile(tight_frame_path)

        context_len = sequence_length - 1
        assert (
            context_len % k_way == 0
        ), "context_len {} must be divisible by k_way {}".format(context_len, k_way)

        self.tight_frame = pickle.load(open(tight_frame_path, "rb"))
        is_train = int(split == CONST_TRAIN)
        data_gen_seed = jrandom.split(jrandom.PRNGKey(seed), 2)[is_train]
        data_gen_rng = np.random.RandomState(seed=data_gen_seed)

        targets = data_gen_rng.choice(
            self.tight_frame.shape[0] - num_holdout if is_train else num_holdout,
            size=num_sequences,
        )

        self._data = {
            "targets": targets,
            "num_sequences": num_sequences,
            "sequence_length": sequence_length,
            "num_holdout": num_holdout,
            "num_classes": self.tight_frame.shape[0],
            "input_dim": self.tight_frame.shape[1],
            "split": split,
            "seed": seed,
            "context_len": context_len,
            "k_way": k_way,
            "n_shot": context_len // k_way,
        }

        if is_train:
            self._num_classes = self._data["num_classes"] - num_holdout
            self._classes = np.arange(self._num_classes)
        else:
            self._num_classes = num_holdout
            self._classes = np.arange(
                self._data["num_classes"] - num_holdout,
                self._data["num_classes"],
            )

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
        sample_rng = np.random.RandomState(idx)

        label = sample_rng.choice(self._classes)
        query = self.tight_frame[label]

        while True:
            repeated_distractor_labels = sample_rng.choice(
                self._classes, size=self._data["k_way"] - 1, replace=True
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
                reps=(self._data["n_shot"]),
            )
        )

        inputs = list(map(lambda ii: self.tight_frame[ii], label_idxes))
        inputs = np.concatenate(
            (*[context_input[None] for context_input in inputs], query[None])
        )

        labels = np.concatenate([label_idxes, [label]])
        label_to_k_way = sample_rng.permutation(np.unique(labels))
        labels = np.array([np.argmax(label_to_k_way == label) for label in labels])

        outputs = np.eye(self._data["num_classes"])[labels]

        return (inputs, outputs)
