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


def sample_from_sphere_surface(
    save_path: str,
    num_classes: int = 1000,
    hidden_dim: int = 64,
    seed: int = 0,
):
    samples = np.random.RandomState(seed).randn(num_classes, hidden_dim)
    sample_norms = np.linalg.norm(samples, axis=-1, keepdims=True)
    pickle.dump(samples / sample_norms, open(save_path, "wb"))


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


# TODO: Try something like a decision tree split?
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
            return {
                "curr_cluster": data,
                "next_cluster": None,
            }

        pairwise_diff = np.sum((data[:, None] - data[None, :]) ** 2, axis=-1)
        pairwise_diff += np.diag(np.full(len(data), np.inf))
        min_idx = np.argmin(pairwise_diff)
        data_i = data[min_idx % len(data)]
        data_j = data[min_idx // len(data)]
        curr_cluster = [data_i, data_j]
        new_center = np.mean(curr_cluster, axis=0)

        data[min_idx % len(data)] = np.inf
        data[min_idx // len(data)] = np.inf
        next_clusters = _hierarhical_clustering(np.concatenate(data, new_center))
        return {
            "curr_cluster": curr_cluster,
            "next_cluster": next_clusters,
        }

    return _hierarhical_clustering(data)


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
            targets = (
                data_gen_rng.choice(
                    num_holdout,
                    size=num_sequences,
                )
                + offset
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


class TightFrameAbstractClassification(Dataset):
    def __init__(
        self,
        tight_frame_path: str,
        num_sequences: int,
        sequence_length: int,
        num_holdout: int,
        split: str,
        abstraction: str = None,
        seed: int = 0,
    ):
        assert os.path.isfile(tight_frame_path)

        context_len = sequence_length - 1
        assert context_len % 2 == 0, "context_len {} must be divisible by 2".format(
            context_len
        )

        self.tight_frame = pickle.load(open(tight_frame_path, "rb"))
        is_train = int(split == CONST_TRAIN)

        labels = self._generate_labels(num_sequences, abstraction, is_train, seed)

        self._data = {
            "labels": labels,
            "num_sequences": num_sequences,
            "sequence_length": sequence_length,
            "num_holdout": num_holdout,
            "num_classes": self.tight_frame.shape[0],
            "input_dim": self.tight_frame.shape[1],
            "split": split,
            "seed": seed,
            "context_len": context_len,
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
        self._context_labels = np.array(
            [0] * (context_len // 2) + [1] * (context_len // 2), dtype=int
        )

    def _generate_labels(
        self, num_sequences: int, abstraction: str, is_train: bool, seed: int
    ):
        data_gen_seed = jrandom.split(jrandom.PRNGKey(seed), 2)[is_train]
        data_gen_rng = np.random.RandomState(seed=data_gen_seed)

        if abstraction.endswith("cos"):
            cos_threshold = float(abstraction.split("-")[0])
            random_boundaries = data_gen_rng.randn(
                num_sequences, self.tight_frame.shape[1]
            )
            positive_cap = (
                jax.vmap(
                    lambda boundary, tight_frame: tight_frame @ boundary,
                    in_axes=[0, None],
                )(random_boundaries, self.tight_frame)
            ) >= cos_threshold
            negative_cap = (
                jax.vmap(
                    lambda boundary, tight_frame: tight_frame @ boundary,
                    in_axes=[0, None],
                )(random_boundaries, self.tight_frame)
            ) <= -cos_threshold
            labels = np.full_like(positive_cap, -1)
            labels[positive_cap] = 1
            labels[negative_cap] = 0
            print(
                "num pos: {}, num neg: {}".format(
                    np.sum(positive_cap), np.sum(negative_cap)
                )
            )
        elif abstraction.endswith("l2"):
            num_closest = int(abstraction.split("-")[0])
            random_boundaries = data_gen_rng.randn(
                num_sequences, self.tight_frame.shape[1]
            )

            dists = np.sum(
                (random_boundaries[:, None] - self.tight_frame[None]) ** 2, axis=-1
            )
            sorted_dists = np.argsort(dists, axis=-1)[:, :num_closest]
            labels = np.zeros((num_sequences, len(self.tight_frame)), dtype=int)
            labels = jax.vmap(
                lambda per_seq_labels, idxes: per_seq_labels.at[idxes].set(1)
            )(labels, sorted_dists)
        else:
            labels = data_gen_rng.choice(2, size=(num_sequences, len(self.tight_frame)))

        return labels

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

        label_idxes = self._data["labels"][idx]

        # Generate query-output pair
        query_idx = sample_rng.choice(len(label_idxes))
        query = self.tight_frame[query_idx]
        label = label_idxes[query_idx]

        # Generate context pairs
        context_zeros = sample_rng.choice(
            np.where(label_idxes == 0)[0], size=(self._data["context_len"] // 2)
        )
        context_ones = sample_rng.choice(
            np.where(label_idxes == 1)[0], size=(self._data["context_len"] // 2)
        )

        shuffle_idxes = sample_rng.permutation(self._data["context_len"])
        context_idxes = np.concatenate((context_zeros, context_ones))[shuffle_idxes]
        labels = np.concatenate((self._context_labels[shuffle_idxes], [label]))

        inputs = list(map(lambda ii: self.tight_frame[ii], context_idxes))
        inputs = np.concatenate(
            (*[context_input[None] for context_input in inputs], query[None])
        )

        outputs = np.eye(self._data["num_classes"])[labels]

        return (inputs, outputs)


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
