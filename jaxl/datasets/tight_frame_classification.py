import _pickle as pickle
import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import os

from torch.utils.data import Dataset

from jaxl.constants import *


def make_tight_frame(
    save_path: str,
    num_classes: int = 1000,
    hidden_dim: int = 64,
    num_epochs: int = 20000,
    lr: float = 1e-4,
    seed: int = 0,
):
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
        frames = optax.apply_updates(a, updates)
        if (ii + 1) % 1000 == 0:
            losses.append(loss)
            print(ii + 1, loss)
    pickle.dump(frames, open(save_path, "wb"))


class TightFrameClassification(Dataset):
    def __init__(
        self,
        tight_frame_path: str,
        num_sequences: int,
        sequence_length: int,
        num_holdout: int,
        split: str,
        p_bursty: float,
        random_label: bool = False,
        seed: int = 0,
    ):
        assert os.path.isfile(tight_frame_path)
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
            "is_bursty": data_gen_rng.rand(num_sequences) < p_bursty,
            "context_len": sequence_length - 1,
            "random_label": random_label,
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
            label_idxes = sample_rng.choice(
                self._classes, size=(self._data["context_len"])
            )

        inputs = list(map(lambda ii: self.tight_frame[ii], label_idxes))
        inputs = np.concatenate(
            (*[context_input[None] for context_input in inputs], query[None])
        )

        labels = np.concatenate([label_idxes, [label]])

        if self._data["random_label"]:
            label_map = sample_rng.permutation(
                self._num_classes,
            )
            labels = label_map[labels]

        outputs = np.eye(self._data["num_classes"])[labels]
        return inputs, outputs
