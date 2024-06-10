# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Dataset utilities."""

import chex
import tensorflow as tf

from types import SimpleNamespace

from jaxl.datasets.tf_omniglot.data_generators import (
    OmniglotDatasetForSampling,
    SeqGenerator,
)


class TFDataset:
    def __init__(self, dataset, num_classes, input_shape, sequence_length):
        self._dataset = dataset
        self._num_classes = num_classes
        self._input_shape = input_shape
        self._sequence_length = sequence_length

    @property
    def dataset(self):
        return self._dataset

    @property
    def output_dim(self) -> chex.Array:
        return (self._num_classes,)

    @property
    def input_dim(self) -> chex.Array:
        return self._input_shape

    @property
    def sequence_length(self) -> int:
        return self._sequence_length


def get_omniglot_seq_generator(
    dataset_kwargs: SimpleNamespace,
    seed: int,
):
    task_name = dataset_kwargs.task_name
    task_config = dataset_kwargs.task_config

    data_generator_factory = SeqGenerator(
        OmniglotDatasetForSampling(
            omniglot_split="all",
            exemplars=task_config.exemplars,
            augment_images=False,
        ),
        n_rare_classes=1603,  # 1623 - 20
        n_common_classes=10,
        n_holdout_classes=10,
        zipf_exponent=0.0,
        use_zipf_for_common_rare=False,
        noise_scale=task_config.noise_scale,
        preserve_ordering_every_n=None,
        random_seed=seed,
    )

    if task_name == "bursty":
        seq_generator = data_generator_factory.get_bursty_seq

        seq_config = (
            dataset_kwargs.sequence_length,
            dataset_kwargs.bursty_shots,
            dataset_kwargs.ways,
            dataset_kwargs.p_bursty,
            0.0,
            1.0,
            "zipfian",
            "ordered",
            "ordered",
            False,
            False,
        )
    elif task_name == "fewshot_holdout":
        seq_generator = data_generator_factory.get_fewshot_seq
        seq_config = (
            "holdout",
            dataset_kwargs.fs_shots,
            dataset_kwargs.ways,
            "unfixed",
            False,
            False,
        )
    elif task_name == "no_support":
        seq_generator = data_generator_factory.get_no_support_seq
        all_unique = False
        seq_config = (
            "zipfian",
            dataset_kwargs.sequence_length,
            all_unique,
            "ordered",
            False,
        )
    else:
        raise NotImplementedError

    example_shape = (dataset_kwargs.sequence_length, 105, 105, 1)
    example_dtype = tf.dtypes.float32

    dataset = tf.data.Dataset.from_generator(
        seq_generator,
        args=seq_config,
        output_signature={
            "example": tf.TensorSpec(shape=example_shape, dtype=example_dtype),
            "label": tf.TensorSpec(
                shape=(dataset_kwargs.sequence_length,), dtype=tf.dtypes.int32
            ),
            "is_rare": tf.TensorSpec(
                shape=(dataset_kwargs.sequence_length,), dtype=tf.dtypes.int32
            ),
        },
    )
    return TFDataset(
        dataset,
        1623,
        (105, 105, 1),
        dataset_kwargs.sequence_length,
    )


def prepare_seqs_for_transformer(
    ds, use_constant_labels=False, interleave_targets=True, downsample=False
):
    """Convert example and label sequences for use by the transformer.

    Args:
      ds: A tf.data.Dataset where each example contains
        'example': a batch of examples with shape
                   (batch_size, seq_len, height, width, channels)
        'label': a batch of labels with shape
                 (batch_size, seq_len)
      use_constant_labels: Whether to use target labels of all ones, instead of
        the true labels.
      interleave_targets: Whether to create targets consisting of alternating
        [label, 0, label, 0, ...] sequences, or just [label, label, ...]
      downsample: Whether to downsample images.

    Returns:
      A new tf.data.Dataset where each example contains
        'examples': a batch of examples
            for images: (batch_size, seq_len, height, width, channels) tf.float32
            for integers: (batch_size, seq_len) tf.int32
        'labels': a batch of labels (batch_size, seq_len) tf.int32
        'target': a batch of labels (batch_size, final_seq_len) tf.int32
                  where final_seq_len = (seq_len*2 - 1) if interleave_targets is
                  True, otherwise final_seq_len = seq_len
    """

    def _convert_dict(example):
        # (dims: B:batch, SS:original seqlen, H:height, W:width, C:channels)
        is_image = len(example["example"].shape) == 5

        # Cast the examples into the correct shape and tf datatype.
        if is_image:
            examples = tf.cast(example["example"], tf.float32)  # (B,SS,H,W,C)
            if downsample:
                examples = tf.map_fn(
                    lambda batch: tf.image.resize(batch, [28, 28]), examples
                )
        else:
            examples = tf.cast(example["example"], tf.int32)  # (B, SS)

        # Cast the labels into the correct tf datatype.
        if use_constant_labels:
            labels = tf.ones_like(example["label"], tf.int32)
        else:
            labels = tf.cast(example["label"], tf.int32)  # (B,SS)
        seq_len = labels.shape[-1]

        # Create the target sequence.
        if interleave_targets:
            # Alternating labels with zeros, e.g. [label, 0, label, 0, ...].
            zeros = tf.zeros_like(labels)
            target = tf.stack((labels[..., None], zeros[..., None]), axis=-1)
            target = tf.reshape(target, [-1, seq_len * 2])[:, :-1]  # (B,SS*2-1)
        else:
            # Just use the original sequence of labels, e.g. [label, label, ...]
            target = labels  # (B,SS)

        ret_dict = {"examples": examples, "labels": labels, "target": target}
        return tf.data.Dataset.from_tensors(ret_dict)

    return ds.flat_map(_convert_dict)


def prepare_seqs_for_transformer_jaxl(ds, num_classes: int):
    """Convert example and label sequences for use by the transformer."""

    def _convert_dict(example):
        # (dims: B:batch, SS:original seqlen, H:height, W:width, C:channels)
        is_image = len(example["example"].shape) == 5

        # Cast the examples into the correct shape and tf datatype.
        examples = tf.cast(example["example"], tf.float32)

        # Cast the labels into the correct tf datatype.
        labels = tf.cast(example["label"], tf.int32)  # (B,SS)

        # Just use the original sequence of labels, e.g. [label, label, ...]
        targets = tf.one_hot(labels, num_classes)  # (B,SS)

        # ret_dict = {"examples": examples, "labels": labels, "target": targets}
        ret_dict = {
            "context_inputs": examples[:, :-1],
            "context_outputs": targets[:, :-1],
            "queries": examples[:, -1:],
            "outputs": targets[:, -1],
        }
        return tf.data.Dataset.from_tensors(ret_dict)

    return ds.flat_map(_convert_dict)
