from torch.utils.data import Dataset, DataLoader
from typing import Any
from types import SimpleNamespace

import chex
import numpy as np
import tensorflow_datasets as tfds


class DatasetWrapper(Dataset):
    """Default dataset wrapper."""

    def __init__(self, dataset: Any):
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)

    @property
    def input_dim(self) -> chex.Array:
        return self._dataset.input_dim

    @property
    def output_dim(self) -> chex.Array:
        return self._dataset.output_dim

    @property
    def sequence_length(self) -> int:
        return self._dataset.sequence_length

    def get_dataloader(
        self,
        config: SimpleNamespace,
    ) -> DataLoader:
        if isinstance(self._dataset, Dataset):
            batch_size = config.batch_size
            shuffle = getattr(config.dataset_config, "shuffle", True)
            drop_last = True
            num_workers = getattr(config.dataset_config, "num_workers", 0)

            return DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
            )
        else:
            tf_type = config.dataset_config.dataset_kwargs.tf_type
            if tf_type == "prepare_seqs_for_transformer":
                from jaxl.datasets.tf_omniglot.utils import (
                    prepare_seqs_for_transformer_jaxl,
                )

                shuffle_buffer_size = 100
                ds_seqs = self._dataset.dataset
                ds = ds_seqs.batch(config.batch_size).prefetch(
                    config.dataset_config.dataset_kwargs.num_workers
                )
                ds = prepare_seqs_for_transformer_jaxl(
                    ds,
                    self._dataset.output_dim[0],
                )
                ds = ds.repeat().shuffle(buffer_size=shuffle_buffer_size)
                return tfds.as_numpy(ds)
            else:
                raise NotImplementedError


class StandardSupervisedDataset(DatasetWrapper):
    """Dataset for standard supervised learning."""

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        input, _ = self._dataset[0]
        self._input_dim = input.shape
        self._output_dim = self._dataset.output_dim

    def __getitem__(self, idx):
        input, output = self._dataset[idx]
        return input, np.zeros((1,)), output, idx

    @property
    def output_dim(self) -> chex.Array:
        return self._output_dim

    @property
    def input_dim(self) -> chex.Array:
        return self._input_dim


class FixedLengthContextDataset(DatasetWrapper):
    """Fixed length dataset for in-context learning."""

    def __init__(self, dataset: Dataset, context_len: int):
        super().__init__(dataset)
        self._context_len = context_len

        # We subtract 1 from sequence length because we have context_len + 1, where 1 is the query
        self._seq_mod = self._dataset.sequence_length - context_len

    def __len__(self):
        return len(self._dataset) * self._seq_mod

    def __getitem__(self, idx):
        seq_i = idx // self._seq_mod
        inputs, outputs = self._dataset[seq_i]

        timestep_i = idx % self._seq_mod

        context_inputs = np.zeros((self._context_len, *inputs.shape[1:]))
        context_outputs = np.zeros((self._context_len, *outputs.shape[1:]))

        context_inputs = inputs[timestep_i : timestep_i + self._context_len]
        context_outputs = outputs[timestep_i : timestep_i + self._context_len]
        query = inputs[[timestep_i + self._context_len]]
        output = outputs[timestep_i + self._context_len]

        ret_dict = {
            "context_inputs": context_inputs,
            "context_outputs": context_outputs,
            "queries": query,
            "outputs": output,
        }

        # return context_inputs, context_outputs, query, output
        return ret_dict


class ContextDataset(DatasetWrapper):
    """Dataset for in-context learning."""

    def __init__(
        self, dataset: Dataset, context_len: int, include_query_class: bool = False
    ):
        super().__init__(dataset)
        self._context_len = context_len
        self._last_context_idx = context_len - 1
        self._include_query_class = include_query_class

        # We subtract 1 from sequence length because we have context_len + 1, where 1 is the query
        self._seq_mod = self._dataset.sequence_length - 1

    def __len__(self):
        return len(self._dataset) * self._seq_mod

    def __getitem__(self, idx):
        seq_i = idx // self._seq_mod
        inputs, outputs = self._dataset[seq_i]

        timestep_i = idx % self._seq_mod

        context_inputs = np.zeros((self._context_len, *inputs.shape[1:]))
        context_outputs = np.zeros((self._context_len, *outputs.shape[1:]))

        out_seq_start_idx = int(
            np.clip(
                self._last_context_idx - timestep_i,
                a_min=0,
                a_max=self._last_context_idx,
            )
        )

        seq_copy_start_idx = int(
            np.clip(timestep_i - self._last_context_idx, a_min=0, a_max=np.inf)
        )

        if self._include_query_class and out_seq_start_idx != 0:
            classes = np.argmax(outputs, axis=-1)
            output_class = classes[-1]
            match_idxes = classes == output_class
            match_idxes[-1] = False
            match_idxes = np.where(match_idxes)[0]

            if len(match_idxes):
                sample_rng = np.random.RandomState(idx)
                idx_to_put = sample_rng.choice(match_idxes)

                if not seq_copy_start_idx <= idx_to_put <= timestep_i:
                    swap_idx = sample_rng.randint(seq_copy_start_idx, timestep_i + 1)
                    inputs[swap_idx] = inputs[idx_to_put]
                    outputs[swap_idx] = outputs[idx_to_put]

        inputs = np.concatenate(
            (
                inputs[seq_copy_start_idx : timestep_i + 1],
                inputs[[-1]],
            )
        )

        outputs = np.concatenate(
            (
                outputs[seq_copy_start_idx : timestep_i + 1],
                outputs[[-1]],
            )
        )

        context_inputs[out_seq_start_idx:] = inputs[:-1]
        context_outputs[out_seq_start_idx:] = outputs[:-1]
        query = inputs[[-1]]
        output = outputs[-1]

        ret_dict = {
            "context_inputs": context_inputs,
            "context_outputs": context_outputs,
            "queries": query,
            "outputs": output,
        }

        # return context_inputs, context_outputs, query, output
        return ret_dict

    # TODO: Generate test query for visualization on context length, then use that for ICL plots
