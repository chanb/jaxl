from torch.utils.data import Dataset

import numpy as np


class DatasetWrapper(Dataset):
    """Default dataset wrapper."""

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __getattr__(self, attr):
        return getattr(self._dataset, attr)

    def __len__(self):
        return len(self._dataset)


class FixedLengthTrajectoryDataset(DatasetWrapper):
    """Dataset for sequence modelling."""

    def __init__(self, dataset: Dataset, seq_length: int):
        super().__init__(dataset)
        self._seq_length = seq_length

    def __len__(self):
        return len(self._dataset) * self._dataset.trajectory_length

    def __getitem__(self, idx):
        traj_i = idx // self._dataset.trajectory_length
        timestep_i = idx % self._dataset.trajectory_length
        traj = self._dataset[traj_i]
        out_traj = np.zeros((self._seq_length, *traj.shape[1:]))
        out_pred = np.zeros((self._seq_length, *traj.shape[1:]))

        out_traj_start_idx = int(
            np.clip(
                self._seq_length - 1 - timestep_i, a_min=0, a_max=self._seq_length - 1
            )
        )
        traj_copy_start_idx = int(
            np.clip(timestep_i - self._seq_length + 1, a_min=0, a_max=timestep_i)
        )

        out_traj[out_traj_start_idx:] = traj[traj_copy_start_idx : timestep_i + 1]
        out_pred[out_traj_start_idx:] = traj[traj_copy_start_idx + 1 : timestep_i + 2]
        return out_traj, out_pred


class ContextDataset(DatasetWrapper):
    """Dataset for in-context learning."""

    def __init__(self, dataset: Dataset, context_len: int, skip_step: int = 1):
        super().__init__(dataset)
        self._context_len = context_len
        self._skip_step = skip_step
        self._total_seq_len = (context_len - 1) * skip_step

        # We subtract 1 from sequence length because we have context_len + 1, where 1 is the query
        self._seq_mod = (
            self._dataset.sequence_length - 1
        )

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
                self._context_len - 1 - timestep_i // self._skip_step,
                a_min=0,
                a_max=self._context_len - 1,
            )
        )

        seq_copy_start_idx = timestep_i - self._total_seq_len
        if seq_copy_start_idx < 0:
            seq_copy_start_idx = timestep_i % self._skip_step

        context_inputs[out_seq_start_idx:] = inputs[
            seq_copy_start_idx : timestep_i + 1 : self._skip_step
        ]
        context_outputs[out_seq_start_idx:] = outputs[
            seq_copy_start_idx : timestep_i + 1 : self._skip_step
        ]
        query = inputs[timestep_i + 1]
        output = outputs[timestep_i + 1]

        return context_inputs, context_outputs, query, output

    # TODO: Generate test query for visualization on context length, then use that for ICL plots
