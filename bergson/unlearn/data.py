import os

import torch
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import TrainerCallback


def is_debug():
    return int(os.environ.get("RANK", "0")) == 0


class AlternatingDataset(TorchIterableDataset):
    """
    Dataset that alternates between transfer and retain phases,
    windowing into the dataset to provide examples_per_epoch items per epoch.

    Must be used with the EpochUpdateCallback.
    """

    def __init__(
        self,
        first_ds,
        second_ds,
        rank,
        world_size,
        max_seq_len,
        num_phases,
        examples_per_phase,
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.max_seq_len = max_seq_len
        self.num_phases = num_phases
        self.examples_per_phase = examples_per_phase
        self._current_phase_idx = 0

        self.first_ds = first_ds.shard(
            num_shards=self.world_size, index=self.rank
        )
        self.second_ds = second_ds.shard(num_shards=self.world_size, index=self.rank)

        self.examples_per_rank = self.examples_per_phase // self.world_size
        remainder = self.examples_per_phase % self.world_size
        if self.rank < remainder:
            self.examples_per_rank += 1

        print("examples per rank", self.examples_per_rank)

    def set_phase(self, phase_idx):
        self._current_phase_idx = int(phase_idx)
    
    def get_current_phase(self):
        """Returns True if in transfer/forget phase (even phase_idx), False if retain phase."""
        return (self._current_phase_idx % 2) == 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 0:
            raise RuntimeError("AlternatingDataset does not support num_workers > 0.")

        # Determine dataset window start based on epoch idx and examples per epoch
        phase_epoch_idx = self._current_phase_idx // 2
        rank_start_idx = (phase_epoch_idx * self.examples_per_rank) % len(
            self.first_ds
        )
        window_indices = [
            (rank_start_idx + i) % len(self.first_ds)
            for i in range(self.examples_per_rank)
        ]

        is_first = (self._current_phase_idx % 2) == 0
        if is_debug():
            print("is first", is_first, self._current_phase_idx)
            
        if is_first:
            yield from self.first_ds.select(window_indices)
        else:
            yield from self.second_ds.select(window_indices)


class PhaseUpdateCallback(TrainerCallback):
    """
    Updates the dataset phase after every N steps.
    Required for AlternatingDataset.
    """
    def __init__(self, N: int):
        self.N = N
        self.i = 0

    def on_step_begin(self, args, state, control, **kwargs):
        step_idx = state.global_step
        if step_idx % self.N == 0 and step_idx > 0:
            train_dataloader = kwargs.get("train_dataloader")

            assert train_dataloader is not None, "Train dataloader is None."
            assert hasattr(
                train_dataloader, "dataset"
            ), "Train dataloader has no dataset attribute."
            assert hasattr(
                train_dataloader.dataset, "set_phase"
            ), "Dataset has no set_phase method."

            train_dataloader.dataset.set_phase(self.i)
            
            self.i += 1
