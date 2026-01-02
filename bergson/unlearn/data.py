import os
import re

import torch
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import TrainerCallback


def is_debug():
    return int(os.environ.get("RANK", "0")) == 0


def transfer_generator(combined_set, max_seq_len):
    """Generator for transfer phase: pairs source with target."""
    combined_iter = iter(combined_set)

    for sample in combined_iter:
        source_ids = sample["source_input_ids"]
        target_ids = sample["target_input_ids"]

        yield {
            "source_input_ids": (
                source_ids[:max_seq_len]
                if isinstance(source_ids, list)
                else source_ids[:max_seq_len].tolist()
            ),
            "source_attention_mask": [1] * min(len(source_ids), max_seq_len),
            "target_input_ids": (
                target_ids[:max_seq_len]
                if isinstance(target_ids, list)
                else target_ids[:max_seq_len].tolist()
            ),
            "target_attention_mask": [1] * min(len(target_ids), max_seq_len),
        }


def retain_generator(ds, max_seq_len):
    """Generator for retain phase: standard language modeling."""
    for sample in ds:
        input_ids = sample.get("input_ids", sample.get("text", []))

        if isinstance(input_ids, str):
            raise ValueError(
                "bio-retain dataset must be tokenized (have 'input_ids' field)"
            )

        yield {
            "input_ids": (
                input_ids[:max_seq_len]
                if isinstance(input_ids, list)
                else input_ids[:max_seq_len].tolist()
            ),
            "attention_mask": [1] * min(len(input_ids), max_seq_len),
        }


class AlternatingDataset(TorchIterableDataset):
    """
    Dataset that alternates between transfer and retain phases,
    windowing into the dataset to provide examples_per_epoch items per epoch.

    Must be used with the EpochUpdateCallback.
    """

    def __init__(
        self,
        transfer_ds,
        retain_set,
        rank,
        world_size,
        max_seq_len,
        num_phases,
        examples_per_phase,
        first_generator=transfer_generator,
        second_generator=retain_generator

    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.max_seq_len = max_seq_len
        self.num_phases = num_phases
        self.examples_per_phase = examples_per_phase
        self._current_phase_idx = 0

        self.first_generator = first_generator
        self.second_generator = second_generator

        self.transfer_ds = transfer_ds.shard(
            num_shards=self.world_size, index=self.rank
        )

        self.retain_ds = retain_set.shard(num_shards=self.world_size, index=self.rank)

    def set_phase(self, phase_idx):
        self._current_phase_idx = int(phase_idx)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 0:
            raise RuntimeError("AlternatingDataset does not support num_workers > 0.")

        # Determine dataset window start based on epoch idx and examples per epoch
        phase_epoch_idx = self._current_phase_idx // 2
        
        examples_per_rank = self.examples_per_phase // self.world_size
        print("examples per rank", examples_per_rank)
        remainder = self.examples_per_phase % self.world_size
        if self.rank < remainder:
            examples_per_rank += 1
        rank_start_idx = (phase_epoch_idx * examples_per_rank) % len(
            self.transfer_ds
        )
        window_indices = [
            (rank_start_idx + i) % len(self.transfer_ds)
            for i in range(examples_per_rank)
        ]

        is_transfer = (self._current_phase_idx % 2) == 0
        if is_debug():
            print("is transfer", is_transfer)
            
        if is_transfer:
            transfer_ds_window = self.transfer_ds.select(window_indices)

            yield from transfer_generator(
                transfer_ds_window,
                self.max_seq_len,
            )
        else:
            retain_ds_window = self.retain_ds.select(window_indices)

            yield from retain_generator(
                retain_ds_window,
                self.max_seq_len,
            )


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
