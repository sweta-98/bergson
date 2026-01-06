import os

import torch
from torch.utils.data import IterableDataset as TorchIterableDataset

def is_debug():
    return int(os.environ.get("RANK", "0")) == 0


class AlternatingDataset(TorchIterableDataset):
    def __init__(
        self,
        first_ds,
        second_ds,
        rank,
        world_size,
        examples_per_phase,
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.examples_per_phase = examples_per_phase
        self.first_ds = first_ds.shard(num_shards=self.world_size, index=self.rank)
        self.second_ds = second_ds.shard(num_shards=self.world_size, index=self.rank)

        self.examples_per_rank = self.examples_per_phase // self.world_size
        remainder = self.examples_per_phase % self.world_size
        if self.rank < remainder:
            self.examples_per_rank += 1
            
        # We handle the "Phase" tracking internally in the iterator
        self._epoch_counter = 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 0:
            raise RuntimeError("AlternatingDataset does not support num_workers > 0.")
        
        # Calculate window based on internal counter
        rank_start_idx = (self._epoch_counter * self.examples_per_rank) % len(self.first_ds)
        
        window_indices = [
            (rank_start_idx + i) % len(self.first_ds)
            for i in range(self.examples_per_rank)
        ]

        if is_debug():
            print("starting first window indices", flush=True)
        yield from self.first_ds.select(window_indices)
        
        if is_debug():
            print("starting second window indices", flush=True)
        yield from self.second_ds.select(window_indices)
        
        # Increment counter so next time __iter__ is called (next epoch), we move the window
        self._epoch_counter += 1
        print("epoch counter now", self._epoch_counter, flush=True)