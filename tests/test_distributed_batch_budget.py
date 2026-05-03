"""Distributed bin-packing + pad must not exceed the per-rank token budget.

Regression test for a NCCL-collective-shaped deadlock observed in the LESS
pipeline FSDP train builds: ``pad_and_tensor`` does an ``all_reduce(MAX)``
on sequence length and pads every rank's batch to the global max.
Combined with bergson's bin-packing (which only enforces the budget per
batch *in isolation*), a rank that gets a single long-example batch can
force every other rank at the same iteration index to pad many short
examples up to that long length, blowing the per-batch token budget by
~40× and OOM-thrashing the GPU.

This test runs on CPU via gloo so it doesn't need a multi-GPU node.
"""

import socket

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from bergson.data import allocate_batches, pad_and_tensor


# Bimodal length distribution: 80 very short examples plus one long one.
# At token_batch_size=2048 the bin-packer produces one (80 × short) batch
# and one (1 × long) batch; round-robin distribution puts them on
# different ranks at iteration 0. The all_reduce(MAX) in pad_and_tensor
# then pads the (80 × short) batch to the long length on its rank.
SHORT_LEN = 25
LONG_LEN = 1000
N_SHORT = 80
TOKEN_BUDGET = 2048


def _doc_lengths() -> list[int]:
    return [SHORT_LEN] * N_SHORT + [LONG_LEN]


def _worker(rank: int, world_size: int, port: int, results) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )

        lengths = _doc_lengths()
        my_batches = allocate_batches(lengths, TOKEN_BUDGET, seed=42)

        # Pad every batch this rank owns and record the worst budget breach.
        worst = 0
        for batch in my_batches:
            input_ids = [[1] * lengths[i] for i in batch]
            # Match the build-path call: local-only padding so the
            # bin-packer's per-rank budget invariant is preserved. The
            # MAGIC path (default sync_max_len=True) still globally
            # syncs and is expected to over-pad in this scenario.
            padded, _, _ = pad_and_tensor(input_ids, sync_max_len=False)
            cost = padded.shape[0] * padded.shape[1]
            worst = max(worst, cost)
        results[rank] = worst
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_pad_respects_per_rank_budget_under_bimodal_lengths():
    """Each rank's post-pad batch should stay within the bin-packer's budget."""
    world_size = 2

    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    manager = mp.Manager()
    results = manager.dict()

    mp.spawn(
        _worker,
        args=(world_size, port, results),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        cost = results.get(rank)
        assert cost is not None, f"rank {rank} did not report"
        # Allow a small slack but blowups (e.g. 80 × 1000 = 80,000) must fail.
        assert cost <= TOKEN_BUDGET * 2, (
            f"rank {rank} padded to {cost} tokens, "
            f"exceeds bin-packer budget {TOKEN_BUDGET}. "
            "pad_and_tensor's global-max all_reduce broke the per-rank invariant."
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
