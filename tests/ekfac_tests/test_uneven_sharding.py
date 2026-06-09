"""Numerical tests for ShardedMul with unevenly sharded dimensions.

With include_bias=True the activation dimension becomes I+1, which is
generally not divisible by the world size. Rank 0 takes the remainder rows
(see shard_bounds). These tests check every sharded op against its dense
single-process reference under a 2-process gloo group on CPU.
"""

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from bergson.hessians.sharded_computation import ShardedMul, shard_bounds

WORLD_SIZE = 2
DIM = 5  # odd on purpose: shards are 3 and 2 rows
N, S, O = 2, 3, 4


def _shard_worker(rank, world_size, port, result_dict):
    """Run all sharded ops and store rank 0's results for comparison."""
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        sharder = ShardedMul()

        # Same seeded data on every rank
        g = torch.Generator().manual_seed(0)
        matrix = torch.randn(DIM, DIM, generator=g)
        vector = torch.randn(N, S, DIM, generator=g)
        grads = torch.randn(N, O, DIM, generator=g)
        lambda_full = torch.randn(O, DIM, generator=g).abs()

        start, end = shard_bounds(DIM, rank, world_size)
        matrix_shard = matrix[start:end].contiguous()

        results = {}
        results["matmul"] = sharder._matmul(vector, matrix_shard)
        results["transpose_matmul"] = sharder._transpose_matmul(grads, matrix_shard)

        o_start, o_end = shard_bounds(O, rank, world_size)
        lambda_shard = lambda_full[o_start:o_end].contiguous()

        hadamard = grads.clone()
        sharder._hadamard(hadamard, lambda_shard, lambda_damp_factor=0.1)
        results["hadamard"] = hadamard

        eigfn = grads.clone()
        sharder._apply_eigfn(eigfn, lambda_shard, fn=torch.rsqrt)
        results["apply_eigfn"] = eigfn

        if rank == 0:
            result_dict.update(results)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_sharded_ops_match_dense_with_uneven_shards():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    manager = mp.Manager()
    result_dict = manager.dict()
    mp.spawn(
        _shard_worker,
        args=(WORLD_SIZE, port, result_dict),
        nprocs=WORLD_SIZE,
        join=True,
    )

    # Dense references with the same seeded data
    g = torch.Generator().manual_seed(0)
    matrix = torch.randn(DIM, DIM, generator=g)
    vector = torch.randn(N, S, DIM, generator=g)
    grads = torch.randn(N, O, DIM, generator=g)
    lambda_full = torch.randn(O, DIM, generator=g).abs()

    torch.testing.assert_close(result_dict["matmul"], vector @ matrix)
    torch.testing.assert_close(result_dict["transpose_matmul"], grads @ matrix.T)

    inverse_lambda = (
        lambda_full + 0.1 * lambda_full.mean()
    ).reciprocal()  # _hadamard dense path
    torch.testing.assert_close(result_dict["hadamard"], grads * inverse_lambda)

    torch.testing.assert_close(
        result_dict["apply_eigfn"], grads * torch.rsqrt(lambda_full)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
