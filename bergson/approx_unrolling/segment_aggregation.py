import os
import shutil
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file

from bergson.config import DistributedConfig
from bergson.distributed import launch_distributed_run
from bergson.hessians.eigenvectors import compute_eigendecomposition
from bergson.utils.logger import get_logger
from bergson.utils.utils import get_device, get_device_index

_SHARD_KINDS = ("activation_sharded", "gradient_sharded")


def aggregate_segment_covariances(
    run_path: str | Path,
    method: str,
    n_segments: int,
    per_segment: int,
    distributed: DistributedConfig,
    *,
    resume: bool = False,
) -> None:
    """Sum per-checkpoint covariances per segment, then eigendecompose.

    Output layout::

        <run_path>/segment_{i}/<method>/
            activation_sharded/shard_*.safetensors
            gradient_sharded/shard_*.safetensors
            eigen_activation_sharded/shard_*.safetensors
            eigen_gradient_sharded/shard_*.safetensors
            total_processed.pt
    """
    logger = get_logger("aggregate_segment_covariances")
    base_run = Path(run_path)

    # Resolve skip-or-clean in the main process before spawning workers, so
    # all workers see a consistent on-disk state when they start. Resume is
    # split into two sub-stages because the worker writes covariances first
    # and eigvecs second; a crash between them must not leave the segment
    # half-done. The worker re-checks ``total_processed.pt`` to decide
    # whether to skip the cov-sum step.
    segments_to_process: list[int] = []
    for seg in range(n_segments):
        seg_dir = base_run / f"segment_{seg}"
        out_dir = seg_dir / method
        # TODO: cov_done only checks the total_processed.pt sentinel, not
        # the actual shard subdirs.
        cov_done = (out_dir / "total_processed.pt").exists()
        eigen_done = all((out_dir / f"eigen_{kind}").exists() for kind in _SHARD_KINDS)

        if resume and cov_done and eigen_done:
            logger.info(f"[seg {seg}] skip — cov + eigvecs both exist")
            continue
        if not resume and out_dir.exists():
            shutil.rmtree(out_dir)

        for i in range(per_segment):
            d = seg_dir / f"ckpt_{i}" / method
            if not d.exists():
                raise FileNotFoundError(
                    f"Expected per-checkpoint covariance dir {d} not found. "
                    "Did step 1 finish for this checkpoint?"
                )
        segments_to_process.append(seg)

    if not segments_to_process:
        logger.info("All segments already aggregated; nothing to do.")
        return

    launch_distributed_run(
        "aggregate_segment_covariances",
        _aggregate_worker,
        [base_run, method, per_segment, segments_to_process],
        distributed,
    )


def _aggregate_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    base_run: Path,
    method: str,
    per_segment: int,
    segments_to_process: list[int],
) -> None:
    """Per-rank worker. Each rank only touches ``shard_{rank}.safetensors``."""
    if torch.cuda.is_available():
        torch.cuda.set_device(get_device_index(local_rank))

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(get_device(local_rank)),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )

    logger = get_logger("aggregate_segment_covariances")
    device = get_device(local_rank)
    shard_name = f"shard_{rank}.safetensors"

    for seg in segments_to_process:
        seg_dir = base_run / f"segment_{seg}"
        out_dir = seg_dir / method
        ckpt_method_dirs = [seg_dir / f"ckpt_{i}" / method for i in range(per_segment)]

        # Cov-sum sub-stage: skip if a previous (resumed) run already wrote
        # the segment-averaged covariances. ``total_processed.pt`` is the
        # last file written in the cov-sum stage, so its presence is a
        # safe sentinel.
        cov_done = (out_dir / "total_processed.pt").exists()
        if cov_done:
            logger.info(
                f"[seg {seg} rank {rank}] cov-sum already done, skipping to eigendecomp"
            )
        else:
            logger.info(
                f"[seg {seg} rank {rank}] summing {per_segment} checkpoints "
                f"into {out_dir} on {device}"
            )

            for kind in _SHARD_KINDS:
                (out_dir / kind).mkdir(parents=True, exist_ok=True)
                # Each rank reads only its own shard from each checkpoint.
                in_paths = [d / kind / shard_name for d in ckpt_method_dirs]
                out_path = out_dir / kind / shard_name
                _sum_my_shard(in_paths, out_path, device=device)

            # total_processed.pt is a tiny scalar; only one rank writes it.
            if rank == 0:
                total = None
                for d in ckpt_method_dirs:
                    t = torch.load(
                        d / "total_processed.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                    total = t if total is None else total + t
                torch.save(total, out_dir / "total_processed.pt")

            # All ranks must finish writing covariances + total_processed
            # before eigendecomposition reads them back.
            if world_size > 1:
                dist.barrier()

        # Eigendecompose the segment-averaged covariances in place.
        # compute_eigendecomposition is itself rank-aware and distributes
        # keys across ranks via fair_distribute_by_cost.
        total_processed = torch.load(
            out_dir / "total_processed.pt",
            map_location="cpu",
            weights_only=False,
        )
        for kind in _SHARD_KINDS:
            compute_eigendecomposition(
                str(out_dir / kind),
                total_processed=total_processed,
            )

    if world_size > 1:
        dist.barrier()


def _sum_my_shard(
    in_paths: list[Path],
    out_path: Path,
    device: str,
) -> None:
    """Sharded version of in place a.add_(b) for a list of tensors.
    Each rank only sums its own shard across the checkpoints and writes one
    output shard.
    """
    acc: dict[str, torch.Tensor] = {}
    for c, p in enumerate(in_paths):
        with safe_open(p, framework="pt", device=device) as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if c == 0:
                    acc[k] = t.clone()
                else:
                    acc[k].add_(t)
    # safetensors save_file writes from CPU.
    save_file({k: v.cpu() for k, v in acc.items()}, out_path)
