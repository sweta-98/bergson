from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from bergson.data import (
    create_eigen_index,
    create_preconditioner_index,
    get_eigen_offset,
    get_preconditioner_offset,
)
from bergson.gradients import GradientProcessor


def mix_and_save_processors(
    query_preconditioner_path: str | None,
    index_preconditioner_path: str | None,
    mixing_coefficient: float,
    save_path: Path,
    target_modules: list[str],
    device: torch.device,
    grad_sizes: dict[str, int],
):
    """Mix query and index preconditioners."""
    # Get distributed info if available
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    use_q = query_preconditioner_path is not None
    use_i = index_preconditioner_path is not None

    assert use_q or use_i, "At least one preconditioner path must be provided"

    # Assign preconditioners to each rank
    rank_modules = [
        name for idx, name in enumerate(target_modules) if idx % world_size == rank
    ]

    if rank == 0:
        print(
            f"Distributing {len(target_modules)} "
            f"preconditioners across {world_size} GPUs..."
        )

    q, i = {}, {}

    if rank == 0:
        print("Loading preconditioners on each rank...")

    if use_q:
        assert query_preconditioner_path is not None
        query_processor = GradientProcessor.load(
            Path(query_preconditioner_path),
            map_location="cpu",
            module_names=rank_modules,
        )
        q = query_processor.preconditioners

    if use_i:
        assert index_preconditioner_path is not None
        index_processor = GradientProcessor.load(
            Path(index_preconditioner_path),
            map_location="cpu",
            module_names=rank_modules,
        )
        i = index_processor.preconditioners

    if rank == 0:
        print("Mixing preconditioners...")

    mixed_preconditioners = (
        {
            k: (
                q[k].to(device=device) * mixing_coefficient
                + i[k].to(device=device) * (1 - mixing_coefficient)
            ).cpu()
            for k in q
        }
        if (q and i)
        else (q or i)
    )

    if rank == 0:
        print("Writing mixed preconditioners to disk...")

    # Determine dtype from first preconditioner
    first_prec = next(iter(mixed_preconditioners.values()))
    np_dtype = np.float32 if first_prec.dtype == torch.float32 else np.float16

    if mixed_preconditioners:
        prec_mmap = create_preconditioner_index(save_path, grad_sizes, np_dtype)

        for name, prec in mixed_preconditioners.items():
            # Always use offsets for unstructured format
            offset = get_preconditioner_offset(grad_sizes, name)
            size = grad_sizes[name]
            prec_mmap[offset:offset + size * size] = prec.cpu().numpy().astype(np_dtype).flatten()

        prec_mmap.flush()

    if rank == 0:
        print("Computing preconditioner eigen decompositions...")

    # Create or load eigen memmap with all target_modules
    eigen_mmap = create_eigen_index(save_path, grad_sizes, np_dtype)

    for name in tqdm(
        rank_modules,
        desc=f"Rank {rank}: Computing preconditioner inversions",
        disable=rank != 0,  # Only show progress bar on rank 0
    ):
        H = mixed_preconditioners[name].to(device=device, dtype=torch.float64)
        damping_val = 0.1 * H.abs().mean()
        H = H + damping_val * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

        eigval, eigvec = torch.linalg.eigh(H)
        # Convert back to original dtype before storing
        original_dtype = mixed_preconditioners[name].dtype
        eigval_cpu = eigval.to(dtype=original_dtype).contiguous().cpu()
        eigvec_cpu = eigvec.to(dtype=original_dtype).contiguous().cpu()

        # Always use offsets for unstructured format
        eigval_offset, eigvec_offset = get_eigen_offset(grad_sizes, name)
        size = grad_sizes[name]
        eigen_mmap[eigval_offset:eigval_offset + size] = eigval_cpu.numpy().astype(np_dtype)
        eigen_mmap[eigvec_offset:eigvec_offset + size * size] = eigvec_cpu.numpy().astype(np_dtype).flatten()

    # Barrier to ensure all ranks have written
    if dist.is_initialized():
        dist.barrier()
