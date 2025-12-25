from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

from bergson.gradients import GradientProcessor


def process_preconditioners(
    processor: GradientProcessor,
    preconditioners: dict[str, torch.Tensor],
    len_data: int,
    grad_sizes: dict[str, int],
    rank: int,
):
    """
    Aggregate preconditioners across ranks and compute their eigen decomposition
    distributed across all ranks.
    """
    preconditioners_eigen = {}

    device = next(iter(preconditioners.values())).device
    dtype = next(iter(preconditioners.values())).dtype

    if rank == 0:
        print("Saving preconditioners...")

    for name, prec in preconditioners.items():
        preconditioners[name] = (prec / len_data).cpu()

    if rank == 0:
        print("Computing preconditioner eigen decompositions...")

    for name in preconditioners.keys():
        prec = preconditioners[name].to(dtype=torch.float64, device=device)
        eigvals, eigvecs = torch.linalg.eigh(prec)
        preconditioners_eigen[name] = (
            eigvals.to(dtype=dtype).contiguous().cpu(),
            eigvecs.to(dtype=dtype).contiguous().cpu(),
        )

    if not dist.is_initialized():
        processor.preconditioners = preconditioners
        processor.preconditioners_eigen = preconditioners_eigen
        return

    if rank == 0:
        print("Gathering preconditioners...")

    cpu_group = dist.new_group(backend="gloo")

    for name, grad_size in grad_sizes.items():
        if name in preconditioners:
            local_prec = preconditioners[name]
            del preconditioners[name]
        else:
            local_prec = torch.zeros([grad_size, grad_size], dtype=dtype, device="cpu")

        dist.reduce(local_prec, dst=0, op=dist.ReduceOp.SUM, group=cpu_group)

        if rank == 0:
            preconditioners[name] = local_prec

    if rank == 0:
        processor.preconditioners = preconditioners

        print("Gathering eigen decompositions...")

    for name, grad_size in grad_sizes.items():
        prec_size = torch.Size([grad_size, grad_size])
        if name not in preconditioners_eigen:
            eigval = torch.zeros(prec_size[0], dtype=dtype)
            eigvec = torch.zeros(prec_size, dtype=dtype)
        else:
            eigval, eigvec = preconditioners_eigen[name]

        dist.reduce(eigval, dst=0, op=dist.ReduceOp.SUM, group=cpu_group)
        dist.reduce(eigvec, dst=0, op=dist.ReduceOp.SUM, group=cpu_group)

        if rank == 0:
            preconditioners_eigen[name] = (eigval, eigvec)

    if rank == 0:
        processor.preconditioners_eigen = preconditioners_eigen

    print("Done!")


def mix_and_save_processors(
    query_preconditioner_path: str | None,
    index_preconditioner_path: str | None,
    mixing_coefficient: float,
    save_path: str,
    target_modules: list[str],
    device: torch.device,
    prec_metadata: dict[str, tuple[torch.Size, torch.dtype]],
):
    """Mix query and index preconditioners."""
    # Get distributed info if available
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    use_q = query_preconditioner_path is not None
    use_i = index_preconditioner_path is not None

    assert use_q or use_i, "At least one preconditioner path must be provided"

    # Assign preconditioners to this rank
    rank_prec_names = [
        name for idx, name in enumerate(target_modules) if idx % world_size == rank
    ]

    if rank == 0:
        print(
            f"Distributing {len(target_modules)} "
            f"preconditioners across {world_size} GPUs..."
        )

    # Load preconditioners: rank 0 loads and distributes to other ranks
    q, i = {}, {}

    # Use rank 0 to distribute preconditioners to other ranks
    # to minimize CPU RAM usage.
    if dist.is_initialized():
        # Create a CPU group for communication
        cpu_group = dist.new_group(backend="gloo")

        if rank == 0:
            # Rank 0 loads full preconditioner files
            q_full, i_full = {}, {}
            if use_q:
                q_full = GradientProcessor.load(
                    Path(query_preconditioner_path),
                    map_location="cpu",
                ).preconditioners

            if use_i:
                i_full = GradientProcessor.load(
                    Path(index_preconditioner_path),
                    map_location="cpu",
                ).preconditioners

            # Send preconditioners to each rank
            for target_rank in tqdm(
                range(1, world_size), desc="Distributing preconditioners"
            ):
                target_rank_prec_names = [
                    name
                    for idx, name in enumerate(target_modules)
                    if idx % world_size == target_rank
                ]

                # Send query preconditioners
                if use_q:
                    for name in target_rank_prec_names:
                        if name in q_full:
                            dist.send(q_full[name], dst=target_rank, group=cpu_group)

                # Send index preconditioners
                if use_i:
                    for name in target_rank_prec_names:
                        if name in i_full:
                            dist.send(i_full[name], dst=target_rank, group=cpu_group)

            # Rank 0 keeps its own preconditioners
            if use_q:
                q = {name: q_full[name] for name in rank_prec_names}
            if use_i:
                i = {name: i_full[name] for name in rank_prec_names}
        else:
            # Receive tensors
            if use_q:
                for name in rank_prec_names:
                    prec_shape, prec_dtype = prec_metadata[name]
                    recv_tensor = torch.zeros(
                        prec_shape, dtype=prec_dtype, device="cpu"
                    )
                    dist.recv(recv_tensor, src=0, group=cpu_group)
                    q[name] = recv_tensor

            if use_i:
                for name in rank_prec_names:
                    prec_shape, prec_dtype = prec_metadata[name]
                    recv_tensor = torch.zeros(
                        prec_shape, dtype=prec_dtype, device="cpu"
                    )
                    dist.recv(recv_tensor, src=0, group=cpu_group)
                    i[name] = recv_tensor

        # Synchronize after distribution
        dist.barrier(group=cpu_group)

    else:
        # Single rank case - just load directly
        if use_q:
            q = GradientProcessor.load(
                Path(query_preconditioner_path),
                map_location="cpu",
            ).preconditioners
            q = {name: q[name] for name in rank_prec_names}

        if use_i:
            i = GradientProcessor.load(
                Path(index_preconditioner_path),
                map_location="cpu",
            ).preconditioners
            i = {name: i[name] for name in rank_prec_names}

    if rank == 0:
        print("Mixing preconditioners...")

    # Mix only assigned preconditioners
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
    mixed_preconditioners = {
        key: val
        for key, val in mixed_preconditioners.items()
        if key in rank_prec_names
    }

    if rank == 0:
        print(
            f"Computing preconditioner eigen decompositions across {world_size} GPUs..."
        )

    # Each rank processes its assigned preconditioners
    eigen_decompositions = {}
    for name in tqdm(
        rank_prec_names,
        desc=f"Rank {rank}: Computing preconditioner inversions",
        disable=rank != 0,  # Only show progress bar on rank 0
    ):
        H = mixed_preconditioners[name].to(device=device, dtype=torch.float64)
        damping_val = 0.1 * H.abs().mean()
        H = H + damping_val * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

        eigval, eigvec = torch.linalg.eigh(H)
        # Convert back to original dtype before storing
        original_dtype = mixed_preconditioners[name].dtype
        eigen_decompositions[name] = (
            eigval.to(dtype=original_dtype).contiguous(),
            eigvec.to(dtype=original_dtype).contiguous(),
        )
        del mixed_preconditioners[name]

    # Gather results from all ranks to rank 0
    if dist.is_initialized():
        # Probably unnecessary
        dist.barrier()

        if rank == 0:
            print("Gathering preconditioner eigen decompositions from all ranks...")

        # Move eigen decompositions to CPU for communication
        for name in eigen_decompositions.keys():
            eigen_decompositions[name] = (
                eigen_decompositions[name][0].cpu(),
                eigen_decompositions[name][1].cpu(),
            )

        # Gather all eigen decomposition results to rank 0
        # using point-to-point communication
        all_eigen_decompositions = {}
        for name in target_modules:
            prec_rank = target_modules.index(name) % world_size
            if rank == prec_rank and rank != 0:
                # Send to rank 0 then free memory
                local_eigen_decomp = eigen_decompositions[name]
                dist.send(local_eigen_decomp[0], dst=0, group=cpu_group)
                dist.send(local_eigen_decomp[1], dst=0, group=cpu_group)
                print(f"Rank {rank} sent {name}")
                del eigen_decompositions[name]
            elif rank == 0:
                # Rank 0 receives immediately after the send
                if prec_rank == 0:
                    print("rank 0 keeping rank 0 prec")
                    recv_eigval = eigen_decompositions[name][0]
                    recv_eigvec = eigen_decompositions[name][1]
                else:
                    # Rank 0 receives from the rank that computed it
                    prec_shape, prec_dtype = prec_metadata[name]
                    # Receive eigenvalues (1D tensor) -
                    # use the same dtype as the preconditioner
                    recv_eigval = torch.zeros(
                        prec_shape[0], dtype=prec_dtype, device="cpu"
                    )
                    # Receive eigenvectors (2D tensor) -
                    # use the same dtype as the preconditioner
                    recv_eigvec = torch.zeros(
                        prec_shape, dtype=prec_dtype, device="cpu"
                    )
                    dist.recv(recv_eigval, src=prec_rank, group=cpu_group)
                    dist.recv(recv_eigvec, src=prec_rank, group=cpu_group)

                all_eigen_decompositions[name] = (recv_eigval, recv_eigvec)

        dist.barrier()
        print(f"Gathering phase finished on rank {rank}")

        eigen_decompositions = all_eigen_decompositions if rank == 0 else {}

    else:
        # Single GPU case - eigen_decompositions already has all results
        all_eigen_decompositions = eigen_decompositions

    if dist.is_initialized():
        dist.barrier()

    # Load all preconditioners for saving on rank 0
    if rank == 0:
        if dist.is_initialized():
            print("Rank 0 loading preconditioners for saving...")
            # Load all preconditioners for saving
            q = {}
            i = {}
            if use_q:
                assert query_preconditioner_path is not None
                q = GradientProcessor.load(
                    Path(query_preconditioner_path),
                    map_location="cpu",
                ).preconditioners
            if use_i:
                assert index_preconditioner_path is not None
                i = GradientProcessor.load(
                    Path(index_preconditioner_path),
                    map_location="cpu",
                ).preconditioners
            print("Loaded preconditioners for saving")

            # Mix all preconditioners for saving
            mixed_preconditioners_for_save = (
                {
                    k: q[k] * mixing_coefficient + i[k] * (1 - mixing_coefficient)
                    for k in q
                }
                if (q and i)
                else (q or i)
            )
            del q, i

            # Determine final eigen decompositions to use
            final_eigen_decompositions = all_eigen_decompositions
        else:
            # Single GPU case - use already computed values
            mixed_preconditioners_for_save = mixed_preconditioners  # type: ignore
            final_eigen_decompositions = eigen_decompositions  # type: ignore

        mixed_processor = GradientProcessor(
            preconditioners=mixed_preconditioners_for_save,
            preconditioners_eigen=final_eigen_decompositions,
        )
        mixed_processor.save(Path(save_path))

    if dist.is_initialized():
        dist.barrier()