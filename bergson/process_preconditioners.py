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


def mixed_eigen_decomp(
    query_preconditioner_path: str | None,
    index_preconditioner_path: str | None,
    mixing_coefficient: float,
    save_path: str,
    target_modules: list[str],
    device: torch.device,
    local_rank: int,
    prec_metadata: dict[str, tuple[torch.Size, torch.dtype]],
    offload_to_cpu: bool = False,
):
    """Mix query and index preconditioners."""
    print("Mixed eigen decomp started")
    # Get distributed info if available
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    use_q = query_preconditioner_path is not None
    use_i = index_preconditioner_path is not None

    assert use_q or use_i, "At least one preconditioner path must be provided"

    # Assign preconditioners to this rank
    rank_prec_names = [
        name 
        for idx, name in enumerate(target_modules) 
        if idx % world_size == rank
    ]

    if rank == 0:
        print(
            f"Distributing {len(target_modules)} "
            f"preconditioners across {world_size} GPUs..."
        )

    print(f"Rank {rank} will process {len(rank_prec_names)} preconditioners")

    # Load preconditioners: rank 0 loads and distributes to other ranks
    q, i = {}, {}
    
    # Use rank 0 to distribute preconditioners to other ranks
    # to minimize CPU RAM usage.
    if dist.is_initialized():
        # Create a CPU group for communication
        cpu_group = dist.new_group(backend="gloo")
        
        if rank == 0:
            print("Rank 0 loading preconditioners...")
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
            
            print(f"Rank 0 distributing preconditioners to {world_size} ranks...")
            
            # Send preconditioners to each rank
            for target_rank in tqdm(range(1, world_size), desc="Distributing preconditioners"):
                target_rank_prec_names = [
                    name 
                    for idx, name in enumerate(target_modules) 
                    if idx % world_size == target_rank
                ]
                
                # Send query preconditioners
                if use_q:
                    for name in target_rank_prec_names:
                        print(f"send {name}")
                        if name in q_full:
                            dist.send(q_full[name], dst=target_rank, group=cpu_group)
                print("send q")
                
                # Send index preconditioners
                if use_i:
                    for name in target_rank_prec_names:
                        print(f"send {name}")
                        if name in i_full:
                            dist.send(i_full[name], dst=target_rank, group=cpu_group)
                print("send i")
            
            # Rank 0 keeps its own preconditioners
            if use_q:
                q = {name: q_full[name] for name in rank_prec_names}
            if use_i:
                i = {name: i_full[name] for name in rank_prec_names}
        else:
            # Other ranks receive their assigned preconditioners
            print(f"Rank {rank} receiving preconditioners from rank 0...")

            # Receive actual tensors using provided metadata
            if use_q:
                for name in rank_prec_names:
                    prec_shape, prec_dtype = prec_metadata[name]
                    recv_tensor = torch.zeros(prec_shape, dtype=prec_dtype, device="cpu")
                    dist.recv(recv_tensor, src=0, group=cpu_group)
                    q[name] = recv_tensor
            
            if use_i:
                for name in rank_prec_names:
                    prec_shape, prec_dtype = prec_metadata[name]
                    recv_tensor = torch.zeros(prec_shape, dtype=prec_dtype, device="cpu")
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
    # If offload_to_cpu is False, they're already on GPU, so mixing happens on GPU
    # If offload_to_cpu is True, they're on CPU, mix on CPU (or move to GPU temporarily)
    if not offload_to_cpu:
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
    else:
        mixed_preconditioners = (
            {k: q[k] * mixing_coefficient + i[k] * (1 - mixing_coefficient) for k in q}
            if (q and i)
            else (q or i)
        )

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

        if offload_to_cpu:
            H = H.cpu()

    # Gather results from all ranks to rank 0
    if dist.is_initialized():
        if rank == 0:
            print("Gathering preconditioner eigen decompositions from all ranks...")

        # Create a CPU group for gathering (similar to process_preconditioners)
        cpu_group = dist.new_group(backend="gloo")

        # Move eigen decompositions to CPU for communication
        for name in eigen_decompositions.keys():
            eigen_decompositions[name] = (
                eigen_decompositions[name][0].cpu(),
                eigen_decompositions[name][1].cpu(),
            )

        # Gather all eigen decomposition results to rank 0
        # using point-to-point communication
        all_eigen_decompositions = {}
        # Build metadata cache once if rank 0 needs to receive
        metadata_cache = {}
        if rank == 0:
            # Load metadata once for all preconditioners
            temp_path = query_preconditioner_path or index_preconditioner_path
            assert temp_path is not None
            temp_processor = GradientProcessor.load(
                Path(temp_path),
                map_location="cpu",
            )
            metadata_cache = {
                name: (
                    temp_processor.preconditioners[name].shape,
                    temp_processor.preconditioners[name].dtype,
                )
                for name in target_modules
            }
            del temp_processor

        # First phase: all ranks send their computed eigen decompositions to rank 0
        for name in target_modules:
            prec_rank = target_modules.index(name) % world_size
            if rank == prec_rank and rank != 0:
                # This rank computed it - send to rank 0
                local_eigen_decomp = eigen_decompositions.get(name)
                if local_eigen_decomp is not None:
                    # Send both tensors to rank 0
                    dist.send(local_eigen_decomp[0], dst=0, group=cpu_group)
                    dist.send(local_eigen_decomp[1], dst=0, group=cpu_group)
                    # Free memory after sending
                    del eigen_decompositions[name]
            elif rank == prec_rank and rank == 0:
                # Rank 0 computed it - keep it
                local_eigen_decomp = eigen_decompositions.get(name)
                if local_eigen_decomp is not None:
                    loc = device if not offload_to_cpu else "cpu"
                    all_eigen_decompositions[name] = (
                        local_eigen_decomp[0].to(device=loc),
                        local_eigen_decomp[1].to(device=loc),
                    )

        # Second phase: rank 0 receives all eigen decompositions from other ranks
        if rank == 0:
            for name in target_modules:
                prec_rank = target_modules.index(name) % world_size
                if prec_rank != 0:
                    # Rank 0 receives from the rank that computed it
                    prec_shape, prec_dtype = metadata_cache[name]
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
                    all_eigen_decompositions[name] = (
                        recv_eigval.to(device=device if not offload_to_cpu else "cpu"),
                        recv_eigvec.to(device=device if not offload_to_cpu else "cpu"),
                    )

        eigen_decompositions = all_eigen_decompositions if rank == 0 else {}

        # Clean up remaining eigen_decompositions on non-zero ranks
        if rank != 0:
            del eigen_decompositions
            del mixed_preconditioners
    else:
        # Single GPU case - eigen_decompositions already has all results
        all_eigen_decompositions = eigen_decompositions

    # Rank 0 needs to load all preconditioners for saving
    mixed_processor = None
    if rank == 0:
        if dist.is_initialized():
            # Load all preconditioners for saving
            q = {}
            i = {}
            if use_q:
                assert query_preconditioner_path is not None
                q = GradientProcessor.load(
                    Path(query_preconditioner_path),
                    map_location="cpu" if offload_to_cpu else device,
                ).preconditioners
            if use_i:
                assert index_preconditioner_path is not None
                i = GradientProcessor.load(
                    Path(index_preconditioner_path),
                    map_location="cpu" if offload_to_cpu else device,
                ).preconditioners

            # Mix all preconditioners for saving
            mixed_preconditioners_for_save = (
                {
                    k: q[k] * mixing_coefficient
                    + i[k] * (1 - mixing_coefficient)
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
            # mixed_preconditioners and eigen_decompositions are defined above (lines 247/259 and 271)
            mixed_preconditioners_for_save = mixed_preconditioners  # type: ignore
            final_eigen_decompositions = eigen_decompositions  # type: ignore

        mixed_processor = GradientProcessor()
        mixed_processor.preconditioners = mixed_preconditioners_for_save
        mixed_processor.preconditioners_eigen = final_eigen_decompositions
        mixed_processor.save(Path(save_path))

    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        return mixed_processor
    else:
        return GradientProcessor.load(Path(save_path), map_location="cpu")
