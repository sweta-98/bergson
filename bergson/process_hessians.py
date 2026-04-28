import torch
import torch.distributed as dist

from bergson.gradients import GradientProcessor


def process_hessians(
    processor: GradientProcessor,
    hessians: dict[str, torch.Tensor],
    len_data: int,
    grad_sizes: dict[str, int],
    rank: int,
):
    """
    Aggregate hessians across ranks and compute their eigen decomposition
    distributed across all ranks.
    """
    hessians_eigen = {}

    device = next(iter(hessians.values())).device
    dtype = next(iter(hessians.values())).dtype

    if rank == 0:
        print("Saving hessians...")

    for name, prec in hessians.items():
        hessians[name] = (prec / len_data).cpu()

    if rank == 0:
        print("Computing hessian eigen decompositions...")

    for name in hessians.keys():
        prec = hessians[name].to(dtype=torch.float64, device=device)
        eigvals, eigvecs = torch.linalg.eigh(prec)
        hessians_eigen[name] = (
            eigvals.to(dtype=dtype).contiguous().cpu(),
            eigvecs.to(dtype=dtype).contiguous().cpu(),
        )

    if not dist.is_initialized():
        processor.hessians = hessians
        processor.hessians_eigen = hessians_eigen
        return

    if rank == 0:
        print("Gathering hessians...")

    cpu_group: dist.ProcessGroup = dist.new_group(backend="gloo")  # type: ignore[assignment]

    for name, grad_size in grad_sizes.items():
        if name in hessians:
            local_prec = hessians[name]
            del hessians[name]
        else:
            local_prec = torch.zeros([grad_size, grad_size], dtype=dtype, device="cpu")

        dist.reduce(local_prec, dst=0, op=dist.ReduceOp.SUM, group=cpu_group)

        if rank == 0:
            hessians[name] = local_prec

    if rank == 0:
        processor.hessians = hessians

        print("Gathering eigen decompositions...")

    for name, grad_size in grad_sizes.items():
        prec_size = torch.Size([grad_size, grad_size])
        if name not in hessians_eigen:
            eigval = torch.zeros(prec_size[0], dtype=dtype)
            eigvec = torch.zeros(prec_size, dtype=dtype)
        else:
            eigval, eigvec = hessians_eigen[name]

        dist.reduce(eigval, dst=0, op=dist.ReduceOp.SUM, group=cpu_group)
        dist.reduce(eigvec, dst=0, op=dist.ReduceOp.SUM, group=cpu_group)

        if rank == 0:
            hessians_eigen[name] = (eigval, eigvec)

    if rank == 0:
        processor.hessians_eigen = hessians_eigen
