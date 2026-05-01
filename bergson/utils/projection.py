from __future__ import annotations

from typing import Literal

import torch

from trak.projectors import BasicProjector, CudaProjector, ProjectionType

# Fixed seed: every gradient build run produces the same projection so that
# train and eval indices are comparable to each other and to indices built in
# separate processes.
PROJECTION_SEED = 0


def supports_fast_jl(device: torch.device) -> bool:
    """Probe whether ``fast_jl`` works for this device. CudaProjector requires
    it; BasicProjector is the CPU/fallback path."""
    if device.type != "cuda":
        return False
    try:
        import fast_jl

        num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count
        # Smallest call shape that exercises the kernel without allocating much.
        fast_jl.project_rademacher_8(
            torch.zeros(8, 1_000, device=device), 512, 0, num_sms
        )
        return True
    except Exception:
        return False


def make_global_projector(
    grad_dim: int,
    proj_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    projection_type: Literal["normal", "rademacher"] = "rademacher",
):
    """Build a TRAK projector for a flattened per-example gradient of length
    ``grad_dim`` to ``proj_dim``. Picks ``CudaProjector`` when ``fast_jl``
    works on this device, else falls back to ``BasicProjector``."""
    proj_type = ProjectionType(projection_type)
    if supports_fast_jl(device):
        return CudaProjector(
            grad_dim=grad_dim,
            proj_dim=proj_dim,
            seed=PROJECTION_SEED,
            proj_type=proj_type,
            device=device,
            dtype=dtype,
            # Block / batch sizes match the values used in the LESS reference
            # implementation; tune later if needed.
            block_size=128,
            max_batch_size=32,
        )
    return BasicProjector(
        grad_dim=grad_dim,
        proj_dim=proj_dim,
        seed=PROJECTION_SEED,
        proj_type=proj_type,
        device=device,
        dtype=dtype,
        block_size=128,
    )
