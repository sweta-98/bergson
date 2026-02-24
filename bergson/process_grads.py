from pathlib import Path
from typing import Literal

import torch

from bergson.config import PreprocessConfig
from bergson.gradients import GradientProcessor
from bergson.utils.math import compute_damped_inverse, psd_rsqrt


def normalize_grad(
    grad_dict: dict[str, torch.Tensor],
    unit_normalize: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Preprocess a single gradient. Optionally unit-normalizes
    across all columns, moves to device."""
    grads = {
        name: g.to(device=device, dtype=torch.float32) for name, g in grad_dict.items()
    }

    if unit_normalize:
        norm = torch.sqrt(torch.stack([g.pow(2).sum() for g in grads.values()]).sum())
        assert norm > 0, "Gradient norm is zero"
        grads = {k: v / norm for k, v in grads.items()}

    return grads


def normalize_flat_grad(
    grad: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Unit-normalize a flat gradient tensor (not a dict)."""
    grad = grad.to(device=device, dtype=torch.float32)
    norm = grad.norm()
    if norm > 0:
        grad = grad / norm
    return grad


def preprocess_grads(
    grad_dict: dict[str, torch.Tensor],
    grad_column_names: list[str],
    unit_normalize: bool,
    device: torch.device,
    aggregate_grads: Literal["mean", "sum", "none"] = "none",
    normalize_aggregated_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """Preprocess the gradients. Returns a dictionary of preprocessed gradients
    with shape [1, grad_dim]. Preprocessing includes some combination of unit
    normalization, accumulation, aggregated gradient normalization, and dtype
    conversion."""

    grad_column_names = list(grad_dict.keys())

    # Short-circuit if possible
    if aggregate_grads == "none" and not unit_normalize:
        return {name: grad_dict[name].to(device=device) for name in grad_column_names}

    num_rows = len(grad_dict[grad_column_names[0]])

    # Get sum and sum of squares of the gradients
    acc = {}
    ss_acc = torch.tensor(0.0, device=device, dtype=torch.float32)
    if not unit_normalize:
        ss_acc.fill_(1.0)

    for name in grad_column_names:
        x = grad_dict[name].to(device=device, dtype=torch.float32)
        acc[name] = x.sum(0)
        if unit_normalize:
            ss_acc += x.pow(2).sum()

    ss_acc = ss_acc.sqrt()
    assert ss_acc > 0, "Sum of squares of entire dataset is zero"

    # Process the gradients
    if aggregate_grads == "mean":
        grads = {
            name: (acc[name] / ss_acc / num_rows).unsqueeze(0)
            for name in grad_column_names
        }
    elif aggregate_grads == "sum":
        grads = {name: (acc[name] / ss_acc).unsqueeze(0) for name in grad_column_names}
    elif aggregate_grads == "none":
        grads = {name: grad_dict[name].to(device=device) for name in grad_column_names}
        if unit_normalize:
            norms = torch.cat(list(grads.values()), dim=1).norm(dim=1, keepdim=True)
            grads = {k: v / norms for k, v in grads.items()}
    else:
        raise ValueError(f"Invalid aggregate_grads: {aggregate_grads}")

    # Normalize the aggregated gradient
    if normalize_aggregated_grad:
        grad_norm = torch.cat(
            [grads[name].flatten() for name in grad_column_names], dim=0
        ).norm()
        for name in grad_column_names:
            grads[name] /= grad_norm

    return grads


def compute_preconditioner(
    query_preconditioner_path: str | None,
    index_preconditioner_path: str | None,
    mixing_coefficient: float,
    unit_normalize: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Compute preconditioner matrices from saved processor files.

    When unit_normalize=True, returns H^(-1/2) for split application to both
    query and index sides.
    When unit_normalize=False, returns H^(-1) for one-sided application.
    """
    use_q = query_preconditioner_path is not None
    use_i = index_preconditioner_path is not None

    if not (use_q or use_i):
        return {}

    q, i = {}, {}
    if use_q:
        q = GradientProcessor.load(
            Path(query_preconditioner_path),
            map_location=device,
        ).preconditioners
    if use_i:
        i = GradientProcessor.load(
            Path(index_preconditioner_path),
            map_location=device,
        ).preconditioners

    mixed = (
        {k: q[k] * mixing_coefficient + i[k] * (1 - mixing_coefficient) for k in q}
        if (q and i)
        else (q or i)
    )

    if unit_normalize:
        # H^(-1/2) for split application to both sides
        return {
            name: psd_rsqrt(H.to(device=device, dtype=torch.float32))
            for name, H in mixed.items()
        }
    else:
        # H^(-1) for one-sided application
        return {
            name: compute_damped_inverse(H.to(device=device))
            for name, H in mixed.items()
        }


def precondition_grad(
    grad: dict[str, torch.Tensor],
    h_inv: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Precondition a single example's gradients."""
    if not h_inv:
        return grad

    return {name: (grad[name].to(device) @ h_inv[name]).cpu() for name in grad.keys()}


def precondition_grads(
    grads: dict[str, torch.Tensor],
    preprocess_cfg: PreprocessConfig,
    target_modules: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Precondition query gradients with the query and/or index preconditioners."""
    h_inv = compute_preconditioner(
        preprocess_cfg.query_preconditioner_path,
        preprocess_cfg.index_preconditioner_path,
        preprocess_cfg.mixing_coefficient,
        preprocess_cfg.unit_normalize,
        device,
    )

    if h_inv:
        return {
            name: (grads[name].to(device) @ h_inv[name]).cpu()
            for name in target_modules
        }

    return grads
