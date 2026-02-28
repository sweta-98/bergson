import json
import warnings
from pathlib import Path
from typing import Literal

import torch

from bergson.config import PreprocessConfig
from bergson.gradients import GradientProcessor
from bergson.utils.math import damped_psd_power


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
        if norm > 0:
            grads = {k: v / norm for k, v in grads.items()}
        else:
            warnings.warn("Gradient norm is zero, skipping normalization")

    return grads


def normalize_flat_grad(
    grad: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Unit-normalize a single gradient tensor."""
    grad = grad.to(device=device, dtype=torch.float32)
    norm = grad.norm()
    if norm > 0:
        grad /= norm
    else:
        warnings.warn("Gradient norm is zero, skipping normalization")
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


def mix_preconditioners(
    query_path: str | Path,
    index_path: str | Path,
    output_path: str | Path,
    mixing_coefficient: float = 0.99,
) -> Path:
    """Mix query and index preconditioners and save the result to disk.

    Computes ``H_mixed = coeff * H_query + (1 - coeff) * H_index`` for
    every module's raw H matrix, then persists a new
    :class:`~bergson.gradients.GradientProcessor` at *output_path*.

    A ``mix_config.json`` file is also written alongside for provenance.

    Parameters
    ----------
    query_path : str | Path
        Directory containing the query GradientProcessor.
    index_path : str | Path
        Directory containing the index GradientProcessor.
    output_path : str | Path
        Directory where the mixed GradientProcessor will be saved.
    mixing_coefficient : float
        Weight for the query preconditioner (1.0 = query only).

    Returns
    -------
    Path
        The *output_path* as a :class:`pathlib.Path`.
    """
    query_path = Path(query_path)
    index_path = Path(index_path)
    output_path = Path(output_path)

    q_proc = GradientProcessor.load(query_path)
    i_proc = GradientProcessor.load(index_path)

    mixed_preconditioners = {
        k: q_proc.preconditioners[k] * mixing_coefficient
        + i_proc.preconditioners[k] * (1 - mixing_coefficient)
        for k in q_proc.preconditioners
    }

    # Build a new processor with the mixed preconditioners
    mixed_proc = GradientProcessor(
        normalizers=q_proc.normalizers,
        preconditioners=mixed_preconditioners,
        preconditioners_eigen={},
        projection_dim=q_proc.projection_dim,
        reshape_to_square=q_proc.reshape_to_square,
        projection_type=q_proc.projection_type,
        include_bias=q_proc.include_bias,
    )
    mixed_proc.save(output_path)

    # Save provenance metadata
    mix_config = {
        "query_path": str(query_path),
        "index_path": str(index_path),
        "mixing_coefficient": mixing_coefficient,
    }
    with (output_path / "mix_config.json").open("w") as f:
        json.dump(mix_config, f, indent=2)

    return output_path


def get_trackstar_preconditioner(
    preconditioner_path: str | None,
    device: torch.device,
    power: float = -0.5,
) -> dict[str, torch.Tensor]:
    """Compute preconditioner matrices from a saved processor file.

    Parameters
    ----------
    preconditioner_path : str | None
        Directory containing the saved GradientProcessor.
    device : torch.device
        Device to load the preconditioner onto.
    power : float
        Matrix power to apply to each H matrix.

        * ``-0.5`` — H^(-1/2), used for split (two-sided) preconditioning
          where both query and index gradients are multiplied by H^(-1/2).
        * ``-1``   — H^(-1), used for one-sided preconditioning where only
          the query gradients are preconditioned.
    """
    if preconditioner_path is None:
        return {}

    preconditioners = GradientProcessor.load(
        Path(preconditioner_path),
        map_location=device,
    ).preconditioners

    return {
        name: damped_psd_power(H.to(device=device), power=power)
        for name, H in preconditioners.items()
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
    """Precondition query gradients with the preconditioner."""
    h_inv = get_trackstar_preconditioner(
        preprocess_cfg.preconditioner_path,
        device=device,
        power=-0.5 if preprocess_cfg.unit_normalize else -1,
    )

    if h_inv:
        return {
            name: (grads[name].to(device) @ h_inv[name]).cpu()
            for name in target_modules
        }

    return grads
