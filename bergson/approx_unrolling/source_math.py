import json
import re
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from bergson.config import ApproxUnrollingConfig


def compute_eta_K_per_segment(
    approx_unrolling_cfg: ApproxUnrollingConfig,
) -> list[float]:
    """Per-segment lr * K. Use ``lr_list * step_size_list`` if set on config;
    else equal-partition log_history.json into segments and sum per-step LRs."""
    cfg = approx_unrolling_cfg
    L = cfg.segments
    if cfg.lr_list and cfg.step_size_list:
        return [lr * k for lr, k in zip(cfg.lr_list, cfg.step_size_list)]

    # TODO: parsing 'checkpoint-N' from dir name is fragile.
    per_segment = len(cfg.checkpoints) // L
    ckpt_steps = [
        int(re.match(r"checkpoint-(\d+)$", Path(str(p)).name).group(1))  # type: ignore
        for p in cfg.checkpoints
    ]
    boundaries = [0] + [ckpt_steps[(l + 1) * per_segment - 1] for l in range(L)]
    log_path = Path(str(cfg.checkpoints[0])).parent / "log_history.json"
    with open(log_path) as f:
        step_to_lr = {e["step"]: e["learning_rate"] for e in json.load(f)}
    return [
        sum(
            step_to_lr.get(s, 0.0)
            for s in range(boundaries[l] + 1, boundaries[l + 1] + 1)
        )
        for l in range(L)
    ]


def f_s(eta_K: float) -> Callable[[Tensor], Tensor]:
    """x -> exp(-eta_K*x)."""

    def fn(sigma: Tensor) -> Tensor:
        return torch.exp(-eta_K * sigma)

    return fn


def f_r(eta_K: float) -> Callable[[Tensor], Tensor]:
    """x -> (1 - exp(-eta_K*x)) / x. Limit at x=0 is eta_K."""

    def fn(sigma: Tensor) -> Tensor:
        # Compute as eta_K * ((1 - exp(-x))/x); the parenthesized ratio is in
        # [0, 1] for x ≥ 0 and uses expm1 for accuracy near zero.
        x = eta_K * sigma
        is_zero = x == 0
        x_safe = x.masked_fill(is_zero, 1.0)
        ratio = -torch.expm1(-x_safe) / x_safe
        return eta_K * ratio.masked_fill(is_zero, 1.0)

    return fn
