from typing import Callable

import torch
from torch import Tensor


def f_s(eta_K: float) -> Callable[[Tensor], Tensor]:
    """x -> exp(-eta_K*x)."""

    def fn(sigma: Tensor) -> Tensor:
        return torch.exp(-eta_K * sigma)

    return fn


def f_r(eta_K: float) -> Callable[[Tensor], Tensor]:
    """x -> (1 - exp(-eta_K*x)) / x. Limit at x=0 is -eta_K."""

    def fn(sigma: Tensor) -> Tensor:
        is_zero = sigma == 0
        result = -torch.expm1(-eta_K * sigma) / sigma.masked_fill(is_zero, 1.0)
        return result.masked_fill(is_zero, eta_K)

    return fn
