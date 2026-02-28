import math

import torch
from torch import Tensor


def optimal_linear_shrinkage(S_n: Tensor, n: int | Tensor) -> Tensor:
    """Optimal linear shrinkage for a sample covariance matrix or batch thereof.

    Given a sample covariance matrix `S_n` of shape (*, p, p) and a sample size `n`,
    this function computes the optimal shrinkage coefficients `alpha` and `beta`, then
    returns the covariance estimate `alpha * S_n + beta * Sigma0`, where ``Sigma0` is
    an isotropic covariance matrix with the same trace as `S_n`.

    The formula is distribution-free and asymptotically optimal in the Frobenius norm
    among all linear shrinkage estimators as the dimensionality `p` and sample size `n`
    jointly tend to infinity, with the ratio `p / n` converging to a finite positive
    constant `c`. The derivation is based on Random Matrix Theory and assumes that the
    underlying distribution has finite moments up to 4 + eps, for some eps > 0.

    See "On the Strong Convergence of the Optimal Linear Shrinkage Estimator for Large
    Dimensional Covariance Matrix" <https://arxiv.org/abs/1308.2608> for details.

    Args:
        S_n: Sample covariance matrices of shape (*, p, p).
        n: Sample size.
    """
    p = S_n.shape[-1]
    assert n > 1 and S_n.shape[-2:] == (p, p)

    # TODO: Make this configurable, try using diag(S_n) or something
    eye = torch.eye(p, dtype=S_n.dtype, device=S_n.device).expand_as(S_n)
    trace_S = trace(S_n)
    sigma0 = eye * trace_S / p

    sigma0_norm_sq = sigma0.pow(2).sum(dim=(-2, -1), keepdim=True)
    S_norm_sq = S_n.pow(2).sum(dim=(-2, -1), keepdim=True)

    prod_trace = trace(S_n @ sigma0)
    top = trace_S.pow(2) * sigma0_norm_sq / n
    bottom = S_norm_sq * sigma0_norm_sq - prod_trace**2

    # Epsilon prevents dividing by zero for the zero matrix. In that case we end up
    # setting alpha = 0, beta = 1, but it doesn't matter since we're shrinking toward
    # tr(0)*I = 0, so it's a no-op.
    eps = torch.finfo(S_n.dtype).eps

    # Ensure that alpha and beta are in [0, 1] and thereby ensure that the resulting
    # covariance matrix is positive semi-definite.
    alpha = torch.clamp(1 - (top + eps) / (bottom + eps), min=0, max=1)
    beta = (1 - alpha) * (prod_trace + eps) / (sigma0_norm_sq + eps)

    return alpha * S_n + beta * sigma0


@torch.compile
def psd_power(H: Tensor, power: float) -> Tensor:
    """Compute a pseudoinverse power of p.s.d. matrix `H` via eigendecomposition.

    Uses the same tolerance heuristic as `torch.linalg.pinv` to zero out
    eigenvalues that are effectively zero, ensuring numerical stability.

    Args:
        H: Positive semi-definite matrix.
        power: Exponent to apply to eigenvalues (e.g. -0.5 for rsqrt, -1 for inverse).
    """
    eigval, eigvec = torch.linalg.eigh(H)
    eigval = eigval[..., None, :].clamp_min(0.0)

    # Zero out eigenvalues below the tolerance threshold (pseudoinverse).
    # Use the same heuristic as `torch.linalg.pinv` to determine the tolerance.
    thresh = eigval[..., None, -1] * H.shape[-1] * torch.finfo(H.dtype).eps
    result = eigvec * torch.where(eigval > thresh, eigval.pow(power), 0.0) @ eigvec.mH

    return result


@torch.compile
def damped_psd_power(
    H: Tensor,
    power: float,
    damping_factor: float = 0.1,
    dtype: torch.dtype = torch.float64,
    regularizer: Tensor | None = None,
) -> Tensor:
    """Compute a damped power of p.s.d. matrix `H` via eigendecomposition.

    Adds adaptive damping before computing the power to improve numerical stability.

    Args:
        H: Positive semi-definite matrix.
        power: Exponent to apply to eigenvalues (e.g. -0.5 for rsqrt, -1 for inverse).
        damping_factor: Multiplier for the damping term (default: 0.1). Set to
            0 to disable damping.
        dtype: Dtype for intermediate computation (default: float64 for stability).
        regularizer: Optional matrix to use as regularizer instead of identity.
            If provided, computes (H + damping_factor * regularizer)^power.
            If None (default), uses scaled identity:
            (H + damping_factor * |H|_mean * I)^power.

    Returns:
        The damped power of H in the original dtype.
    """
    original_dtype = H.dtype
    H = H.to(dtype=dtype)
    if regularizer is not None:
        regularizer = regularizer.to(dtype=dtype, device=H.device)
        H = H + damping_factor * regularizer
    else:
        damping_val = damping_factor * H.abs().mean()
        H = H + damping_val * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

    eigval, eigvec = torch.linalg.eigh(H)
    return (eigvec * eigval.pow(power) @ eigvec.mH).to(original_dtype)


def trace(matrices: Tensor) -> Tensor:
    """Version of `torch.trace` that works for batches of matrices."""
    diag = torch.linalg.diagonal(matrices)
    return diag.sum(dim=-1, keepdim=True).unsqueeze(-1)


def reshape_to_nearest_square(a: torch.Tensor) -> torch.Tensor:
    """
    Reshape a 2-D (or any-D) tensor into the *most square* 2-D shape
    that preserves the total number of elements.

    Returns
    -------
    out   : reshaped tensor (view when possible)
    shape : tuple (rows, cols) that was chosen
    """
    n = math.prod(a.shape[-2:])
    if n == 0:
        raise ValueError("empty tensor")

    # search divisors closest to sqrt(n)
    root = math.isqrt(n)
    cols, rows = None, None
    for d in range(root, 0, -1):
        if n % d == 0:
            rows = d
            cols = n // d
            break

    if rows is None or cols is None:
        raise ValueError("could not find a valid shape for the tensor")

    return a.reshape(*a.shape[:-2], rows, cols)
