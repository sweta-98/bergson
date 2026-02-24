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
def psd_rsqrt(A: Tensor) -> Tensor:
    """Efficiently compute the p.s.d. pseudoinverse sqrt of p.s.d. matrix `A`."""
    L, U = torch.linalg.eigh(A)
    L = L[..., None, :].clamp_min(0.0)

    # We actually compute the pseudo-inverse here for numerical stability.
    # Use the same heuristic as `torch.linalg.pinv` to determine the tolerance.
    thresh = L[..., None, -1] * A.shape[-1] * torch.finfo(A.dtype).eps
    rsqrt = U * torch.where(L > thresh, L.rsqrt(), 0.0) @ U.mH

    return rsqrt


def compute_damped_inverse(
    H: Tensor,
    damping_factor: float = 0.1,
    dtype: torch.dtype = torch.float64,
    regularizer: Tensor | None = None,
) -> Tensor:
    """Compute H^(-1) with damping for numerical stability.

    Uses eigendecomposition to compute the inverse of a positive semi-definite
    matrix with adaptive damping based on the matrix's mean absolute value.

    Args:
        H: Positive semi-definite matrix to invert.
        damping_factor: Multiplier for the damping term (default: 0.1).
        dtype: Dtype for intermediate computation (default: float64 for stability).
        regularizer: Optional matrix to use as regularizer instead of identity.
            If provided, computes inv(H + damping_factor * regularizer).
            If None (default), uses scaled identity:
            inv(H + damping_factor * |H|_mean * I).

    Returns:
        The damped inverse H^(-1) in the original dtype of H.
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
    return (eigvec * (1.0 / eigval) @ eigvec.mT).to(original_dtype)


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
