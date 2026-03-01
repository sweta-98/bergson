"""Benchmark Builder.reduce() variants.

Tests the current implementation against optimized alternatives,
with and without unit normalization and preconditioning.

Usage:
    python scripts/bench_reduce.py [--repeats 200] [--warmup 50]
"""

import argparse
import time
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def reduce_current(
    mod_grads: dict[str, torch.Tensor],
    h_inv: dict[str, torch.Tensor],
    grad_sizes: dict[str, int],
    buffer: torch.Tensor,
    unit_normalize: bool,
    indices: list[int],
):
    """Current implementation from data.py."""
    device = next(iter(mod_grads.values())).device
    eps = torch.finfo(torch.float32).eps

    # Precondition
    if h_inv:
        mod_grads = {name: mod_grads[name] @ h_inv[name] for name in mod_grads}

    if unit_normalize:
        ssqs = torch.zeros(len(indices), device=device)
        for mod_grad in mod_grads.values():
            ssqs += mod_grad.pow(2).sum(dim=-1)
        norms = ssqs.sqrt()
    else:
        norms = torch.ones(len(indices), device=device)

    offset = 0
    for module_name in grad_sizes.keys():
        grads = mod_grads[module_name]
        if unit_normalize:
            grads = grads / (norms.unsqueeze(1) + eps)
        grads = grads.sum(dim=0).to(torch.float32)
        buffer[0, offset : offset + grads.shape[0]] += grads
        offset += grads.shape[0]


def reduce_rsqrt(
    mod_grads: dict[str, torch.Tensor],
    h_inv: dict[str, torch.Tensor],
    grad_sizes: dict[str, int],
    buffer: torch.Tensor,
    unit_normalize: bool,
    indices: list[int],
):
    """Loop accumulation (no stack) with rsqrt + multiply."""
    device = next(iter(mod_grads.values())).device
    eps = torch.finfo(torch.float32).eps

    # Precondition
    if h_inv:
        mod_grads = {name: mod_grads[name] @ h_inv[name] for name in mod_grads}

    if unit_normalize:
        ssqs = torch.zeros(len(indices), device=device)
        for g in mod_grads.values():
            ssqs.add_(g.pow(2).sum(dim=-1))
        inv_norms = ssqs.clamp_min_(eps * eps).rsqrt().unsqueeze(1)
    else:
        inv_norms = None

    offset = 0
    for module_name in grad_sizes.keys():
        grads = mod_grads[module_name]
        if inv_norms is not None:
            grads = grads * inv_norms
        grads = grads.sum(dim=0).to(torch.float32)
        buffer[0, offset : offset + grads.shape[0]] += grads
        offset += grads.shape[0]


def reduce_stack(
    mod_grads: dict[str, torch.Tensor],
    h_inv: dict[str, torch.Tensor],
    grad_sizes: dict[str, int],
    buffer: torch.Tensor,
    unit_normalize: bool,
    indices: list[int],
):
    """Stack accumulation with rsqrt + multiply."""
    eps = torch.finfo(torch.float32).eps

    # Precondition
    if h_inv:
        mod_grads = {name: mod_grads[name] @ h_inv[name] for name in mod_grads}

    if unit_normalize:
        ssqs = torch.stack([g.pow(2).sum(dim=-1) for g in mod_grads.values()]).sum(0)
        inv_norms = ssqs.clamp_min_(eps * eps).rsqrt().unsqueeze(1)
    else:
        inv_norms = None

    offset = 0
    for module_name in grad_sizes.keys():
        grads = mod_grads[module_name]
        if inv_norms is not None:
            grads = grads * inv_norms
        grads = grads.sum(dim=0).to(torch.float32)
        buffer[0, offset : offset + grads.shape[0]] += grads
        offset += grads.shape[0]


# --- Compiled variants ---


@torch.compile(fullgraph=True)
def _compiled_fused_reduce(
    all_grads: torch.Tensor, buffer: torch.Tensor, do_normalize: bool
):
    """Cat all grads into one tensor, normalize + sum in a single fused kernel."""
    eps = torch.finfo(torch.float32).eps
    if do_normalize:
        inv_norms = (
            all_grads.pow(2).sum(dim=-1).clamp_min_(eps * eps).rsqrt().unsqueeze(1)
        )
        all_grads = all_grads * inv_norms
    buffer[0] += all_grads.sum(dim=0).to(torch.float32)


def reduce_compiled(
    mod_grads: dict[str, torch.Tensor],
    h_inv: dict[str, torch.Tensor],
    grad_sizes: dict[str, int],
    buffer: torch.Tensor,
    unit_normalize: bool,
    indices: list[int],
):
    """Cat + compiled fused kernel."""
    # Precondition (not compiled — matmul is already fast, and dict iteration
    # doesn't play well with fullgraph)
    if h_inv:
        mod_grads = {name: mod_grads[name] @ h_inv[name] for name in mod_grads}

    all_grads = torch.cat([mod_grads[m] for m in grad_sizes.keys()], dim=-1)
    _compiled_fused_reduce(all_grads, buffer, unit_normalize)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


@dataclass
class BenchConfig:
    label: str
    unit_normalize: bool
    use_preconditioner: bool


CONFIGS = [
    BenchConfig("dot", unit_normalize=False, use_preconditioner=False),
    BenchConfig("cosine", unit_normalize=True, use_preconditioner=False),
    BenchConfig("dot+precond", unit_normalize=False, use_preconditioner=True),
    BenchConfig("cosine+precond", unit_normalize=True, use_preconditioner=True),
]


@dataclass
class ShapeProfile:
    label: str
    n_modules: int
    grad_dim: int
    batch_size: int


SHAPES = [
    ShapeProfile("6x2048  bs=32", 6, 2048, 32),
    ShapeProfile("6x2048  bs=128", 6, 2048, 128),
    ShapeProfile("6x2048  bs=512", 6, 2048, 512),
    ShapeProfile("12x512  bs=128", 12, 512, 128),
]


def make_fake_data(profile: ShapeProfile, device: torch.device, dtype: torch.dtype):
    modules = [f"module_{i}" for i in range(profile.n_modules)]
    grad_sizes = {m: profile.grad_dim for m in modules}
    mod_grads = {
        m: torch.randn(profile.batch_size, profile.grad_dim, device=device, dtype=dtype)
        for m in modules
    }
    total_dim = profile.n_modules * profile.grad_dim
    buffer = torch.zeros(1, total_dim, device=device, dtype=torch.float32)
    # Well-conditioned preconditioners
    preconditioners = {}
    for m in modules:
        d = profile.grad_dim
        preconditioners[m] = torch.eye(
            d, device=device, dtype=dtype
        ) + 0.1 * torch.randn(d, d, device=device, dtype=dtype)
    return modules, grad_sizes, mod_grads, buffer, preconditioners


def percentile(sorted_vals: list[float], p: float) -> float:
    idx = p / 100.0 * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def bench_round_robin(fns: list, all_kwargs: list[dict], *, warmup: int, repeats: int):
    """Run functions in rotating order. Each function gets its own kwargs
    (separate buffers) to avoid cross-contamination."""
    n = len(fns)
    for _ in range(warmup):
        for fn, kw in zip(fns, all_kwargs):
            fn(**kw)
    torch.cuda.synchronize()

    all_times = [[] for _ in range(n)]
    for i in range(repeats):
        order = [(i + j) % n for j in range(n)]
        torch.cuda.synchronize()
        for slot in order:
            t0 = time.perf_counter()
            fns[slot](**all_kwargs[slot])
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            all_times[slot].append((t1 - t0) * 1e3)

    for t in all_times:
        t.sort()
    return all_times


def fmt_stats(sorted_times: list[float]) -> str:
    p25 = percentile(sorted_times, 25)
    p50 = percentile(sorted_times, 50)
    p75 = percentile(sorted_times, 75)
    return f"{p50:6.3f} [{p25:5.3f}-{p75:5.3f}]"


def run_benchmarks(args):
    device = torch.device("cuda")
    dtype = torch.float32

    IMPLS = [
        ("current", reduce_current),
        ("loop", reduce_rsqrt),
        ("compiled", reduce_compiled),
    ]
    impl_names = [name for name, _ in IMPLS]

    header = (
        f"{'config':<20} "
        + "  ".join(f"{name + ' p50 [p25-p75]':>24}" for name in impl_names)
        + f"  {'l/cur':>7}"
        + f"  {'c/cur':>7}"
        + f"  {'match':>6}"
    )
    sep = "-" * len(header)

    for profile in SHAPES:
        total = profile.n_modules * profile.grad_dim
        print(f"\n=== {profile.label}  (total cols={total}) ===")
        print(header)
        print(sep)

        for cfg in CONFIGS:
            modules, grad_sizes, mod_grads, buffer, preconditioners = make_fake_data(
                profile, device, dtype
            )
            precond = preconditioners if cfg.use_preconditioner else {}
            indices = list(range(profile.batch_size))

            # Each impl gets its own buffer copy to avoid accumulation differences
            def make_kwargs(fn):
                return dict(
                    mod_grads={m: mod_grads[m].clone() for m in modules},
                    h_inv=precond,
                    grad_sizes=grad_sizes,
                    buffer=buffer.clone(),
                    unit_normalize=cfg.unit_normalize,
                    indices=indices,
                )

            # Verify numerical equivalence
            with torch.inference_mode():
                ref_buf = buffer.clone()
                reduce_current(**dict(make_kwargs(None), buffer=ref_buf))
                matches = []
                for name, fn in IMPLS:
                    test_buf = buffer.clone()
                    fn(**dict(make_kwargs(None), buffer=test_buf))
                    max_err = (ref_buf - test_buf).abs().max().item()
                    ref_scale = ref_buf.abs().mean().item()
                    ok = max_err < 1e-3 * max(ref_scale, 1e-6)
                    matches.append(ok)

            with torch.inference_mode():
                fns = [fn for _, fn in IMPLS]
                all_kwargs = [make_kwargs(fn) for fn in fns]
                all_times = bench_round_robin(
                    fns,
                    all_kwargs,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )

            p50s = [percentile(t, 50) for t in all_times]
            r_loop = p50s[1] / p50s[0] if p50s[0] > 0 else float("inf")
            r_comp = p50s[2] / p50s[0] if p50s[0] > 0 else float("inf")
            match_str = "ok" if all(matches) else "ERR"

            print(
                f"{cfg.label:<20} "
                + "  ".join(f"{fmt_stats(t):>24}" for t in all_times)
                + f"  {r_loop:>6.2f}x"
                + f"  {r_comp:>6.2f}x"
                + f"  {match_str:>6}"
            )

    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    run_benchmarks(args)


if __name__ == "__main__":
    main()
