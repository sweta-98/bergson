"""Benchmark: compiled scorer vs eager scorer.

Compares the torch.compiled fused scoring kernel against the eager
(non-compiled) equivalent across all four (cosine_sim x precondition)
configurations and multiple shape profiles.

Usage:
    python scripts/bench_cat_vs_loop.py [--repeats 200] [--warmup 50]
"""

import argparse
import time
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Compiled kernel (same as scorer.py)
# ---------------------------------------------------------------------------


@torch.compile(fullgraph=True)
def _fused_cosine_score(
    index_grads: torch.Tensor,
    query_grads_t: torch.Tensor,
) -> torch.Tensor:
    scores = index_grads @ query_grads_t
    i_norm = index_grads.pow(2).sum(dim=1).sqrt().clamp_min_(1e-12).unsqueeze(1)
    scores.div_(i_norm)
    return scores


# ---------------------------------------------------------------------------
# Implementations: what scorer.py does now vs what it used to do
# ---------------------------------------------------------------------------


def score_old(
    all_index: torch.Tensor,
    query_grads_t: torch.Tensor,
    unit_normalize: bool,
) -> torch.Tensor:
    """Old eager path for everything."""
    scores = all_index @ query_grads_t
    if unit_normalize:
        i_norm = all_index.pow(2).sum(dim=1).sqrt().clamp_min_(1e-12).unsqueeze(1)
        scores.div_(i_norm)
    return scores


def score_new(
    all_index: torch.Tensor,
    query_grads_t: torch.Tensor,
    unit_normalize: bool,
) -> torch.Tensor:
    """New path: compiled for cosine, eager matmul for dot."""
    if unit_normalize:
        return _fused_cosine_score(all_index, query_grads_t)
    else:
        return all_index @ query_grads_t


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
    n_queries: int
    batch_size: int


SHAPES = [
    ShapeProfile("6x2048  bs=512  nq=10", 6, 2048, 10, 512),
    ShapeProfile("12x512  bs=2048 nq=10", 12, 512, 10, 2048),
    ShapeProfile("4x4096  bs=256  nq=10", 4, 4096, 10, 256),
    ShapeProfile("6x2048  bs=512  nq=100", 6, 2048, 100, 512),
]


def make_fake_data(profile: ShapeProfile, device: torch.device, dtype: torch.dtype):
    modules = [f"module_{i}" for i in range(profile.n_modules)]
    query_grads = {
        m: torch.randn(profile.n_queries, profile.grad_dim, device=device, dtype=dtype)
        for m in modules
    }
    index_grads = {
        m: torch.randn(profile.batch_size, profile.grad_dim, device=device, dtype=dtype)
        for m in modules
    }
    preconditioners = {}
    for m in modules:
        d = profile.grad_dim
        preconditioners[m] = torch.eye(
            d, device=device, dtype=dtype
        ) + 0.1 * torch.randn(d, d, device=device, dtype=dtype)
    return modules, query_grads, index_grads, preconditioners


def prepare_inputs(
    modules: list[str],
    query_grads: dict[str, torch.Tensor],
    index_grads: dict[str, torch.Tensor],
    preconditioners: dict[str, torch.Tensor],
    unit_normalize: bool,
):
    """Pre-cat and precondition, returning tensors ready for the scoring kernel."""
    if preconditioners:
        q_list = [query_grads[m] @ preconditioners[m] for m in modules]
    else:
        q_list = [query_grads[m] for m in modules]
    query_grads_t = torch.cat(q_list, dim=-1).T

    i_list = []
    for m in modules:
        g = index_grads[m]
        if unit_normalize and preconditioners and m in preconditioners:
            g = g @ preconditioners[m]
        i_list.append(g)
    all_index = torch.cat(i_list, dim=-1)

    return all_index, query_grads_t


def percentile(sorted_vals: list[float], p: float) -> float:
    idx = p / 100.0 * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def bench_round_robin(fns: list, all_kwargs: list[dict], *, warmup: int, repeats: int):
    """Run functions in rotating order. Each function gets its own kwargs."""
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
    return f"{p50:6.2f} [{p25:5.2f}-{p75:5.2f}]"


def run_benchmarks(args):
    device = torch.device("cuda")
    dtype = torch.float32

    IMPLS = [
        ("old", score_old),
        ("new", score_new),
    ]
    impl_names = [name for name, _ in IMPLS]

    header = (
        f"{'config':<20} "
        + "  ".join(f"{name + ' p50 [p25-p75]':>22}" for name in impl_names)
        + f"  {'new/old':>7}"
        + f"  {'match':>6}"
    )
    sep = "-" * len(header)

    for profile in SHAPES:
        modules, query_grads, index_grads, preconditioners = make_fake_data(
            profile, device, dtype
        )

        total = profile.n_modules * profile.grad_dim
        print(f"\n=== {profile.label}  (total cols={total}) ===")
        print(header)
        print(sep)

        for cfg in CONFIGS:
            precond = preconditioners if cfg.use_preconditioner else {}
            all_index, query_grads_t = prepare_inputs(
                modules,
                query_grads,
                index_grads,
                precond,
                cfg.unit_normalize,
            )

            kwargs = dict(
                all_index=all_index,
                query_grads_t=query_grads_t,
                unit_normalize=cfg.unit_normalize,
            )

            # Verify numerical equivalence
            with torch.inference_mode():
                ref = score_old(**kwargs)
                matches = []
                for name, fn in IMPLS:
                    out = fn(**kwargs)
                    max_err = (ref - out).abs().max().item()
                    rel_scale = ref.abs().mean().item()
                    matches.append(max_err < 1e-2 * max(rel_scale, 1e-6))

            with torch.inference_mode():
                fns = [fn for _, fn in IMPLS]
                all_kwargs = [dict(**kwargs) for _ in fns]
                all_times = bench_round_robin(
                    fns,
                    all_kwargs,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )

            p50s = [percentile(t, 50) for t in all_times]
            ratio = p50s[1] / p50s[0] if p50s[0] > 0 else float("inf")
            match_str = "ok" if all(matches) else "ERR"

            print(
                f"{cfg.label:<20} "
                + "  ".join(f"{fmt_stats(t):>22}" for t in all_times)
                + f"  {ratio:>6.2f}x"
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
