#!/usr/bin/env python3
"""Inspect normalizer and preconditioner value distributions from existing trackstar runs.

Loads saved normalizers.pth and preconditioners.pth from completed runs and
reports their value distributions, conditioning, and potential issues.

Usage:
    python scripts/inspect_normalizers_preconditioners.py [RUN_PATH]
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bergson.gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    GradientProcessor,
    Normalizer,
)


def tensor_stats(t: torch.Tensor, name: str, indent: int = 4):
    """Print distribution stats for a tensor."""
    prefix = " " * indent
    t_f = t.float().cpu()
    print(f"{prefix}{name}:")
    print(f"{prefix}  shape={list(t.shape)}  dtype={t.dtype}")
    print(f"{prefix}  min={t_f.min().item():.6e}  max={t_f.max().item():.6e}")
    print(f"{prefix}  mean={t_f.mean().item():.6e}  std={t_f.std().item():.6e}")
    n_zero = (t_f == 0).sum().item()
    n_neg = (t_f < 0).sum().item()
    n_nan = t_f.isnan().sum().item()
    n_inf = t_f.isinf().sum().item()
    if n_zero or n_neg or n_nan or n_inf:
        print(f"{prefix}  zeros={n_zero}  negatives={n_neg}  nans={n_nan}  infs={n_inf}")
    # Percentiles
    pcts = ""
    for p in [1, 5, 25, 50, 75, 95, 99]:
        val = torch.quantile(t_f, p / 100).item()
        pcts += f"p{p}={val:.4e} "
    print(f"{prefix}  {pcts}")


def inspect_normalizers(proc: GradientProcessor, label: str):
    """Inspect all normalizers in a GradientProcessor."""
    print(f"\n{'='*80}")
    print(f"NORMALIZERS — {label} ({len(proc.normalizers)} modules)")
    print(f"{'='*80}")

    n_adafactor = sum(1 for n in proc.normalizers.values() if isinstance(n, AdafactorNormalizer))
    n_adam = sum(1 for n in proc.normalizers.values() if isinstance(n, AdamNormalizer))
    print(f"  Types: {n_adafactor} Adafactor, {n_adam} Adam")

    all_row_mins = []
    all_col_mins = []
    zero_row_modules = []
    zero_col_modules = []

    for name, norm in sorted(proc.normalizers.items()):
        if isinstance(norm, AdafactorNormalizer):
            row_min = norm.row.min().item()
            col_min = norm.col.min().item()
            all_row_mins.append(row_min)
            all_col_mins.append(col_min)
            if row_min == 0:
                zero_row_modules.append(name)
            if col_min == 0:
                zero_col_modules.append(name)

    # Print summary first
    if all_row_mins:
        print(f"\n  Adafactor row min across all modules: {min(all_row_mins):.6e}")
        print(f"  Adafactor col min across all modules: {min(all_col_mins):.6e}")
        if zero_row_modules:
            print(f"  *** {len(zero_row_modules)} modules have zero row entries ***")
            for m in zero_row_modules[:5]:
                print(f"      {m}")
        if zero_col_modules:
            print(f"  *** {len(zero_col_modules)} modules have zero col entries ***")
            for m in zero_col_modules[:5]:
                print(f"      {m}")

    # Detailed stats for a sample of modules
    module_names = sorted(proc.normalizers.keys())
    if len(module_names) > 10:
        # First 3, middle 2, last 3
        sample = (
            module_names[:3]
            + module_names[len(module_names) // 2 - 1 : len(module_names) // 2 + 1]
            + module_names[-3:]
        )
    else:
        sample = module_names

    for name in sample:
        norm = proc.normalizers[name]
        short_name = name.split(".")[-3] + "." + ".".join(name.split(".")[-2:])
        print(f"\n  [{short_name}]")

        if isinstance(norm, AdafactorNormalizer):
            tensor_stats(norm.row, "row [O]")
            tensor_stats(norm.col, "col [I]")

            # Show the effective normalization factors
            r, c = norm.row.add(1e-30), norm.col.add(1e-30)
            denom = r.mean()
            a = denom.sqrt() * r.rsqrt()
            b = c.rsqrt()
            tensor_stats(a, "row_factor (denom.sqrt * row.rsqrt)")
            tensor_stats(b, "col_factor (col.rsqrt)")

            if norm.bias_avg_sq is not None:
                tensor_stats(norm.bias_avg_sq, "bias_avg_sq")

        elif isinstance(norm, AdamNormalizer):
            tensor_stats(norm.weight_avg_sq, "weight_avg_sq [O, I]")
            if norm.bias_avg_sq is not None:
                tensor_stats(norm.bias_avg_sq, "bias_avg_sq")


def inspect_preconditioners(proc: GradientProcessor, label: str):
    """Inspect all preconditioners (H matrices) in a GradientProcessor."""
    if not proc.preconditioners:
        print(f"\n  No preconditioners in {label}")
        return

    print(f"\n{'='*80}")
    print(f"PRECONDITIONERS (H = E[P^T P]) — {label} ({len(proc.preconditioners)} modules)")
    print(f"{'='*80}")

    module_names = sorted(proc.preconditioners.keys())
    if len(module_names) > 10:
        sample = (
            module_names[:3]
            + module_names[len(module_names) // 2 - 1 : len(module_names) // 2 + 1]
            + module_names[-3:]
        )
    else:
        sample = module_names

    for name in sample:
        H = proc.preconditioners[name].float().cpu()
        short_name = name.split(".")[-3] + "." + ".".join(name.split(".")[-2:])
        print(f"\n  [{short_name}]  shape={list(H.shape)}")

        tensor_stats(H, "H values")

        # Symmetry check
        asym = (H - H.T).abs().max().item()
        print(f"      asymmetry (max|H - H^T|): {asym:.6e}")

        # Eigendecomposition
        eigvals = torch.linalg.eigvalsh(H.double()).float()
        tensor_stats(eigvals, "eigenvalues")

        pos_eigvals = eigvals[eigvals > 1e-10]
        neg_eigvals = eigvals[eigvals < -1e-10]
        if len(pos_eigvals) > 0:
            cond = pos_eigvals.max().item() / pos_eigvals.min().item()
            print(f"      condition number: {cond:.2e}")
        if len(neg_eigvals) > 0:
            print(f"      *** {len(neg_eigvals)} significantly negative eigenvalues ***")

        # Effective rank
        all_pos = eigvals.clamp(min=0)
        if all_pos.sum() > 0:
            p = all_pos / all_pos.sum()
            p = p[p > 0]
            entropy = -(p * p.log()).sum().item()
            eff_rank = math.exp(entropy)
            print(f"      effective rank: {eff_rank:.1f} / {H.shape[0]}")

        # Diagonal dominance
        diag = H.diag()
        off_diag_max = (H - torch.diag(diag)).abs().max().item()
        print(f"      diag mean={diag.mean().item():.4e}  off-diag max={off_diag_max:.4e}")

    # Eigenvalue spectra summary across all modules
    if proc.preconditioners_eigen:
        print(f"\n  --- Eigenvalue spectra summary (all modules) ---")
        all_eigvals = []
        for name, (eigvals, _) in proc.preconditioners_eigen.items():
            all_eigvals.append(eigvals.float().cpu())
        all_eigvals = torch.cat(all_eigvals)
        tensor_stats(all_eigvals, f"all eigenvalues ({len(all_eigvals)} total)")
        n_neg = (all_eigvals < 0).sum().item()
        if n_neg > 0:
            print(f"      *** {n_neg}/{len(all_eigvals)} negative eigenvalues across all modules ***")


def inspect_query_grads(run_path: Path):
    """Inspect query gradient distributions."""
    query_path = run_path / "query"
    grad_file = query_path / "gradients.bin"
    info_file = query_path / "info.json"

    if not grad_file.exists():
        print(f"\n  No query gradients at {query_path}")
        return

    import json
    import numpy as np

    with open(info_file) as f:
        info = json.load(f)

    print(f"\n{'='*80}")
    print(f"QUERY GRADIENTS — {query_path}")
    print(f"{'='*80}")
    print(f"  num_grads: {info['num_grads']}")
    print(f"  base_dtype: {info['base_dtype']}")
    print(f"  grad_sizes: {len(info['grad_sizes'])} modules")

    total_dim = sum(info["grad_sizes"].values())
    print(f"  total grad dim: {total_dim}")

    from bergson.data import load_gradients

    grads = load_gradients(query_path, structured=False)
    grads_t = torch.from_numpy(np.array(grads))

    tensor_stats(grads_t, f"query gradients [{grads_t.shape[0]} x {grads_t.shape[1]}]")

    # Per-module stats
    offset = 0
    module_norms = {}
    for name, size in info["grad_sizes"].items():
        chunk = grads_t[:, offset : offset + size]
        norm = chunk.float().norm(dim=1).mean().item()
        module_norms[name] = norm
        offset += size

    # Show top/bottom modules by gradient norm
    sorted_modules = sorted(module_norms.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 5 modules by mean gradient norm:")
    for name, norm in sorted_modules[:5]:
        short = ".".join(name.split(".")[-3:])
        print(f"    {short}: {norm:.6e}")
    print(f"  Bottom 5 modules by mean gradient norm:")
    for name, norm in sorted_modules[-5:]:
        short = ".".join(name.split(".")[-3:])
        print(f"    {short}: {norm:.6e}")

    # If multiple gradients, check cosine similarity
    if grads_t.shape[0] > 1:
        norms = grads_t.float().norm(dim=1, keepdim=True)
        normed = grads_t.float() / norms.clamp(min=1e-8)
        sim = normed @ normed.T
        triu = sim[torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)]
        print(f"\n  Pairwise cosine similarity ({len(triu)} pairs):")
        print(f"    mean={triu.mean().item():.4f}  std={triu.std().item():.4f}")
        print(f"    min={triu.min().item():.4f}  max={triu.max().item():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_path",
        nargs="?",
        default="runs/olmo_wmdp",
        help="Path to a completed trackstar run directory",
    )
    args = parser.parse_args()
    run_path = Path(args.run_path)

    subdirs = ["value_preconditioner", "query_preconditioner", "mixed_preconditioner", "query"]

    for subdir in subdirs:
        proc_path = run_path / subdir
        if not (proc_path / "normalizers.pth").exists():
            print(f"\nSkipping {subdir} (no normalizers.pth)")
            continue

        skip_precond = not (proc_path / "preconditioners.pth").exists()
        proc = GradientProcessor.load(proc_path, skip_preconditioners=skip_precond)

        inspect_normalizers(proc, f"{run_path.name}/{subdir}")
        if not skip_precond:
            inspect_preconditioners(proc, f"{run_path.name}/{subdir}")

    # Inspect query gradients
    inspect_query_grads(run_path)


if __name__ == "__main__":
    main()
