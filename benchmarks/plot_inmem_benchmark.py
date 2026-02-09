"""Generate plots from Bergson in-memory benchmark results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from benchmarks.benchmark_bergson import RunRecord, load_records
from benchmarks.benchmark_utils import (
    extract_gpu_info,
    format_tokens,
    get_hardware_info,
)


def create_inmem_dataframe(
    records: list[RunRecord],
) -> pd.DataFrame:
    """Create a dataframe from in-memory benchmark records."""
    rows = []

    for r in records:
        if r.status == "success":
            # Prefer hardware from the record; fall back to
            # current machine for old records without it.
            hw = getattr(r, "hardware", None) or get_hardware_info()
            rows.append(
                {
                    "model_key": r.model_key,
                    "model_name": r.model_name,
                    "model_params": r.params,
                    "train_tokens": r.train_tokens,
                    "eval_tokens": r.eval_tokens,
                    "dataset": r.dataset,
                    "batch_size": r.batch_size,
                    "build_seconds": r.build_seconds,
                    "score_seconds": r.score_seconds,
                    "run_path": r.run_path,
                    "num_gpus": r.num_gpus,
                    "hardware": hw,
                    "gpu_name": getattr(r, "gpu_name", None),
                    "num_gpus_available": getattr(r, "num_gpus_available", None),
                    "gpu_vram_gb": getattr(r, "gpu_vram_gb", None),
                    "token_batch_size": r.token_batch_size,
                    "projection_dim": r.projection_dim,
                }
            )

    return pd.DataFrame(rows)


def plot_inmem_benchmark(df: pd.DataFrame, output_path: Path, config_str: str) -> None:
    """Create plots for in-memory benchmark results."""
    if df.empty:
        print("No data to plot")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Bergson In-Memory Benchmark ({config_str})",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Plot 1: Score runtime vs train tokens (by model)
    ax1 = axes[0]
    for model_key in df["model_key"].unique():
        subset = df[df["model_key"] == model_key]
        subset = subset.sort_values("train_tokens")
        if not subset.empty:
            ax1.plot(
                subset["train_tokens"],
                subset["score_seconds"],
                marker="o",
                label=model_key,
                linewidth=2,
            )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Training Tokens", fontsize=12)
    ax1.set_ylabel("Score Runtime (seconds)", fontsize=12)
    ax1.set_title(
        "In-Memory: Score Runtime vs Training Tokens",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.legend(fontsize=10)

    # Plot 2: Score runtime vs model params (by token scale)
    ax2 = axes[1]
    for train_tokens in sorted(df["train_tokens"].unique())[:5]:
        subset = df[df["train_tokens"] == train_tokens]
        subset = subset.sort_values("model_params")
        if not subset.empty:
            ax2.plot(
                subset["model_params"],
                subset["score_seconds"],
                marker="o",
                label=format_tokens(train_tokens),
                linewidth=2,
            )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Model Parameters", fontsize=12)
    ax2.set_ylabel("Score Runtime (seconds)", fontsize=12)
    ax2.set_title(
        "In-Memory: Score Runtime Scaling by Model Size",
        fontsize=14,
        fontweight="bold",
    )
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.legend(fontsize=10)

    # Plot 3: Build vs Score breakdown
    ax3 = axes[2]
    for model_key in df["model_key"].unique():
        subset = df[df["model_key"] == model_key]
        subset = subset.sort_values("train_tokens")
        if not subset.empty and subset["build_seconds"].notna().any():
            ax3.plot(
                subset["train_tokens"],
                subset["build_seconds"],
                marker="^",
                label=f"{model_key} (build)",
                linewidth=2,
                linestyle="--",
            )
        if not subset.empty and subset["score_seconds"].notna().any():
            ax3.plot(
                subset["train_tokens"],
                subset["score_seconds"],
                marker="D",
                label=f"{model_key} (score)",
                linewidth=2,
                linestyle=":",
            )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Training Tokens", fontsize=12)
    ax3.set_ylabel("Runtime (seconds)", fontsize=12)
    ax3.set_title(
        "In-Memory: Build vs. Score",
        fontsize=14,
        fontweight="bold",
    )
    ax3.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax3.legend(fontsize=9, ncol=2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots from Bergson in-memory benchmark results",
    )
    parser.add_argument(
        "--run_root",
        required=True,
        help="Root directory containing in-memory benchmark results",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save CSV data",
    )
    parser.add_argument(
        "--filter_num_gpus",
        type=int,
        default=None,
        help="Filter to only include runs with this GPU count",
    )

    args = parser.parse_args(argv)

    # Load records
    run_root = Path(args.run_root)
    if not run_root.exists():
        print(
            f"Error: Run root directory does not exist: {run_root}, not plotting",
            file=sys.stderr,
        )
        return

    records = load_records(run_root)

    if not records:
        print(f"Warning: No benchmark records found in {run_root}", file=sys.stderr)
        return

    print(f"Found {len(records)} benchmark records")

    # Create dataframe
    df = create_inmem_dataframe(records)

    if df.empty:
        print("Warning: No successful benchmark runs found", file=sys.stderr)
        return

    print(f"Loaded {len(df)} successful benchmark runs")

    # Filter by GPU count if specified
    if args.filter_num_gpus is not None:
        df = df[df["num_gpus"] == args.filter_num_gpus]
        print(f"Filtered to {len(df)} runs with {args.filter_num_gpus} GPU(s)")
        if df.empty:
            print(
                f"Warning: No runs found with {args.filter_num_gpus} GPU(s)",
                file=sys.stderr,
            )
            return

    # Group by (num_gpus, hardware) and create separate plots
    groups = df.groupby(["num_gpus", "hardware"], dropna=False)

    if len(groups) == 0:
        print("Warning: No data to plot after grouping", file=sys.stderr)
        return

    print(f"\nFound {len(groups)} different GPU/hardware configurations:")
    for (num_gpus, hardware), group_df in groups:
        print(f"  - {num_gpus} GPU(s), {hardware}: {len(group_df)} runs")

    # Generate plots for each group
    output_path = Path(args.output_path)

    for (num_gpus, hardware), group_df in groups:
        # Create title suffix with GPU/hardware info
        gpu_info = extract_gpu_info(hardware)
        print(f"GPU info: {gpu_info}")
        print(f"Num GPUs: {num_gpus}")
        if gpu_info and gpu_info.startswith("8x") and num_gpus != 8:
            gpu_info = gpu_info.replace("8x", f"{num_gpus}x")
        print(f"GPU info: {gpu_info}")

        config_str = gpu_info or f"{num_gpus} GPU{'s' if num_gpus > 1 else ''}"

        plot_path = output_path / f"inmem_benchmark_{config_str.replace(' ', '_')}.png"
        csv_path = (
            output_path
            / "archive"
            / f"inmem_benchmark_{config_str.replace(' ', '_')}.csv"
        )

        # Save CSV for this group
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        group_df.to_csv(csv_path, index=False)
        print(f"\nSaved CSV for {num_gpus} GPU(s) to {csv_path}")

        # Create plot for this group
        plot_inmem_benchmark(group_df, plot_path, config_str)
        print(f"Saved in-memory benchmark plot to {plot_path}")

    print(
        f"\nGenerated {len(groups)} separate plots (one per GPU/hardware configuration)"
    )


if __name__ == "__main__":
    main()
