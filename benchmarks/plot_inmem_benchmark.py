"""Generate plots from Bergson in-memory benchmark results."""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from benchmarks.benchmark_bergson import RunRecord, load_records
from benchmarks.benchmark_utils import format_tokens


def get_gpu_info() -> str:
    """Get GPU info string for hardware identification."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=count,name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")
        if lines:
            count, name = lines[0].split(", ", 1)
            return f"{count}x {name}"
    return "Unknown GPU"


def get_hardware_string() -> str:
    """Get a string identifying the current hardware."""
    hostname = socket.gethostname()
    gpu_info = get_gpu_info()
    return f"{hostname} ({gpu_info})"


def create_inmem_dataframe(records: list[RunRecord]) -> pd.DataFrame:
    """Create a dataframe from in-memory benchmark records."""
    rows = []
    hardware = get_hardware_string()

    for r in records:
        if r.status == "success":
            # Calculate total runtime
            total_seconds = (
                (r.query_seconds or 0) + (r.build_seconds or 0) + (r.score_seconds or 0)
            )
            rows.append(
                {
                    "model_key": r.model_key,
                    "model_name": r.model_name,
                    "model_params": r.params,
                    "train_tokens": r.train_tokens,
                    "eval_tokens": r.eval_tokens,
                    "dataset": r.dataset,
                    "batch_size": r.batch_size,
                    "query_seconds": r.query_seconds,
                    "build_seconds": r.build_seconds,
                    "score_seconds": r.score_seconds,
                    "total_runtime_seconds": total_seconds,
                    "run_path": r.run_path,
                    "num_gpus": r.num_gpus,
                    "hardware": hardware,
                    "token_batch_size": r.token_batch_size,
                    "projection_dim": r.projection_dim,
                }
            )

    return pd.DataFrame(rows)


def plot_inmem_benchmark(
    df: pd.DataFrame, output_path: Path, num_gpus: int, hardware: str | None
) -> None:
    """Create plots for in-memory benchmark results."""
    if df.empty:
        print("No data to plot")
        return

    # Create title suffix with GPU/hardware info
    config_str = f"{num_gpus} GPU{'s' if num_gpus > 1 else ''}"
    if hardware:
        config_str += f" - {hardware}"

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Bergson In-Memory Benchmark ({config_str})",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Plot 1: Total runtime vs train tokens (by model)
    ax1 = axes[0, 0]
    for model_key in df["model_key"].unique():
        subset = df[df["model_key"] == model_key]
        subset = subset.sort_values("train_tokens")
        if not subset.empty:
            ax1.plot(
                subset["train_tokens"],
                subset["total_runtime_seconds"],
                marker="o",
                label=model_key,
                linewidth=2,
            )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Training Tokens", fontsize=12)
    ax1.set_ylabel("Total Runtime (seconds)", fontsize=12)
    ax1.set_title(
        "In-Memory: Total Runtime vs Training Tokens",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.legend(fontsize=10)

    # Plot 2: Runtime vs model params (by token scale)
    ax2 = axes[0, 1]
    for train_tokens in sorted(df["train_tokens"].unique())[:5]:
        subset = df[df["train_tokens"] == train_tokens]
        subset = subset.sort_values("model_params")
        if not subset.empty:
            ax2.plot(
                subset["model_params"],
                subset["total_runtime_seconds"],
                marker="o",
                label=format_tokens(train_tokens),
                linewidth=2,
            )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Model Parameters", fontsize=12)
    ax2.set_ylabel("Total Runtime (seconds)", fontsize=12)
    ax2.set_title(
        "In-Memory: Runtime Scaling by Model Size",
        fontsize=14,
        fontweight="bold",
    )
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.legend(fontsize=10)

    # Plot 3: Query vs Build vs Score breakdown
    ax3 = axes[1, 0]
    for model_key in df["model_key"].unique():
        subset = df[df["model_key"] == model_key]
        subset = subset.sort_values("train_tokens")
        if not subset.empty and subset["query_seconds"].notna().any():
            ax3.plot(
                subset["train_tokens"],
                subset["query_seconds"],
                marker="s",
                label=f"{model_key} (query)",
                linewidth=2,
                linestyle="-",
            )
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
        "In-Memory: Query vs Build vs Score Breakdown",
        fontsize=14,
        fontweight="bold",
    )
    ax3.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax3.legend(fontsize=9, ncol=2)

    # Plot 4: Stacked bar chart of time breakdown for a specific model
    ax4 = axes[1, 1]
    model_counts = df["model_key"].value_counts()
    if not model_counts.empty:
        main_model = model_counts.index[0]
        subset = df[df["model_key"] == main_model]
        subset = subset.sort_values("train_tokens")

        if not subset.empty and len(subset) > 0:
            x_labels = [format_tokens(t) for t in subset["train_tokens"]]
            x_pos = range(len(x_labels))

            query_times = subset["query_seconds"].fillna(0).astype(float).values
            build_times = subset["build_seconds"].fillna(0).astype(float).values
            score_times = subset["score_seconds"].fillna(0).astype(float).values

            ax4.bar(x_pos, query_times, label="Query", alpha=0.8)
            ax4.bar(x_pos, build_times, bottom=query_times, label="Build", alpha=0.8)
            ax4.bar(
                x_pos,
                score_times,
                bottom=query_times + build_times,
                label="Score",
                alpha=0.8,
            )

            ax4.set_xticks(list(x_pos))
            ax4.set_xticklabels(x_labels, rotation=45, ha="right")
            ax4.set_xlabel("Training Tokens", fontsize=12)
            ax4.set_ylabel("Runtime (seconds)", fontsize=12)
            ax4.set_title(
                f"Time Breakdown for {main_model}",
                fontsize=14,
                fontweight="bold",
            )
            ax4.legend(fontsize=10)
            ax4.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved in-memory benchmark plot to {output_path}")


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
        "--output_csv",
        required=True,
        help="Path to save CSV data",
    )
    parser.add_argument(
        "--output_plot",
        required=True,
        help="Path to save plot",
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
    csv_path = Path(args.output_csv)
    plot_path = Path(args.output_plot)

    for (num_gpus, hardware), group_df in groups:
        # Save CSV for this group
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        group_df.to_csv(csv_path, index=False)
        print(f"\nSaved CSV for {num_gpus} GPU(s) to {csv_path}")

        # Create plot for this group
        plot_inmem_benchmark(group_df, plot_path, num_gpus, hardware)

    print(
        f"\nGenerated {len(groups)} separate plots (one per GPU/hardware configuration)"
    )


if __name__ == "__main__":
    main()
