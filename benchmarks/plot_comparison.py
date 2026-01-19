"""Generate comparison plots for all 3 benchmark methods."""

from __future__ import annotations

import argparse
import socket
import subprocess
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from benchmarks.benchmark_bergson import RunRecord as InMemRecord
from benchmarks.benchmark_bergson import load_records as load_inmem_records
from benchmarks.benchmark_bergson_cli import CLIRunRecord
from benchmarks.benchmark_bergson_cli import load_records as load_cli_records
from benchmarks.benchmark_dattri import RunRecord as DattriRecord
from benchmarks.benchmark_dattri import load_records as load_dattri_records
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


def create_combined_dataframe(
    cli_records: list[CLIRunRecord],
    inmem_records: list[InMemRecord],
    dattri_records: list[DattriRecord],
) -> pd.DataFrame:
    """Create a combined dataframe from all benchmark records."""
    rows = []
    hardware = get_hardware_string()

    # Add CLI records
    for r in cli_records:
        if r.status == "success" and r.total_runtime_seconds is not None:
            rows.append(
                {
                    "method": "bergson-cli",
                    "model_key": r.model_key,
                    "model_params": r.params,
                    "train_tokens": r.train_tokens,
                    "eval_tokens": r.eval_tokens,
                    "runtime_seconds": r.total_runtime_seconds,
                    "num_gpus": r.num_gpus,
                    "hardware": r.hardware or hardware,
                    "projection_dim": r.projection_dim,
                }
            )

    # Add in-memory records
    for r in inmem_records:
        if r.status == "success":
            total = (
                (r.query_seconds or 0) + (r.build_seconds or 0) + (r.score_seconds or 0)
            )
            rows.append(
                {
                    "method": "bergson-inmem",
                    "model_key": r.model_key,
                    "model_params": r.params,
                    "train_tokens": r.train_tokens,
                    "eval_tokens": r.eval_tokens,
                    "runtime_seconds": total,
                    "num_gpus": r.num_gpus,
                    "hardware": hardware,
                    "projection_dim": r.projection_dim,
                }
            )

    # Add dattri records
    for r in dattri_records:
        if r.status == "success" and r.runtime_seconds is not None:
            rows.append(
                {
                    "method": "dattri",
                    "model_key": r.model_key,
                    "model_params": r.params,
                    "train_tokens": r.train_tokens,
                    "eval_tokens": r.eval_tokens,
                    "runtime_seconds": r.runtime_seconds,
                    "num_gpus": r.num_gpus,
                    "hardware": hardware,
                    "projection_dim": r.projection_dim,
                }
            )

    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, output_path: Path, num_gpus: int) -> None:
    """Create comparison plots for all methods."""
    if df.empty:
        print("No data to plot")
        return

    # Define colors and markers for methods
    method_styles = {
        "bergson-cli": {"color": "#1f77b4", "marker": "o", "label": "Bergson CLI"},
        "bergson-inmem": {
            "color": "#2ca02c",
            "marker": "s",
            "label": "Bergson In-Memory",
        },
        "dattri": {"color": "#d62728", "marker": "^", "label": "Dattri"},
    }

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Benchmark Comparison: 3 Methods ({num_gpus} GPU)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Plot 1: Runtime vs train tokens (all methods, one model)
    ax1 = axes[0, 0]
    # Pick the model with the most data points
    model_counts = df.groupby("model_key").size()
    main_model = model_counts.idxmax()
    model_df = df[df["model_key"] == main_model]

    for method, style in method_styles.items():
        subset = model_df[model_df["method"] == method]
        if not subset.empty:
            subset = subset.sort_values("train_tokens")
            ax1.plot(
                subset["train_tokens"],
                subset["runtime_seconds"],
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linewidth=2,
                markersize=8,
            )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Training Tokens", fontsize=12)
    ax1.set_ylabel("Total Runtime (seconds)", fontsize=12)
    ax1.set_title(f"Runtime Scaling ({main_model})", fontsize=14, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.legend(fontsize=10)

    # Plot 2: Runtime vs model size for fixed token count
    ax2 = axes[0, 1]
    # Pick the most common token scale
    token_counts = df.groupby("train_tokens").size()
    if not token_counts.empty:
        main_tokens = token_counts.idxmax()
        token_df = df[df["train_tokens"] == main_tokens]

        for method, style in method_styles.items():
            subset = token_df[token_df["method"] == method]
            if not subset.empty:
                subset = subset.sort_values("model_params")
                ax2.plot(
                    subset["model_params"],
                    subset["runtime_seconds"],
                    marker=style["marker"],
                    color=style["color"],
                    label=style["label"],
                    linewidth=2,
                    markersize=8,
                )

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Model Parameters", fontsize=12)
        ax2.set_ylabel("Total Runtime (seconds)", fontsize=12)
        ax2.set_title(
            f"Model Size Scaling ({format_tokens(main_tokens)} tokens)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax2.legend(fontsize=10)

    # Plot 3: All methods, all models (runtime vs tokens)
    ax3 = axes[1, 0]
    for method, style in method_styles.items():
        method_df = df[df["method"] == method]
        for model_key in method_df["model_key"].unique():
            subset = method_df[method_df["model_key"] == model_key]
            if not subset.empty:
                subset = subset.sort_values("train_tokens")
                ax3.plot(
                    subset["train_tokens"],
                    subset["runtime_seconds"],
                    marker=style["marker"],
                    color=style["color"],
                    label=f"{style['label']} ({model_key})",
                    linewidth=1.5,
                    markersize=6,
                    alpha=0.8,
                )

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Training Tokens", fontsize=12)
    ax3.set_ylabel("Total Runtime (seconds)", fontsize=12)
    ax3.set_title("All Methods & Models", fontsize=14, fontweight="bold")
    ax3.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax3.legend(fontsize=8, ncol=2, loc="upper left")

    # Plot 4: Speedup comparison (dattri / bergson methods)
    ax4 = axes[1, 1]
    speedup_data = []
    for model_key in df["model_key"].unique():
        for train_tokens in df["train_tokens"].unique():
            dattri_subset = df[
                (df["method"] == "dattri")
                & (df["model_key"] == model_key)
                & (df["train_tokens"] == train_tokens)
            ]
            for bergson_method in ["bergson-cli", "bergson-inmem"]:
                bergson_subset = df[
                    (df["method"] == bergson_method)
                    & (df["model_key"] == model_key)
                    & (df["train_tokens"] == train_tokens)
                ]

                if not dattri_subset.empty and not bergson_subset.empty:
                    dattri_time = dattri_subset["runtime_seconds"].iloc[0]
                    bergson_time = bergson_subset["runtime_seconds"].iloc[0]
                    speedup = dattri_time / bergson_time if bergson_time > 0 else None
                    if speedup is not None:
                        speedup_data.append(
                            {
                                "model_key": model_key,
                                "train_tokens": train_tokens,
                                "speedup": speedup,
                                "vs_method": bergson_method,
                            }
                        )

    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        for vs_method in speedup_df["vs_method"].unique():
            method_speedup = speedup_df[speedup_df["vs_method"] == vs_method]
            label = "vs CLI" if vs_method == "bergson-cli" else "vs In-Memory"
            color = "#1f77b4" if vs_method == "bergson-cli" else "#2ca02c"
            for model_key in method_speedup["model_key"].unique():
                subset = method_speedup[
                    method_speedup["model_key"] == model_key
                ].sort_values("train_tokens")
                ax4.plot(
                    subset["train_tokens"],
                    subset["speedup"],
                    marker="o" if "cli" in vs_method else "s",
                    color=color,
                    label=f"{label} ({model_key})",
                    linewidth=1.5,
                    markersize=6,
                    alpha=0.8,
                )

        ax4.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax4.set_xscale("log")
        ax4.set_xlabel("Training Tokens", fontsize=12)
        ax4.set_ylabel("Speedup (Dattri / Bergson)", fontsize=12)
        ax4.set_title(
            "Relative Performance: Dattri vs Bergson", fontsize=14, fontweight="bold"
        )
        ax4.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax4.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for all 3 benchmark methods",
    )
    parser.add_argument(
        "--cli_root",
        default="runs/bergson_cli_benchmark_2",
        help="Root directory for CLI benchmark results",
    )
    parser.add_argument(
        "--inmem_root",
        default="runs/bergson_inmem_benchmark",
        help="Root directory for in-memory benchmark results",
    )
    parser.add_argument(
        "--dattri_root",
        default="runs/dattri_benchmark",
        help="Root directory for dattri benchmark results",
    )
    parser.add_argument(
        "--output_csv",
        default="runs/benchmarks/comparison_benchmark.csv",
        help="Path to save CSV data",
    )
    parser.add_argument(
        "--output_plot",
        default="figures/comparison_benchmark.png",
        help="Path to save plot",
    )
    parser.add_argument(
        "--filter_num_gpus",
        type=int,
        default=1,
        help="Filter to only include runs with this GPU count",
    )

    args = parser.parse_args(argv)

    # Load records from all sources
    cli_root = Path(args.cli_root)
    inmem_root = Path(args.inmem_root)
    dattri_root = Path(args.dattri_root)

    cli_records = load_cli_records(cli_root) if cli_root.exists() else []
    inmem_records = load_inmem_records(inmem_root) if inmem_root.exists() else []
    dattri_records = load_dattri_records(dattri_root) if dattri_root.exists() else []

    print(f"Found {len(cli_records)} CLI benchmark records")
    print(f"Found {len(inmem_records)} in-memory benchmark records")
    print(f"Found {len(dattri_records)} dattri benchmark records")

    if not cli_records and not inmem_records and not dattri_records:
        print("No benchmark records found in any source")
        return

    # Create combined dataframe
    df = create_combined_dataframe(cli_records, inmem_records, dattri_records)

    if df.empty:
        print("No successful benchmark runs found")
        return

    print(f"Total: {len(df)} successful benchmark runs")

    # Filter by GPU count
    df = df[df["num_gpus"] == args.filter_num_gpus]
    print(f"Filtered to {len(df)} runs with {args.filter_num_gpus} GPU(s)")

    if df.empty:
        print(f"No runs found with {args.filter_num_gpus} GPU(s)")
        return

    # Show summary
    print("\nData summary:")
    for method in df["method"].unique():
        method_df = df[df["method"] == method]
        models = sorted(method_df["model_key"].unique())
        tokens = sorted(method_df["train_tokens"].unique())
        print(f"  {method}: {len(method_df)} runs")
        print(f"    Models: {models}")
        print(f"    Token scales: {[format_tokens(t) for t in tokens]}")

    # Save CSV
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    # Create plot
    plot_path = Path(args.output_plot)
    plot_comparison(df, plot_path, args.filter_num_gpus)


if __name__ == "__main__":
    main()
