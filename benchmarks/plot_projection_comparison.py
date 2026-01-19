"""Generate comparison plots for projection vs no-projection benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from benchmarks.benchmark_bergson import RunRecord as InMemRecord
from benchmarks.benchmark_bergson import load_records as load_inmem_records
from benchmarks.benchmark_dattri import RunRecord as DattriRecord
from benchmarks.benchmark_dattri import load_records as load_dattri_records
from benchmarks.benchmark_utils import format_tokens


def create_dataframe(
    bergson_proj: list[InMemRecord],
    bergson_noproj: list[InMemRecord],
    dattri_proj: list[DattriRecord],
    dattri_noproj: list[DattriRecord],
) -> pd.DataFrame:
    """Create a combined dataframe from all benchmark records."""
    rows = []

    # Bergson with projection
    for r in bergson_proj:
        if r.status == "success":
            total = (
                (r.query_seconds or 0) + (r.build_seconds or 0) + (r.score_seconds or 0)
            )
            rows.append(
                {
                    "method": "bergson",
                    "projection": "with",
                    "model_key": r.model_key,
                    "model_params": r.params,
                    "train_tokens": r.train_tokens,
                    "runtime_seconds": total,
                }
            )

    # Bergson without projection
    for r in bergson_noproj:
        if r.status == "success":
            total = (
                (r.query_seconds or 0) + (r.build_seconds or 0) + (r.score_seconds or 0)
            )
            rows.append(
                {
                    "method": "bergson",
                    "projection": "without",
                    "model_key": r.model_key,
                    "model_params": r.params,
                    "train_tokens": r.train_tokens,
                    "runtime_seconds": total,
                }
            )

    # Dattri with projection
    for r in dattri_proj:
        if r.status == "success" and r.runtime_seconds is not None:
            rows.append(
                {
                    "method": "dattri",
                    "projection": "with",
                    "model_key": r.model_key,
                    "model_params": r.params,
                    "train_tokens": r.train_tokens,
                    "runtime_seconds": r.runtime_seconds,
                }
            )

    # Dattri without projection
    for r in dattri_noproj:
        if r.status == "success" and r.runtime_seconds is not None:
            rows.append(
                {
                    "method": "dattri",
                    "projection": "without",
                    "model_key": r.model_key,
                    "model_params": r.params,
                    "train_tokens": r.train_tokens,
                    "runtime_seconds": r.runtime_seconds,
                }
            )

    return pd.DataFrame(rows)


def plot_projection_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create comparison plots for projection vs no-projection."""
    if df.empty:
        print("No data to plot")
        return

    # Define styles
    styles = {
        ("bergson", "with"): {
            "color": "#2ca02c",
            "marker": "o",
            "linestyle": "-",
            "label": "Bergson (proj)",
        },
        ("bergson", "without"): {
            "color": "#2ca02c",
            "marker": "o",
            "linestyle": "--",
            "label": "Bergson (no proj)",
        },
        ("dattri", "with"): {
            "color": "#d62728",
            "marker": "^",
            "linestyle": "-",
            "label": "Dattri (proj)",
        },
        ("dattri", "without"): {
            "color": "#d62728",
            "marker": "^",
            "linestyle": "--",
            "label": "Dattri (no proj)",
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Projection vs No-Projection Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Plot 1: Runtime vs tokens for each model (all methods)
    ax1 = axes[0, 0]
    model_counts = df.groupby("model_key").size()
    if not model_counts.empty:
        main_model = model_counts.idxmax()
        model_df = df[df["model_key"] == main_model]

        for (method, proj), style in styles.items():
            subset = model_df[
                (model_df["method"] == method) & (model_df["projection"] == proj)
            ]
            if not subset.empty:
                subset = subset.sort_values("train_tokens")
                ax1.plot(
                    subset["train_tokens"],
                    subset["runtime_seconds"],
                    marker=style["marker"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    label=style["label"],
                    linewidth=2,
                    markersize=8,
                )

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Training Tokens", fontsize=12)
        ax1.set_ylabel("Runtime (seconds)", fontsize=12)
        ax1.set_title(f"Runtime Scaling ({main_model})", fontsize=14, fontweight="bold")
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax1.legend(fontsize=10)

    # Plot 2: Projection speedup by method (proj_time / noproj_time)
    ax2 = axes[0, 1]
    speedup_data = []
    for method in ["bergson", "dattri"]:
        for model_key in df["model_key"].unique():
            for train_tokens in df["train_tokens"].unique():
                proj_df = df[
                    (df["method"] == method)
                    & (df["projection"] == "with")
                    & (df["model_key"] == model_key)
                    & (df["train_tokens"] == train_tokens)
                ]
                noproj_df = df[
                    (df["method"] == method)
                    & (df["projection"] == "without")
                    & (df["model_key"] == model_key)
                    & (df["train_tokens"] == train_tokens)
                ]
                if not proj_df.empty and not noproj_df.empty:
                    proj_time = proj_df["runtime_seconds"].iloc[0]
                    noproj_time = noproj_df["runtime_seconds"].iloc[0]
                    # Speedup = noproj / proj (how much faster is projection)
                    speedup = noproj_time / proj_time if proj_time > 0 else None
                    if speedup is not None:
                        speedup_data.append(
                            {
                                "method": method,
                                "model_key": model_key,
                                "train_tokens": train_tokens,
                                "speedup": speedup,
                            }
                        )

    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        for method in ["bergson", "dattri"]:
            color = "#2ca02c" if method == "bergson" else "#d62728"
            method_df = speedup_df[speedup_df["method"] == method]
            for model_key in method_df["model_key"].unique():
                subset = method_df[method_df["model_key"] == model_key].sort_values(
                    "train_tokens"
                )
                ax2.plot(
                    subset["train_tokens"],
                    subset["speedup"],
                    marker="o" if method == "bergson" else "^",
                    color=color,
                    label=f"{method} ({model_key})",
                    linewidth=1.5,
                    markersize=6,
                )

        ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax2.set_xscale("log")
        ax2.set_xlabel("Training Tokens", fontsize=12)
        ax2.set_ylabel("Speedup (no_proj / proj)", fontsize=12)
        ax2.set_title(
            "Projection Speedup (>1 = projection faster)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax2.legend(fontsize=8, ncol=2)

    # Plot 3: Bergson vs Dattri (with projection)
    ax3 = axes[1, 0]
    proj_df = df[df["projection"] == "with"]
    for model_key in proj_df["model_key"].unique():
        for method in ["bergson", "dattri"]:
            subset = proj_df[
                (proj_df["method"] == method) & (proj_df["model_key"] == model_key)
            ]
            if not subset.empty:
                subset = subset.sort_values("train_tokens")
                style = styles[(method, "with")]
                ax3.plot(
                    subset["train_tokens"],
                    subset["runtime_seconds"],
                    marker=style["marker"],
                    color=style["color"],
                    label=f"{method} ({model_key})",
                    linewidth=1.5,
                    markersize=6,
                    alpha=0.8,
                )

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Training Tokens", fontsize=12)
    ax3.set_ylabel("Runtime (seconds)", fontsize=12)
    ax3.set_title("With Projection: Bergson vs Dattri", fontsize=14, fontweight="bold")
    ax3.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax3.legend(fontsize=8, ncol=2)

    # Plot 4: Bergson vs Dattri (without projection)
    ax4 = axes[1, 1]
    noproj_df = df[df["projection"] == "without"]
    for model_key in noproj_df["model_key"].unique():
        for method in ["bergson", "dattri"]:
            subset = noproj_df[
                (noproj_df["method"] == method) & (noproj_df["model_key"] == model_key)
            ]
            if not subset.empty:
                subset = subset.sort_values("train_tokens")
                style = styles[(method, "without")]
                ax4.plot(
                    subset["train_tokens"],
                    subset["runtime_seconds"],
                    marker=style["marker"],
                    color=style["color"],
                    label=f"{method} ({model_key})",
                    linewidth=1.5,
                    markersize=6,
                    alpha=0.8,
                )

    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("Training Tokens", fontsize=12)
    ax4.set_ylabel("Runtime (seconds)", fontsize=12)
    ax4.set_title(
        "Without Projection: Bergson vs Dattri", fontsize=14, fontweight="bold"
    )
    ax4.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax4.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved projection comparison plot to {output_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate projection comparison plots",
    )
    parser.add_argument(
        "--bergson_proj_root",
        default="runs/proj_comparison/bergson_proj",
        help="Root directory for Bergson with projection results",
    )
    parser.add_argument(
        "--bergson_noproj_root",
        default="runs/proj_comparison/bergson_noproj",
        help="Root directory for Bergson without projection results",
    )
    parser.add_argument(
        "--dattri_proj_root",
        default="runs/proj_comparison/dattri_proj",
        help="Root directory for Dattri with projection results",
    )
    parser.add_argument(
        "--dattri_noproj_root",
        default="runs/proj_comparison/dattri_noproj",
        help="Root directory for Dattri without projection results",
    )
    parser.add_argument(
        "--output_csv",
        default="runs/benchmarks/projection_comparison.csv",
        help="Path to save CSV data",
    )
    parser.add_argument(
        "--output_plot",
        default="figures/projection_comparison.png",
        help="Path to save plot",
    )

    args = parser.parse_args(argv)

    # Load records from all sources
    bergson_proj = (
        load_inmem_records(Path(args.bergson_proj_root))
        if Path(args.bergson_proj_root).exists()
        else []
    )
    bergson_noproj = (
        load_inmem_records(Path(args.bergson_noproj_root))
        if Path(args.bergson_noproj_root).exists()
        else []
    )
    dattri_proj = (
        load_dattri_records(Path(args.dattri_proj_root))
        if Path(args.dattri_proj_root).exists()
        else []
    )
    dattri_noproj = (
        load_dattri_records(Path(args.dattri_noproj_root))
        if Path(args.dattri_noproj_root).exists()
        else []
    )

    print(f"Bergson with projection: {len(bergson_proj)} records")
    print(f"Bergson without projection: {len(bergson_noproj)} records")
    print(f"Dattri with projection: {len(dattri_proj)} records")
    print(f"Dattri without projection: {len(dattri_noproj)} records")

    total = (
        len(bergson_proj) + len(bergson_noproj) + len(dattri_proj) + len(dattri_noproj)
    )
    if total == 0:
        print("No benchmark records found")
        return

    # Create combined dataframe
    df = create_dataframe(bergson_proj, bergson_noproj, dattri_proj, dattri_noproj)

    if df.empty:
        print("No successful benchmark runs found")
        return

    print(f"\nTotal: {len(df)} successful benchmark runs")

    # Show summary
    print("\nData summary:")
    for method in df["method"].unique():
        for proj in df["projection"].unique():
            subset = df[(df["method"] == method) & (df["projection"] == proj)]
            if not subset.empty:
                models = sorted(subset["model_key"].unique())
                tokens = sorted(subset["train_tokens"].unique())
                print(f"  {method} ({proj} proj): {len(subset)} runs")
                print(f"    Models: {models}")
                print(f"    Token scales: {[format_tokens(t) for t in tokens]}")

    # Save CSV
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    # Create plot
    plot_path = Path(args.output_plot)
    plot_projection_comparison(df, plot_path)


if __name__ == "__main__":
    main()
