"""Generate grouped bar charts comparing factor computation
times and peak GPU memory.

Loads benchmark records from a run root directory and produces
a 1x2 panel: time on the left, peak VRAM on the right.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from benchmarks.benchmark_factors import RunRecord, load_records
from benchmarks.benchmark_utils import (
    extract_gpu_info,
    format_tokens,
)

# Colors per method
METHOD_COLORS: dict[str, str] = {
    "bergson": "#1f77b4",
    "kronfluence": "#ff7f0e",
    "dattri": "#2ca02c",
}

# Hatching per factor type for visual distinction within a method
FACTOR_HATCHES: dict[str, str] = {
    "normalizer": "",
    "preconditioner": "//",
    "kfac": "xx",
    "diagonal": "",
    "ekfac": "xx",
    "datainf": "",
    "arnoldi": "//",
}


def create_factor_dataframe(
    records: list[RunRecord],
) -> pd.DataFrame:
    """Create a dataframe from factor benchmark records."""
    rows = []
    for r in records:
        if r.status == "success" and r.factor_seconds is not None:
            rows.append(
                {
                    "model_key": r.model_key,
                    "model_name": r.model_name,
                    "params": r.params,
                    "train_tokens": r.train_tokens,
                    "method": r.method,
                    "factor_type": r.factor_type,
                    "factor_seconds": r.factor_seconds,
                    "peak_memory_mb": getattr(r, "peak_memory_mb", None),
                    "label": f"{r.method}/{r.factor_type}",
                    "hardware": getattr(r, "hardware", None),
                }
            )
    return pd.DataFrame(rows)


def _draw_bars(
    ax: Axes,
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    value_fmt: str,
    groups: list[tuple],
    labels: list[str],
) -> None:
    """Draw grouped bars on *ax* for *value_col*."""
    n_groups = len(groups)
    n_bars = len(labels)
    bar_width = 0.7 / max(n_groups, 1)
    x_positions = range(n_bars)

    for gi, (model_key, train_tokens) in enumerate(groups):
        subset = df[
            (df["model_key"] == model_key) & (df["train_tokens"] == train_tokens)
        ]
        values = []
        colors = []
        hatches = []
        for label in labels:
            row = subset[subset["label"] == label]
            if not row.empty and pd.notna(row[value_col].iloc[0]):
                values.append(row[value_col].iloc[0])
            else:
                values.append(0)
            method = label.split("/")[0]
            factor = label.split("/")[1]
            colors.append(METHOD_COLORS.get(method, "#999999"))
            hatches.append(FACTOR_HATCHES.get(factor, ""))

        offsets = [x + gi * bar_width for x in x_positions]
        group_label = f"{model_key} ({format_tokens(train_tokens)})"
        bars = ax.bar(
            offsets,
            values,
            bar_width,
            label=group_label,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        # Value labels on top of bars
        for offset, val in zip(offsets, values):
            if val > 0:
                ax.text(
                    offset,
                    val,
                    value_fmt.format(val),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    # X-axis labels
    center_offsets = [x + bar_width * (n_groups - 1) / 2 for x in x_positions]
    ax.set_xticks(center_offsets)
    short_labels = [lb.replace("/", "\n") for lb in labels]
    ax.set_xticklabels(short_labels, rotation=0, ha="center", fontsize=9)

    ax.set_ylabel(ylabel, fontsize=11)
    # Add headroom so value labels aren't clipped
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.20)
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.6,
    )

    # Legend: method colors
    legend_handles = []
    for method, color in METHOD_COLORS.items():
        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor="black",
                label=method,
            )
        )
    ax.legend(
        handles=legend_handles,
        fontsize=9,
        loc="upper left",
    )


def plot_factor_comparison(
    df: pd.DataFrame,
    figure_path: Path,
    suptitle: str,
    formats: list[str] | None = None,
) -> None:
    """Create a 1x2 grouped bar chart: time (left) and
    peak VRAM (right).

    Groups by model, one bar per method/factor_type. If
    multiple runs exist for the same combination, uses the
    latest (last) one.
    """
    if formats is None:
        formats = ["png"]

    if df.empty:
        print("No data to plot", file=sys.stderr)
        return

    # Deduplicate: keep last run per
    # (model, train_tokens, method, factor_type)
    df = df.drop_duplicates(
        subset=[
            "model_key",
            "train_tokens",
            "method",
            "factor_type",
        ],
        keep="last",
    )

    groups = sorted(
        df.groupby(["model_key", "train_tokens"]).groups.keys(),
        key=lambda x: (x[1], x[0]),
    )
    labels = sorted(df["label"].unique())

    n_groups = len(groups)
    n_bars = len(labels)
    if n_groups == 0 or n_bars == 0:
        print("No data to plot", file=sys.stderr)
        return

    has_memory = "peak_memory_mb" in df.columns and df["peak_memory_mb"].notna().any()
    if has_memory:
        df = df.copy()
        df["peak_memory_gb"] = df["peak_memory_mb"] / 1024
    ncols = 2 if has_memory else 1
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(max(8, n_bars * 1.2) * ncols, 6),
    )
    if ncols == 1:
        axes = [axes]

    # Left panel: time
    _draw_bars(
        axes[0],
        df,
        "factor_seconds",
        "Factor Computation Time (seconds)",
        "{:.1f}s",
        groups,
        labels,
    )

    # Right panel: peak VRAM (if data available)
    if has_memory:
        _draw_bars(
            axes[1],
            df,
            "peak_memory_gb",
            "Peak GPU Memory (GB)",
            "{:.1f}",
            groups,
            labels,
        )

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = figure_path.with_suffix(f".{fmt}")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved factor benchmark plot to {out}")
    plt.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=("Plot factor computation benchmark results"),
    )
    parser.add_argument(
        "run_root",
        help="Root directory containing benchmark results",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output path (default:"
            " <run_root>/factor_benchmark.png)."
            " Extension is replaced per format."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help=("Output formats (default: png)." " E.g. --formats png pdf"),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Filter to a single model key (e.g. pythia-160m).",
    )

    args = parser.parse_args(argv)
    run_root = Path(args.run_root)

    if not run_root.exists():
        print(
            f"Run root not found: {run_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    records = load_records(run_root)
    if not records:
        print(
            f"No records found in {run_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    df = create_factor_dataframe(records)
    if args.model:
        df = df[df["model_key"] == args.model]
    print(f"Loaded {len(df)} successful runs from {run_root}")

    if df.empty:
        print(
            "No successful runs to plot",
            file=sys.stderr,
        )
        sys.exit(1)

    # Extract hardware info for title
    hw = df["hardware"].dropna()
    hw_label = ""
    if not hw.empty:
        gpu_info = extract_gpu_info(hw.iloc[0])
        if gpu_info:
            hw_label = f" ({gpu_info})"

    output = Path(args.output) if args.output else run_root / "factor_benchmark.png"
    plot_factor_comparison(
        df,
        output,
        f"Factor Computation Overhead{hw_label}",
        formats=args.formats,
    )


if __name__ == "__main__":
    main()
