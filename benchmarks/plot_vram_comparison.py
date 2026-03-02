"""Generate a combined VRAM scaling plot across bergson,
dattri, and kronfluence.

Loads per-tool VRAM benchmark CSVs and produces a grouped bar
chart of peak VRAM by model size, one bar per tool.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from benchmarks.benchmark_utils import extract_gpu_info

# Colors per tool (same as plot_factor_benchmark)
TOOL_COLORS: dict[str, str] = {
    "bergson": "#1f77b4",
    "kronfluence": "#ff7f0e",
    "dattri": "#2ca02c",
}

# Model display order (ascending params)
MODEL_ORDER = [
    "pythia-14m",
    "pythia-70m",
    "pythia-160m",
    "pythia-1b",
    "pythia-6.9b",
    "pythia-12b",
]


def _load_bergson(path: Path) -> pd.DataFrame:
    """Load bergson in-memory VRAM CSV."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Use score_peak_vram_mb as the representative VRAM
    # (highest phase for bergson)
    vram_col = "score_peak_vram_mb"
    if vram_col not in df.columns:
        return pd.DataFrame()
    # Take max VRAM across data scales per model
    grouped = (
        df.groupby("model_key")[vram_col]
        .max()
        .reset_index()
    )
    grouped = grouped.rename(
        columns={vram_col: "peak_vram_mb"}
    )
    grouped["tool"] = "bergson"
    return grouped[["model_key", "peak_vram_mb", "tool"]]


def _load_dattri(path: Path) -> pd.DataFrame:
    """Load dattri VRAM CSV."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "peak_vram_mb" not in df.columns:
        return pd.DataFrame()
    grouped = (
        df.groupby("model_key")["peak_vram_mb"]
        .max()
        .reset_index()
    )
    grouped["tool"] = "dattri"
    return grouped[["model_key", "peak_vram_mb", "tool"]]


def _load_kronfluence(path: Path) -> pd.DataFrame:
    """Load kronfluence VRAM CSV."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "peak_vram_mb" not in df.columns:
        return pd.DataFrame()
    grouped = (
        df.groupby("model_key")["peak_vram_mb"]
        .max()
        .reset_index()
    )
    grouped["tool"] = "kronfluence"
    return grouped[["model_key", "peak_vram_mb", "tool"]]


def _hw_label(dfs: list[pd.DataFrame]) -> str:
    """Extract GPU label from any loaded dataframe."""
    for df in dfs:
        if "hardware" in df.columns:
            hw = df["hardware"].dropna()
            if not hw.empty:
                info = extract_gpu_info(hw.iloc[0])
                if info:
                    return info
    return ""


def plot_vram_comparison(
    df: pd.DataFrame,
    figure_path: Path,
    suptitle: str,
    formats: list[str] | None = None,
) -> None:
    """Create grouped bar chart: peak VRAM by model,
    one bar per tool."""
    if formats is None:
        formats = ["png"]
    if df.empty:
        print("No data to plot", file=sys.stderr)
        return

    # Order models by param count
    present = [
        m for m in MODEL_ORDER
        if m in df["model_key"].values
    ]
    tools = [
        t for t in TOOL_COLORS
        if t in df["tool"].values
    ]

    n_models = len(present)
    n_tools = len(tools)
    bar_width = 0.7 / max(n_tools, 1)

    fig, ax = plt.subplots(figsize=(max(7, n_models * 2), 5))

    for ti, tool in enumerate(tools):
        tool_df = df[df["tool"] == tool]
        vals = []
        for model in present:
            row = tool_df[tool_df["model_key"] == model]
            if not row.empty:
                vals.append(
                    row["peak_vram_mb"].iloc[0] / 1024
                )
            else:
                vals.append(0)

        offsets = [
            x + ti * bar_width for x in range(n_models)
        ]
        ax.bar(
            offsets,
            vals,
            bar_width,
            label=tool,
            color=TOOL_COLORS[tool],
            edgecolor="black",
            linewidth=0.5,
        )
        # Value labels
        for offset, val in zip(offsets, vals):
            if val > 0:
                ax.text(
                    offset,
                    val,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    center = [
        x + bar_width * (n_tools - 1) / 2
        for x in range(n_models)
    ]
    ax.set_xticks(center)
    ax.set_xticklabels(present, fontsize=10)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Peak VRAM (GB)", fontsize=11)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.15)
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.6,
    )
    ax.legend(fontsize=10, loc="upper left")

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = figure_path.with_suffix(f".{fmt}")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved VRAM comparison plot to {out}")
    plt.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot combined VRAM scaling across bergson,"
            " dattri, and kronfluence."
        ),
    )
    parser.add_argument(
        "--bergson_csv",
        default="runs/benchmarks/"
        "inmem_vram_benchmark_1gpu.csv",
        help="Bergson in-memory VRAM benchmark CSV.",
    )
    parser.add_argument(
        "--dattri_csv",
        default="runs/benchmarks/"
        "dattri_vram_benchmark_1gpu.csv",
        help="Dattri VRAM benchmark CSV.",
    )
    parser.add_argument(
        "--kronfluence_csv",
        default="runs/benchmarks/"
        "kronfluence_vram_benchmark_1gpu.csv",
        help="Kronfluence VRAM benchmark CSV.",
    )
    parser.add_argument(
        "--output",
        default="runs/benchmarks/vram_comparison_1gpu",
        help="Output path (extension added per format).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Output formats (default: png).",
    )

    args = parser.parse_args(argv)

    bergson_csv = Path(args.bergson_csv)
    dattri_csv = Path(args.dattri_csv)
    kronfluence_csv = Path(args.kronfluence_csv)

    parts = []
    raw_dfs = []
    for loader, path in [
        (_load_bergson, bergson_csv),
        (_load_dattri, dattri_csv),
        (_load_kronfluence, kronfluence_csv),
    ]:
        loaded = loader(path)
        if not loaded.empty:
            parts.append(loaded)
            raw_dfs.append(pd.read_csv(path))
            print(
                f"Loaded {len(loaded)} models"
                f" from {path}"
            )
        else:
            print(f"No data from {path}", file=sys.stderr)

    if not parts:
        print("No data to plot.", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(parts, ignore_index=True)
    hw = _hw_label(raw_dfs)
    title = "Peak VRAM by Model Size"
    if hw:
        title += f" ({hw})"

    output = Path(args.output)
    plot_vram_comparison(
        df, output, title, formats=args.formats
    )


if __name__ == "__main__":
    main()
