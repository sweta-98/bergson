"""Generate a combined 1-GPU vs 8-GPU plot from CLI benchmarks.

Produces a single 1x2 figure where each subplot shows build and
score runtime by training tokens, broken down by model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from benchmarks.benchmark_bergson_cli import (
    CLIRunRecord,
    load_records,
)
from benchmarks.benchmark_utils import extract_gpu_info


def create_cli_dataframe(
    records: list[CLIRunRecord],
) -> pd.DataFrame:
    """Create a dataframe from CLI benchmark records."""
    rows = []
    for r in records:
        if r.status == "success" and r.total_runtime_seconds is not None:
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
                    "reduce_seconds": r.reduce_seconds,
                    "score_seconds": r.score_seconds,
                    "total_runtime_seconds": (r.total_runtime_seconds),
                    "run_path": r.run_path,
                    "num_gpus": r.num_gpus,
                    "hardware": r.hardware,
                }
            )
    return pd.DataFrame(rows)


def _plot_build_score(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
) -> None:
    """Plot build and score runtime on a single axes."""
    for model_key in sorted(df["model_key"].unique()):
        subset = df[df["model_key"] == model_key]
        subset = subset.sort_values("train_tokens")
        if subset["build_seconds"].notna().any():
            ax.plot(
                subset["train_tokens"],
                subset["build_seconds"],
                marker="s",
                label=f"{model_key} (build)",
                linewidth=2,
                linestyle="-",
            )
        if subset["score_seconds"].notna().any():
            ax.plot(
                subset["train_tokens"],
                subset["score_seconds"],
                marker="D",
                label=f"{model_key} (score)",
                linewidth=2,
                linestyle=":",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training Tokens", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
        alpha=0.6,
    )
    ax.legend(fontsize=8, ncol=2)


def plot_cli_benchmark(
    df_1gpu: pd.DataFrame | None,
    df_8gpu: pd.DataFrame | None,
    figure_path: Path,
    hardware_str: str,
) -> None:
    """Create a combined 1x2 plot (1-GPU left, 8-GPU right)."""
    panels: list[tuple[pd.DataFrame, str]] = []
    if df_1gpu is not None and not df_1gpu.empty:
        panels.append((df_1gpu, "1 GPU: Build vs Score"))
    if df_8gpu is not None and not df_8gpu.empty:
        panels.append((df_8gpu, "8 GPU: Build vs Score"))

    if not panels:
        print("No data to plot")
        return

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    fig.suptitle(
        f"Bergson CLI Benchmark ({hardware_str})",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    for ax, (df, title) in zip(axes, panels):
        _plot_build_score(ax, df, title)

    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved CLI benchmark plot to {figure_path}")


def _gpu_name_from_df(
    df: pd.DataFrame | None,
) -> str:
    """Extract GPU name (without Nx prefix) from a df."""
    if df is None or df.empty:
        return "unknown"
    hw_col = df["hardware"].dropna()
    if hw_col.empty:
        return "unknown"
    gpu_info = extract_gpu_info(hw_col.iloc[0])
    if gpu_info:
        parts = gpu_info.split(" ", 1)
        if len(parts) == 2 and parts[0].endswith("x"):
            return parts[1]
        return gpu_info
    return "unknown"


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a dataframe."""
    if not path.exists():
        print(
            f"CSV not found: {path}",
            file=sys.stderr,
        )
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def _load_run_root(run_root: Path, num_gpus: int | None = None) -> pd.DataFrame:
    """Load records from a run root directory."""
    if not run_root.exists():
        print(
            f"Run root not found: {run_root}",
            file=sys.stderr,
        )
        return pd.DataFrame()
    records = load_records(run_root)
    if not records:
        print(f"No records in {run_root}", file=sys.stderr)
        return pd.DataFrame()
    df = create_cli_dataframe(records)
    if num_gpus is not None:
        df = df[df["num_gpus"] == num_gpus]
    print(f"Loaded {len(df)} runs from {run_root}")
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=("Generate combined 1-GPU vs 8-GPU CLI " "benchmark plot"),
    )
    parser.add_argument(
        "--csv_1gpu",
        default=None,
        help="CSV file with 1-GPU benchmark data",
    )
    parser.add_argument(
        "--csv_8gpu",
        default=None,
        help="CSV file with 8-GPU benchmark data",
    )
    parser.add_argument(
        "--run_root_1gpu",
        default=None,
        help=("Run root directory for 1-GPU data " "(alternative to --csv_1gpu)"),
    )
    parser.add_argument(
        "--run_root_8gpu",
        default=None,
        help=("Run root directory for 8-GPU data " "(alternative to --csv_8gpu)"),
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Directory to save figure and CSVs",
    )

    args = parser.parse_args(argv)
    output_path = Path(args.output_path)

    # Load 1-GPU data
    df_1gpu: pd.DataFrame | None = None
    if args.csv_1gpu:
        df_1gpu = _load_csv(Path(args.csv_1gpu))
    elif args.run_root_1gpu:
        df_1gpu = _load_run_root(Path(args.run_root_1gpu), num_gpus=1)

    # Load 8-GPU data
    df_8gpu: pd.DataFrame | None = None
    if args.csv_8gpu:
        df_8gpu = _load_csv(Path(args.csv_8gpu))
    elif args.run_root_8gpu:
        df_8gpu = _load_run_root(Path(args.run_root_8gpu), num_gpus=8)

    if (df_1gpu is None or df_1gpu.empty) and (df_8gpu is None or df_8gpu.empty):
        print(
            "No data loaded. Provide --csv_1gpu/--csv_8gpu "
            "or --run_root_1gpu/--run_root_8gpu.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Derive hardware per dataset
    hw_1gpu = _gpu_name_from_df(df_1gpu)
    hw_8gpu = _gpu_name_from_df(df_8gpu)

    # Save CSVs with per-dataset hardware names
    if df_1gpu is not None and not df_1gpu.empty:
        label = hw_1gpu.replace(" ", "_")
        csv_1 = output_path / f"cli_benchmark_1x_{label}.csv"
        df_1gpu.to_csv(csv_1, index=False)
        print(f"Saved 1-GPU CSV to {csv_1}")

    if df_8gpu is not None and not df_8gpu.empty:
        label = hw_8gpu.replace(" ", "_")
        csv_8 = output_path / f"cli_benchmark_8x_{label}.csv"
        df_8gpu.to_csv(csv_8, index=False)
        print(f"Saved 8-GPU CSV to {csv_8}")

    # Figure title: combine hardware names if different
    hw_names = sorted({n for n in (hw_1gpu, hw_8gpu) if n != "unknown"})
    title_hw = " & ".join(hw_names) if hw_names else "unknown"
    file_hw = "_".join(hw_names).replace(" ", "_")

    figure_path = output_path / f"cli_benchmark_{file_hw}.png"
    plot_cli_benchmark(df_1gpu, df_8gpu, figure_path, title_hw)


if __name__ == "__main__":
    main()
