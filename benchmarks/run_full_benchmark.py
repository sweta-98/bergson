"""Coordinate running dattri and bergson benchmarks and generate comparison plots."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from matplotlib import pyplot as plt
from simple_parsing import ArgumentParser, field

# Import from same directory
from benchmarks.benchmark_bergson import RunRecord as BergsonRecord
from benchmarks.benchmark_bergson import load_records as load_bergson_records
from benchmarks.benchmark_dattri import RunRecord as DattriRecord
from benchmarks.benchmark_dattri import load_records as load_dattri_records
from benchmarks.benchmark_utils import MODEL_SPECS, format_tokens, parse_tokens


def run_benchmark(
    method: str,
    model: str,
    train_tokens: int,
    eval_tokens: int,
    run_root: str,
    num_gpus: int = 1,
    **kwargs: Any,
) -> bool:
    """Run a single benchmark."""
    if method == "dattri":
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.benchmark_dattri",
            "run",
            model,
            format_tokens(train_tokens),
            format_tokens(eval_tokens),
            "--run_root",
            run_root,
            "--num_gpus",
            str(num_gpus),
        ]
        if "batch_size" in kwargs:
            cmd.extend(["--batch_size", str(kwargs["batch_size"])])
        if "max_length" in kwargs:
            cmd.extend(["--max_length", str(kwargs["max_length"])])
    elif method == "bergson":
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.benchmark_bergson",
            "run",
            model,
            format_tokens(train_tokens),
            format_tokens(eval_tokens),
            "--run_root",
            run_root,
            "--num_gpus",
            str(num_gpus),
        ]
        if "max_eval_examples" in kwargs:
            cmd.extend(["--max_eval_examples", str(kwargs["max_eval_examples"])])
        if "batch_size" in kwargs:
            cmd.extend(["--batch_size", str(kwargs["batch_size"])])
        if "max_length" in kwargs:
            cmd.extend(["--max_length", str(kwargs["max_length"])])
        # Enable FSDP for larger models (>= 1B parameters)
        if model in MODEL_SPECS and MODEL_SPECS[model].params >= 1_000_000_000:
            cmd.append("--fsdp")
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running {method} benchmark:")
        print(result.stderr)
        return False

    print(f"Successfully ran {method} benchmark")
    return True


def load_all_records(
    dattri_root: Path,
    bergson_root: Path,
) -> tuple[list[DattriRecord], list[BergsonRecord]]:
    """Load all benchmark records."""
    dattri_records = load_dattri_records(dattri_root) if dattri_root.exists() else []
    bergson_records = (
        load_bergson_records(bergson_root) if bergson_root.exists() else []
    )
    return dattri_records, bergson_records


def create_comparison_dataframe(
    dattri_records: list[DattriRecord],
    bergson_records: list[BergsonRecord],
) -> pd.DataFrame:
    """Create a combined dataframe for comparison."""
    rows = []

    # Add dattri records
    for r in dattri_records:
        if r.status == "success" and r.runtime_seconds is not None:
            # Handle records that may not have num_gpus (backwards compatibility)
            num_gpus = getattr(r, "num_gpus", 1)
            rows.append(
                {
                    "method": "dattri",
                    "model_key": r.model_key,
                    "model_params": r.params,
                    "train_tokens": r.train_tokens,
                    "eval_tokens": r.eval_tokens,
                    "num_gpus": num_gpus,
                    "runtime_seconds": r.runtime_seconds,
                    "reduce_seconds": None,  # Dattri doesn't separate reduce/score
                    "score_seconds": None,
                }
            )

    # Add bergson records
    for r in bergson_records:
        if r.status == "success":
            # Calculate total runtime for in-memory bergson (query + build + score)
            total_runtime = None
            if all(
                x is not None
                for x in [r.query_seconds, r.build_seconds, r.score_seconds]
            ):
                total_runtime = (
                    (r.query_seconds or 0)
                    + (r.build_seconds or 0)
                    + (r.score_seconds or 0)
                )

            if total_runtime is not None:
                # Handle records that may not have num_gpus (backwards compatibility)
                num_gpus = getattr(r, "num_gpus", 1)
                rows.append(
                    {
                        "method": "bergson-inmem",
                        "model_key": r.model_key,
                        "model_params": r.params,
                        "train_tokens": r.train_tokens,
                        "eval_tokens": r.eval_tokens,
                        "num_gpus": num_gpus,
                        "runtime_seconds": total_runtime,
                        "reduce_seconds": None,  # No separate reduce step in in-memory
                        "score_seconds": r.score_seconds,
                    }
                )

    return pd.DataFrame(rows)


def plot_comparison(
    df: pd.DataFrame, output_path: Path, title_suffix: str = ""
) -> None:
    """Create comparison plots."""
    if df.empty:
        print("No data to plot")
        return

    # Filter successful runs
    df = df[df["runtime_seconds"].notna()].copy()

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Runtime vs train tokens (by model)
    ax1 = axes[0, 0]
    for method in df["method"].unique():
        for model_key in df["model_key"].unique():
            subset = df[(df["method"] == method) & (df["model_key"] == model_key)]
            if not subset.empty:
                subset = subset.sort_values("train_tokens")
                ax1.plot(
                    subset["train_tokens"],
                    subset["runtime_seconds"],
                    marker="o",
                    label=f"{method}-{model_key}",
                    linewidth=1.5,
                )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Training Tokens")
    ax1.set_ylabel("Total Runtime (seconds)")
    ax1.set_title(f"Runtime Scaling: Total Time{title_suffix}")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.legend(fontsize="small", ncol=2)

    # Plot 2: Runtime vs model params (by token scale)
    ax2 = axes[0, 1]
    for method in df["method"].unique():
        for train_tokens in sorted(df["train_tokens"].unique())[
            :5
        ]:  # Top 5 token scales
            subset = df[(df["method"] == method) & (df["train_tokens"] == train_tokens)]
            if not subset.empty:
                subset = subset.sort_values("model_params")
                ax2.plot(
                    subset["model_params"],
                    subset["runtime_seconds"],
                    marker="o",
                    label=f"{method}-{format_tokens(train_tokens)}",
                    linewidth=1.5,
                )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Model Parameters")
    ax2.set_ylabel("Total Runtime (seconds)")
    ax2.set_title(f"Runtime Scaling: Model Size{title_suffix}")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.legend(fontsize="small", ncol=2)

    # Plot 3: Bergson reduce vs score breakdown
    ax3 = axes[1, 0]
    bergson_df = df[df["method"] == "bergson"]
    if not bergson_df.empty and bergson_df["reduce_seconds"].notna().any():
        for model_key in bergson_df["model_key"].unique():
            subset = bergson_df[bergson_df["model_key"] == model_key].sort_values(
                "train_tokens"
            )
            if subset["score_seconds"].notna().any():
                ax3.plot(
                    subset["train_tokens"],
                    subset["score_seconds"],
                    marker="^",
                    label=f"{model_key} (score)",
                    linewidth=1.5,
                    linestyle="--",
                )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Training Tokens")
    ax3.set_ylabel("Runtime (seconds)")
    ax3.set_title(f"Bergson: Score Breakdown{title_suffix}")
    ax3.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax3.legend(fontsize="small")

    # Plot 4: Speedup comparison (dattri / bergson)
    ax4 = axes[1, 1]
    speedup_data = []
    for model_key in df["model_key"].unique():
        for train_tokens in df["train_tokens"].unique():
            dattri_subset = df[
                (df["method"] == "dattri")
                & (df["model_key"] == model_key)
                & (df["train_tokens"] == train_tokens)
            ]
            bergson_subset = df[
                (df["method"] == "bergson")
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
                        }
                    )

    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        for model_key in speedup_df["model_key"].unique():
            subset = speedup_df[speedup_df["model_key"] == model_key].sort_values(
                "train_tokens"
            )
            ax4.plot(
                subset["train_tokens"],
                subset["speedup"],
                marker="o",
                label=model_key,
                linewidth=1.5,
            )
        ax4.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax4.set_xscale("log")
        ax4.set_xlabel("Training Tokens")
        ax4.set_ylabel("Speedup (dattri / bergson)")
        ax4.set_title(f"Relative Performance: Dattri vs Bergson{title_suffix}")
        ax4.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax4.legend(fontsize="small")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved comparison plot to {output_path}")


@dataclass
class RunBenchmarkConfig:
    """Configuration for running benchmarks."""

    run_root: str = field(positional=True)
    """Root directory for benchmark runs."""

    models: list[str] | None = None
    """Models to benchmark (default: pythia-14m, pythia-70m)."""

    token_scales: list[str] = field(default_factory=lambda: ["1M", "2M", "5M", "10M"])
    """Token scales to test (e.g. 1M 10M)."""

    eval_tokens: str = "100K"
    """Evaluation tokens."""

    batch_size: int = 4
    """Batch size."""

    max_length: int = 512
    """Maximum sequence length."""

    num_gpus: int = 1
    """Number of GPUs to use."""

    num_test: int = 10
    """Number of test examples for bergson."""

    skip_dattri: bool = False
    """Skip dattri benchmarks."""

    skip_bergson: bool = False
    """Skip bergson benchmarks."""

    force: bool = False
    """Re-run existing benchmarks."""


@dataclass
class BenchmarkRun:
    """Run benchmarks for specified models and token scales."""

    run_cfg: RunBenchmarkConfig

    def execute(self) -> None:
        """Run benchmarks for specified models and token scales."""
        models = self.run_cfg.models or ["pythia-14m", "pythia-70m"]
        token_scales = [parse_tokens(ts) for ts in self.run_cfg.token_scales]
        eval_tokens = parse_tokens(self.run_cfg.eval_tokens)
        num_gpus = self.run_cfg.num_gpus

        dattri_root = Path(self.run_cfg.run_root) / "dattri-scaling"
        bergson_root = Path(self.run_cfg.run_root) / "bergson-scaling"

        # Check existing runs
        dattri_records, bergson_records = load_all_records(dattri_root, bergson_root)
        existing_dattri = {
            (r.model_key, r.train_tokens, r.eval_tokens, getattr(r, "num_gpus", 1))
            for r in dattri_records
            if r.status == "success"
        }
        existing_bergson = {
            (r.model_key, r.train_tokens, r.eval_tokens, getattr(r, "num_gpus", 1))
            for r in bergson_records
            if r.status == "success"
        }

        # Run benchmarks
        for model in models:
            if model not in MODEL_SPECS:
                print(f"Warning: Unknown model {model}, skipping")
                continue

            for train_tokens in token_scales:
                # Run dattri
                if not self.run_cfg.skip_dattri:
                    key = (model, train_tokens, eval_tokens, num_gpus)
                    if key not in existing_dattri or self.run_cfg.force:
                        print(f"\n{'='*60}")
                        print(
                            f"Running Dattri: {model}, {format_tokens(train_tokens)} "
                            f"train tokens, {num_gpus} GPU(s)"
                        )
                        print(f"{'='*60}")
                        success = run_benchmark(
                            "dattri",
                            model,
                            train_tokens,
                            eval_tokens,
                            str(dattri_root),
                            num_gpus=num_gpus,
                            batch_size=self.run_cfg.batch_size,
                            max_length=self.run_cfg.max_length,
                        )
                        if not success:
                            print(
                                f"Failed to run dattri benchmark for {model} "
                                f"{format_tokens(train_tokens)}"
                            )
                    else:
                        print(
                            f"Skipping dattri {model} {format_tokens(train_tokens)} "
                            f"{num_gpus}gpu (already exists)"
                        )

                # Run bergson
                if not self.run_cfg.skip_bergson:
                    key = (model, train_tokens, eval_tokens, num_gpus)
                    if key not in existing_bergson or self.run_cfg.force:
                        print(f"\n{'='*60}")
                        print(
                            f"Running Bergson: {model}, {format_tokens(train_tokens)} "
                            f"train tokens, {num_gpus} GPU(s)"
                        )
                        print(f"{'='*60}")
                        success = run_benchmark(
                            "bergson",
                            model,
                            train_tokens,
                            eval_tokens,
                            str(bergson_root),
                            num_gpus=num_gpus,
                            batch_size=self.run_cfg.batch_size,
                            max_length=self.run_cfg.max_length,
                            max_eval_examples=self.run_cfg.num_test,
                        )
                        if not success:
                            print(
                                f"Failed to run bergson benchmark for {model} "
                                f"{format_tokens(train_tokens)}"
                            )
                    else:
                        print(
                            f"Skipping bergson {model} {format_tokens(train_tokens)} "
                            f"{num_gpus}gpu (already exists)"
                        )


@dataclass
class PlotConfig:
    """Configuration for generating comparison plots."""

    run_root: str = field(positional=True)
    """Root directory for benchmark runs."""

    output_csv: str = field(positional=True)
    """Path to save comparison data CSV."""

    plot_output: str = field(positional=True)
    """Path to save comparison plots."""

    num_gpus: int | None = None
    """Filter plots to specific GPU count (default: generate all)."""

    skip_plots: bool = False
    """Skip generating plots."""


@dataclass
class Plot:
    """Generate comparison plots from existing benchmark results."""

    plot_cfg: PlotConfig

    def execute(self) -> None:
        """Generate comparison plots from existing benchmark results."""
        dattri_root = Path(self.plot_cfg.run_root) / "dattri-scaling"
        bergson_root = Path(self.plot_cfg.run_root) / "bergson_inmem_benchmark"

        dattri_records, bergson_records = load_all_records(dattri_root, bergson_root)

        df = create_comparison_dataframe(dattri_records, bergson_records)

        # Save CSV
        csv_path = Path(self.plot_cfg.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved comparison data to {csv_path}")

        # Create plots
        if not self.plot_cfg.skip_plots:
            plot_dir = Path(self.plot_cfg.plot_output).parent
            plot_stem = Path(self.plot_cfg.plot_output).stem
            plot_ext = Path(self.plot_cfg.plot_output).suffix or ".png"

            # Get unique GPU counts in the data
            gpu_counts = (
                sorted(df["num_gpus"].unique()) if "num_gpus" in df.columns else [1]
            )

            if self.plot_cfg.num_gpus is not None:
                # Filter to specific GPU count
                gpu_counts = [self.plot_cfg.num_gpus]

            for num_gpus in gpu_counts:
                if "num_gpus" in df.columns:
                    gpu_df = df[df["num_gpus"] == num_gpus]
                else:
                    gpu_df = df

                if gpu_df.empty:
                    print(f"No data for {num_gpus} GPU(s), skipping")
                    continue

                plot_path = plot_dir / f"{plot_stem}_{num_gpus}gpu{plot_ext}"
                title_suffix = f" ({num_gpus} GPU{'s' if num_gpus > 1 else ''})"
                plot_comparison(gpu_df, plot_path, title_suffix)


@dataclass
class Main:
    """Coordinate dattri and bergson benchmarks."""

    command: BenchmarkRun | Plot

    def execute(self) -> None:
        """Run the selected command."""
        self.command.execute()


def get_parser() -> ArgumentParser:
    """Get the argument parser. Used for documentation generation."""
    parser = ArgumentParser(description="Coordinate dattri and bergson benchmarks")
    parser.add_arguments(Main, dest="prog")
    return parser


def main(args: Optional[list[str]] = None) -> None:
    """Parse CLI arguments and dispatch to the selected subcommand."""
    parser = get_parser()
    prog: Main = parser.parse_args(args=args).prog
    prog.execute()


if __name__ == "__main__":
    main()
