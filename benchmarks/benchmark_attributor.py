"""Benchmark Attributor query performance: normal vs TrackStar modes.

Workflow
--------
1. Build a gradient index (with hessians) – timed.
2. Load the model (used to run forward/backward passes inside the trace context).
3. For each attributor mode ("normal" and "trackstar"):
   a. Initialise the Attributor – timed.
   b. For each query-batch size in ``query_counts``:
      - Create a synthetic batch of ``num_queries`` token sequences.
      - Time a full ``attributor.trace(model, k)`` block, which runs a
        forward pass, backward pass, and nearest-neighbour search.
4. Write a JSON record with all timings to ``<run_path>/benchmark_attributor.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path
from typing import Optional

import torch
from simple_parsing import ArgumentParser, ConflictResolution, field
from transformers import PreTrainedModel

from benchmarks.benchmark_utils import (
    MODEL_SPECS,
    get_hardware_info,
    get_run_path,
    parse_tokens,
    prepare_benchmark_ds_path,
    save_record,
    timestamp,
)
from bergson.config import IndexConfig
from bergson.data import load_gradients
from bergson.query.attributor import Attributor
from bergson.utils.worker_utils import setup_model_and_peft

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------


@dataclass
class TraceResult:
    """Timing result for one (mode, num_queries) combination."""

    mode: str
    """'normal' or 'trackstar'."""

    num_queries: int
    """Batch size fed into the trace context (= number of query examples)."""

    trace_seconds: float
    """Wall-clock time of the full trace block (fwd + bwd + search)."""

    

@dataclass
class AttributorBenchmarkRecord:
    """Full record for one attributor benchmark run."""

    schema_version: int
    status: str
    model_key: str
    model_name: str
    params: float
    train_tokens: int
    projection_dim: int
    index_size: int
    """Number of examples in the gradient index."""

    build_seconds: float | None
    """Wall-clock time for the ``bergson build`` step."""

    normal_init_seconds: float | None
    """Wall-clock time to initialise the normal Attributor."""

    trackstar_init_seconds: float | None
    """Wall-clock time to initialise the TrackStar Attributor."""

    trace_results: list[dict]
    """List of :class:`TraceResult` dicts, one per (mode, num_queries)."""

    start_time: str
    end_time: str
    run_path: str
    hardware: str | None
    notes: str | None
    error: str | None
    num_gpus: int = 1


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Configuration for an attributor benchmark run."""

    model: str = field(positional=True)
    """Key for the model to benchmark (e.g., pythia-14m, pythia-70m)."""

    train_tokens: str = field(positional=True)
    """Target training tokens (e.g., 1M, 10M)."""

    run_root: str = field(positional=True)
    """Root directory for benchmark runs."""

    dataset: str = ""
    """Dataset to use (defaults to the standard benchmark dataset)."""

    query_counts: list[int] = dc_field(default_factory=lambda: [1, 4, 16, 64])
    """Batch sizes to benchmark (each value = num query examples in one trace)."""

    seq_len: int = 512
    """Sequence length for the synthetic query inputs."""

    num_gpus: int = 1
    """Number of GPUs used for the build step."""

    tag: str | None = None
    """Optional tag for the run directory."""

    notes: str | None = None
    """Optional free-text notes saved to the record."""

    device: str = "cuda"
    """Device for the Attributor and model (e.g., 'cuda', 'cpu')."""

    precision: str = "fp32"
    """Model precision for query forward passes (e.g., 'bf16', 'fp32')."""

    skip_build: bool = False
    """Re-use an existing index if the run_path already contains one."""

    run_path: str | None = None
    """Override the auto-generated run directory."""

    projection_dim: int = 16
    """Projection dimension used during build."""

    num_trace_repeats: int = 3
    """Number of repeated trace calls per (mode, num_queries); minimum is used."""


# ---------------------------------------------------------------------------
# Build helper
# ---------------------------------------------------------------------------


def _build_index(
    index_path: Path,
    model_hf_id: str,
    dataset: str,
    train_tokens: int,
    num_gpus: int,
    projection_dim: int,
) -> tuple[bool, float, str]:
    """Run ``bergson build`` and return (success, elapsed_seconds, error)."""
    cmd = [
        "bergson",
        "build",
        str(index_path),
        "--model",
        model_hf_id,
        "--dataset",
        dataset,
        "--split",
        "train[:1%]",
        "--overwrite",
        "--truncation",
        "--max_tokens",
        str(train_tokens),
        "--nproc_per_node",
        str(num_gpus),
        "--projection_dim",
        str(projection_dim),
        "--token_batch_size",
        "512",
        # Keep hessians so TrackStar preconditioning has data to work with.
    ]
    print(f"Building index: {' '.join(cmd)}")
    start = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            return False, elapsed, f"build failed (exit {result.returncode})"
        print(f"Build completed in {elapsed:.2f}s")
        return True, elapsed, ""
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return False, elapsed, f"build error: {exc}"


# ---------------------------------------------------------------------------
# Trace timing helper
# ---------------------------------------------------------------------------


def _make_input_ids(
    num_queries: int,
    seq_len: int,
    vocab_size: int,
    device: str,
) -> torch.Tensor:
    """Create a synthetic batch of token sequences."""
    return torch.randint(0, vocab_size, (num_queries, seq_len), device=device)


def _time_trace(
    attributor: Attributor,
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    num_repeats: int,
) -> float:
    """Return the minimum wall-clock time of a full trace block across *num_repeats*.

    Each iteration runs: enter context → forward pass → backward pass → search
    (search is triggered at context exit).
    """
    def _one_trace() -> None:
        with torch.enable_grad():
            with attributor.trace(model.base_model, k=20) as _result:
                out = model(input_ids=input_ids, labels=input_ids)
                out.loss.backward()
                model.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Warm-up (not counted)
    _one_trace()

    best = float("inf")
    for _ in range(num_repeats):
        start = time.perf_counter()
        _one_trace()
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
    return best


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


@dataclass
class Run:
    """Execute a single attributor benchmark run."""

    run_cfg: RunConfig

    def execute(self) -> None:  # noqa: C901
        if not self.run_cfg.dataset:
            self.run_cfg.dataset = str(prepare_benchmark_ds_path())

        if self.run_cfg.model not in MODEL_SPECS:
            raise ValueError(f"Unknown model '{self.run_cfg.model}'")

        spec = MODEL_SPECS[self.run_cfg.model]
        train_tokens = parse_tokens(self.run_cfg.train_tokens)

        run_root = Path(self.run_cfg.run_root).resolve()
        run_root.mkdir(parents=True, exist_ok=True)

        if self.run_cfg.run_path:
            benchmark_path = Path(self.run_cfg.run_path).resolve()
        else:
            benchmark_path = get_run_path(
                run_root,
                spec,
                train_tokens,
                eval_tokens=0,
                eval_sequences=0,
                tag=self.run_cfg.tag,
                num_gpus=self.run_cfg.num_gpus,
            )
        benchmark_path.mkdir(parents=True, exist_ok=True)
        index_path = benchmark_path / "index"

        start_wall = timestamp()
        overall_start = time.perf_counter()
        status = "success"
        error_message: str | None = None
        build_seconds: float | None = None
        normal_init_seconds: float | None = None
        trackstar_init_seconds: float | None = None
        trace_results: list[dict] = []
        index_size: int = 0
        projection_dim: int = self.run_cfg.projection_dim

        try:
            # ------------------------------------------------------------------
            # Step 1: Build gradient index
            # ------------------------------------------------------------------
            if self.run_cfg.skip_build and index_path.exists():
                print(f"Skipping build – reusing existing index at {index_path}")
            else:
                ok, build_seconds, err = _build_index(
                    index_path=index_path,
                    model_hf_id=spec.hf_id,
                    dataset=self.run_cfg.dataset,
                    train_tokens=train_tokens,
                    num_gpus=self.run_cfg.num_gpus,
                    projection_dim=self.run_cfg.projection_dim,
                )
                if not ok:
                    raise RuntimeError(f"Build failed: {err}")

            # Read confirmed projection_dim and index size from the saved index
            cfg_path = index_path / "index_config.yaml"
            if cfg_path.exists():
                import yaml

                with open(cfg_path) as f:
                    saved_cfg = yaml.safe_load(f)
                projection_dim = saved_cfg.get(
                    "projection_dim", self.run_cfg.projection_dim
                )

            # Determine index size from the stored gradient file
            from bergson.data import load_gradients

            mmap = load_gradients(index_path)
            if mmap.dtype.names:
                index_size = mmap[mmap.dtype.names[0]].shape[0]

            print(
                f"Index: {index_size} examples, projection_dim={projection_dim}"
            )

            # ------------------------------------------------------------------
            # Step 2: Load the model for forward/backward passes
            # ------------------------------------------------------------------
            device = self.run_cfg.device
            if device == "cuda" and not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                device = "cpu"

            print("Loading model for trace forward/backward passes...")
            load_cfg = IndexConfig(
                run_path=str(benchmark_path),
                model=spec.hf_id,
                precision=self.run_cfg.precision,
            )
            hf_model, _ = setup_model_and_peft(load_cfg)
            hf_model.eval()
            vocab_size: int = hf_model.config.vocab_size
            print(f"  Model loaded. vocab_size={vocab_size}")

            # ------------------------------------------------------------------
            # Step 3: Initialise Attributors
            # ------------------------------------------------------------------
            print("Initialising normal Attributor...")
            t0 = time.perf_counter()
            attributor_normal = Attributor(
                index_path=index_path,
                device=device,
                unit_norm=False,
                precondition=False,
            )
            normal_init_seconds = time.perf_counter() - t0
            print(f"  Normal Attributor init: {normal_init_seconds:.3f}s")

            print("Initialising TrackStar Attributor (unit_norm=True, precondition=True)...")
            t0 = time.perf_counter()
            attributor_trackstar = Attributor(
                index_path=index_path,
                device=device,
                unit_norm=True,
                precondition=True,
            )
            trackstar_init_seconds = time.perf_counter() - t0
            print(f"  TrackStar Attributor init: {trackstar_init_seconds:.3f}s")

            # ------------------------------------------------------------------
            # Step 4: Trace benchmarks
            # ------------------------------------------------------------------
            
            for num_q in self.run_cfg.query_counts:
                input_ids = _make_input_ids(
                    num_q, self.run_cfg.seq_len, vocab_size, device
                )

                for mode, attributor in [
                    ("normal", attributor_normal),
                    ("trackstar", attributor_trackstar),
                ]:
                    print(
                        f"  Trace [{mode:10s}] num_queries={num_q:4d} ...",
                        end=" ",
                        flush=True,
                    )
                    elapsed = _time_trace(
                        attributor,
                        hf_model,
                        input_ids,
                        num_repeats=self.run_cfg.num_trace_repeats,
                    )
                    print(f"{elapsed*1000:.2f} ms")
                    trace_results.append(
                        asdict(
                            TraceResult(
                                mode=mode,
                                num_queries=num_q,
                                trace_seconds=elapsed,
                              
                            )
                        )
                    )

        except Exception as exc:
            status = "error"
            error_message = repr(exc)
            import traceback

            traceback.print_exc()

        end_wall = timestamp()
        total_runtime = time.perf_counter() - overall_start

        record = AttributorBenchmarkRecord(
            schema_version=SCHEMA_VERSION,
            status=status,
            model_key=spec.key,
            model_name=spec.hf_id,
            params=spec.params,
            train_tokens=train_tokens,
            projection_dim=projection_dim,
            index_size=index_size,
            build_seconds=build_seconds,
            normal_init_seconds=normal_init_seconds,
            trackstar_init_seconds=trackstar_init_seconds,
            trace_results=trace_results,
            start_time=start_wall,
            end_time=end_wall,
            run_path=str(benchmark_path),
            hardware=get_hardware_info(),
            notes=self.run_cfg.notes,
            error=error_message,
            num_gpus=self.run_cfg.num_gpus,
        )
        save_record(benchmark_path, record, "benchmark_attributor.json")
        print(json.dumps(asdict(record), indent=2))

        if status != "success":
            sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class Main:
    """Benchmark Attributor query performance (normal vs TrackStar)."""

    run_cfg: RunConfig

    def execute(self) -> None:
        Run(run_cfg=self.run_cfg).execute()


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        conflict_resolution=ConflictResolution.EXPLICIT,
        description="Benchmark Attributor query performance (normal vs TrackStar)",
    )
    parser.add_arguments(Main, dest="prog")
    return parser


def main(args: Optional[list[str]] = None) -> None:
    parser = get_parser()
    prog: Main = parser.parse_args(args=args).prog
    prog.execute()


if __name__ == "__main__":
    main()
