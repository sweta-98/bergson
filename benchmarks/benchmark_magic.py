"""Benchmark MAGIC attribution: time training and gradient phases separately.

Validation is skipped. Peak VRAM is measured per phase via nvidia-smi polling.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from datasets import Dataset
from simple_parsing import ArgumentParser, ConflictResolution, field
from torchopt.pytree import tree_iter
from benchmarks.benchmark_utils import (
    MODEL_SPECS,
    get_hardware_info,
    get_run_path,
    parse_tokens,
    save_record,
    timestamp,
)
from bergson.config import DataConfig
from bergson.distributed import launch_distributed_run
from bergson.magic.cli import (
    compute_query_gradients,
    pad_dataset_to_batch_size,
    prepare_trainer,
)
from benchmarks.benchmark_bergson_cli import VramMonitor
from bergson.magic.config import MagicConfig
from bergson.magic.data_stream import DataStream
from bergson.magic.trainer import BackwardState
from bergson.utils.worker_utils import setup_data_pipeline
from bergson.utils.worker_utils import filter_by_max_tokens

SCHEMA_VERSION = 1


@dataclass
class MagicRunRecord:
    """Record of a MAGIC benchmark run."""

    schema_version: int
    status: str
    model_key: str
    model_name: str
    params: float
    train_tokens: int
    dataset: str
    batch_size: int
    train_seconds: float | None
    """Wall-clock seconds for the forward training pass."""
    gradient_seconds: float | None
    """Wall-clock seconds for query gradient + backward attribution pass."""
    total_runtime_seconds: float | None
    start_time: str
    end_time: str
    run_path: str
    notes: str | None
    error: str | None
    num_gpus: int = 1
    hardware: str | None = None
    num_queries: int = 1
    # Peak VRAM (MB) per phase, measured via nvidia-smi polling
    train_peak_vram_mb: float | None = None
    gradient_peak_vram_mb: float | None = None


@dataclass
class RunConfig:
    """Configuration for a MAGIC benchmark run."""

    model: str = field(positional=True)
    """Key for the model to benchmark (e.g., pythia-14m, pythia-70m)."""

    train_tokens: str = field(positional=True)
    """Target training tokens (e.g., 1M, 10M)."""

    run_root: str = field(positional=True)
    """Root directory for benchmark runs."""

    dataset: str = ""
    """Dataset path/identifier to use for benchmarking."""

    query_dataset: str = ""
    """Dataset path/identifier for the query. Defaults to a single-example dataset."""

    run_path: str | None = None
    """Explicit run path (overrides auto-generated path)."""

    tag: str | None = None
    """Tag for the run (used in auto-generated path)."""

    notes: str | None = None
    """Optional notes for the run."""

    num_gpus: int = 1
    """Number of GPUs to use for benchmarking."""

    batch_size: int = 8
    """Batch size for training and query streams."""

    skip_existing: bool = True
    """Skip benchmark if a successful run exists for this model/token combo."""

    num_queries: int = 1
    """Number of query examples to score against the training set."""

    precision: str = "fp32"
    """Precision for the model (fp32, bf16, fp16)."""

    save_mode: str = "sqrt"
    """MAGIC checkpoint saving mode (all, sqrt, log)."""


def load_records(root: Path) -> list[MagicRunRecord]:
    """Load all benchmark records from a directory tree."""
    records: list[MagicRunRecord] = []
    for meta in root.rglob("benchmark_magic.json"):
        try:
            with open(meta, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            records.append(MagicRunRecord(**payload))
        except Exception as exc:
            print(f"Warning: failed to read {meta}: {exc}", file=sys.stderr)
    return records


def find_existing_successful_run(
    run_root: Path,
    model_key: str,
    train_tokens: int,
    dataset: str,
    num_gpus: int,
    num_queries: int = 1,
) -> MagicRunRecord | None:
    """Return a successful benchmark record if one exists, else None."""
    for record in load_records(run_root):
        if (
            record.status == "success"
            and record.model_key == model_key
            and record.train_tokens == train_tokens
            and record.dataset == dataset
            and record.num_gpus == num_gpus
            and record.num_queries == num_queries
        ):
            return record
    return None


# ---------------------------------------------------------------------------
# Instrumented worker (no validation, per-phase timing + VRAM)
# ---------------------------------------------------------------------------

# These are filled in by the main process before launching the distributed run.
_BENCHMARK_RESULTS: dict = {}


def _benchmark_worker(
    global_rank: int,
    rank: int,
    world_size: int,
    train_dataset: Dataset,
    query_dataset: Dataset,
    num_train_docs: int,
    num_query_docs: int,
    run_cfg: MagicConfig,
) -> None:
    """Mirror of magic worker with timing/VRAM instrumentation and no validation."""
    torch.cuda.set_device(rank)

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{rank}"),
            rank=rank,
            world_size=world_size,
        )

    if run_cfg.num_epochs > 1:
        train_dataset = train_dataset.repeat(run_cfg.num_epochs)

    assert run_cfg.batch_size % world_size == 0

    train_dataset, num_train_docs, pad_count = pad_dataset_to_batch_size(
        train_dataset, run_cfg.batch_size, num_train_docs, "Train", global_rank
    )

    if run_cfg.per_token:
        seq_len = run_cfg.data.chunk_length
        if seq_len <= 0:
            seq_len = max(train_dataset["length"])
        w_shape = (len(train_dataset), seq_len)
    else:
        w_shape = (num_train_docs,)

    stream = DataStream(
        train_dataset,
        run_cfg.batch_size,
        device=f"cuda:{rank}",
        input_key=run_cfg.data.prompt_column,
        weight_shape=w_shape,
    )
    if pad_count:
        stream.weights.data[-pad_count:] = 0.0

    schedule = run_cfg.lr_schedule.get_schedule(len(stream))
    trainer, fwd_state, model = prepare_trainer(run_cfg, rank, schedule)

    ckpts_path = os.path.join(run_cfg.run_path, "checkpoints")
    path0 = os.path.join(ckpts_path, "state0.pt")

    save_fut = fwd_state.save(path0)

    # ------------------------------------------------------------------
    # Phase 1: Training (forward pass)
    # ------------------------------------------------------------------
    train_monitor = VramMonitor(gpu_index=rank, num_gpus=world_size)
    train_monitor.start()
    train_start = time.perf_counter()

    fwd_state = trainer.train(
        fwd_state,
        stream,
        debug=run_cfg.debug,
        inplace=True,
        save_dir=ckpts_path,
        save_mode=run_cfg.save_mode,
        log_fn=None,
        resume=False,
        fsdp=run_cfg.fsdp,
    )

    train_seconds = time.perf_counter() - train_start
    train_peak_vram_mb = train_monitor.stop()

    if save_fut is not None:
        save_fut.result()

    # ------------------------------------------------------------------
    # Phase 2: Query gradient + backward attribution
    # ------------------------------------------------------------------
    query_dataset, num_query_docs, query_pad_count = pad_dataset_to_batch_size(
        query_dataset, run_cfg.batch_size, num_query_docs, "Query", global_rank
    )
    if len(query_dataset) < run_cfg.batch_size:
        raise ValueError(
            f"Query dataset has {len(query_dataset)} examples, fewer than "
            f"batch_size={run_cfg.batch_size}. Use a larger query dataset or "
            f"smaller batch_size."
        )

    query_stream = DataStream(
        query_dataset,
        run_cfg.batch_size,
        device=f"cuda:{rank}",
        input_key=run_cfg.query.prompt_column,
        weight_shape=(num_query_docs,),
    )
    if query_pad_count:
        query_stream.weights.data[-query_pad_count:] = 0.0

    grad_monitor = VramMonitor(gpu_index=rank, num_gpus=world_size)
    grad_monitor.start()
    grad_start = time.perf_counter()

    query_grads, baseline = compute_query_gradients(
        fwd_state, model, query_stream, run_cfg.query_method, run_cfg.fsdp
    )

    stream.requires_grad = True
    opt_grads = [
        torch.zeros_like(buf)
        for buf in tree_iter(fwd_state.opt_state)
        if isinstance(buf, torch.Tensor) and buf.is_floating_point()
    ]
    bwd_state = BackwardState(query_grads, opt_grads, torch.zeros_like(stream.weights))

    bwd_state = trainer.backward(
        ckpts_path,
        stream,
        bwd_state,
        fwd_state,
        debug=run_cfg.debug,
        inplace=True,
        fsdp=run_cfg.fsdp,
        resume=False,
        save_every=run_cfg.backward_save_every,
        save_mode=run_cfg.save_mode,
    )

    if world_size > 1:
        dist.all_reduce(bwd_state.weight_grads, op=dist.ReduceOp.SUM)

    gradient_seconds = time.perf_counter() - grad_start
    gradient_peak_vram_mb = grad_monitor.stop()

    scores = bwd_state.weight_grads.cpu()
    if pad_count:
        scores = scores[:-pad_count]

    if global_rank == 0:
        print(f"Baseline loss: {baseline}")
        print(f"Training phase:  {train_seconds:.2f}s  (peak VRAM: {train_peak_vram_mb:.0f} MB)")
        print(f"Gradient phase:  {gradient_seconds:.2f}s  (peak VRAM: {gradient_peak_vram_mb:.0f} MB)")

        score_path = os.path.join(run_cfg.run_path, "scores.pt")
        torch.save(scores, score_path)
        print(f"Saved attribution scores to {score_path}")

        # Communicate results back via global dict (single-process path always
        # ends up here; multi-GPU workers share the same process memory on rank 0).
        _BENCHMARK_RESULTS["train_seconds"] = train_seconds
        _BENCHMARK_RESULTS["gradient_seconds"] = gradient_seconds
        _BENCHMARK_RESULTS["train_peak_vram_mb"] = train_peak_vram_mb
        _BENCHMARK_RESULTS["gradient_peak_vram_mb"] = gradient_peak_vram_mb


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------


@dataclass
class Run:
    """Execute a single MAGIC benchmark run."""

    run_cfg: RunConfig

    def execute(self) -> None:
        if not self.run_cfg.dataset:
            from benchmarks.benchmark_utils import prepare_benchmark_ds_path
            self.run_cfg.dataset = str(prepare_benchmark_ds_path())

        if self.run_cfg.model not in MODEL_SPECS:
            raise ValueError(f"Unknown model '{self.run_cfg.model}'")

        spec = MODEL_SPECS[self.run_cfg.model]
        train_tokens = parse_tokens(self.run_cfg.train_tokens)

        print(
            f"Running MAGIC benchmark for {self.run_cfg.model} with "
            f"{train_tokens} train tokens."
        )

        run_root = Path(self.run_cfg.run_root).resolve()
        run_root.mkdir(parents=True, exist_ok=True)

        if self.run_cfg.skip_existing:
            existing = find_existing_successful_run(
                run_root=run_root,
                model_key=spec.key,
                train_tokens=train_tokens,
                dataset=self.run_cfg.dataset,
                num_gpus=self.run_cfg.num_gpus,
                num_queries=self.run_cfg.num_queries,
            )
            if existing is not None:
                print(
                    f"Skipping: found existing successful run at {existing.run_path}. "
                    f"Use --skip_existing=False to force re-run."
                )
                return

        num_queries = self.run_cfg.num_queries

        benchmark_path = (
            Path(self.run_cfg.run_path).resolve()
            if self.run_cfg.run_path
            else get_run_path(
                run_root,
                spec,
                train_tokens,
                eval_tokens=0,
                eval_sequences=num_queries,
                tag=self.run_cfg.tag,
                num_gpus=self.run_cfg.num_gpus,
            )
        )

        # Build an n-example query dataset if not provided
        if self.run_cfg.query_dataset:
            query_dataset_path = Path(self.run_cfg.query_dataset).resolve()
        else:
            query_dataset_path = benchmark_path / "query_dataset"
            if not query_dataset_path.exists():
                print(f"Creating {num_queries}-example query dataset...")
                qds = Dataset.from_dict({"text": [f"Query example {i}." for i in range(num_queries)]})
                qds.save_to_disk(str(query_dataset_path))

        # Build MagicConfig
        magic_cfg = MagicConfig(
            run_path=str(benchmark_path),
            model=spec.hf_id,
            precision=self.run_cfg.precision,
            overwrite=True,
            batch_size=self.run_cfg.batch_size,
            token_batch_size=1024, 
            save_mode=self.run_cfg.save_mode,
            num_subsets=0,  # skip validation
            data=DataConfig(
                dataset=self.run_cfg.dataset,
                split="train[:1%]",
                truncation=True,
            ),
            query=DataConfig(
                dataset=str(query_dataset_path),
            ),
        )

        # Prepare datasets
        train_ds, train_n = setup_data_pipeline(magic_cfg)
        # Limit to train_tokens using the built-in filter
        magic_cfg.max_tokens = train_tokens
        train_ds = filter_by_max_tokens(train_ds, magic_cfg)

        train_n = len(train_ds)
        print(f"Train dataset: {train_n} sequences after max_tokens={train_tokens} filter")
        previous_max = magic_cfg.max_tokens
        magic_cfg.max_tokens = None  # disable max_tokens for query dataset
        query_ds, query_n = setup_data_pipeline(magic_cfg, magic_cfg.query)
        magic_cfg.max_tokens = previous_max  # restore for worker use
        benchmark_path.mkdir(parents=True, exist_ok=True)

        _BENCHMARK_RESULTS.clear()

        start_wall = timestamp()
        start = time.perf_counter()
        status = "success"
        error_message: str | None = None

        try:
            launch_distributed_run(
                "benchmark_magic_worker",
                _benchmark_worker,
                [train_ds, query_ds, train_n, query_n, magic_cfg],
                magic_cfg.distributed,
            )
        except Exception as exc:
            status = "error"
            error_message = repr(exc)
            import traceback
            traceback.print_exc()

        total_runtime = time.perf_counter() - start
        end_wall = timestamp()

        record = MagicRunRecord(
            schema_version=SCHEMA_VERSION,
            status=status,
            model_key=spec.key,
            model_name=spec.hf_id,
            params=spec.params,
            train_tokens=train_tokens,
            dataset=self.run_cfg.dataset,
            batch_size=self.run_cfg.batch_size,
            train_seconds=_BENCHMARK_RESULTS.get("train_seconds"),
            gradient_seconds=_BENCHMARK_RESULTS.get("gradient_seconds"),
            total_runtime_seconds=total_runtime,
            start_time=start_wall,
            end_time=end_wall,
            run_path=str(benchmark_path),
            notes=self.run_cfg.notes,
            error=error_message,
            num_gpus=self.run_cfg.num_gpus,
            hardware=get_hardware_info(),
            num_queries=num_queries,
            train_peak_vram_mb=_BENCHMARK_RESULTS.get("train_peak_vram_mb"),
            gradient_peak_vram_mb=_BENCHMARK_RESULTS.get("gradient_peak_vram_mb"),
        )
        save_record(benchmark_path, record, "benchmark_magic.json")

        print(json.dumps(asdict(record), indent=2))

        if status != "success":
            sys.exit(1)


@dataclass
class Main:
    """Benchmark MAGIC attribution (training and gradient phases)."""

    command: Run

    def execute(self) -> None:
        self.command.execute()


def get_parser() -> ArgumentParser:
    """Get the argument parser. Used for documentation generation."""
    parser = ArgumentParser(
        conflict_resolution=ConflictResolution.EXPLICIT,
        description="Benchmark MAGIC attribution",
    )
    parser.add_arguments(Main, dest="prog")
    return parser


def main(args: Optional[list[str]] = None) -> None:
    """Parse CLI arguments and dispatch to the selected subcommand."""
    parser = get_parser()
    prog: Main = parser.parse_args(args=args).prog
    prog.execute()


if __name__ == "__main__":
    main()
