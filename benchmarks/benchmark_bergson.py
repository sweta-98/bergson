"""Benchmark Bergson influence analysis scaling (in-memory reduce + score)."""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from datasets import Dataset
from simple_parsing import ArgumentParser, ConflictResolution, field
from transformers import AutoTokenizer

from benchmarks.benchmark_utils import (
    DEFAULT_DATASET,
    MODEL_SPECS,
    ModelSpec,
    get_run_path,
    parse_tokens,
    save_record,
    timestamp,
)
from bergson.build import build
from bergson.collection import collect_gradients
from bergson.collector.collector import CollectorComputer
from bergson.collector.in_memory_collector import InMemoryCollector
from bergson.config import DataConfig, IndexConfig, ReduceConfig, ScoreConfig
from bergson.data import allocate_batches, load_gradients
from bergson.gradients import GradientProcessor
from bergson.reduce import reduce
from bergson.score.score import create_scorer
from bergson.score.score_writer import InMemoryScoreWriter
from bergson.score.scorer import Scorer
from bergson.utils.auto_batch_size import (
    determine_batch_size_disk,
    determine_batch_size_in_memory,
    get_optimal_batch_size,
)
from bergson.utils.utils import assert_type
from bergson.utils.worker_utils import (
    create_processor,
    setup_data_pipeline,
    setup_model_and_peft,
)

SCHEMA_VERSION = 1
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_EVAL_SPLIT = "validation"


@dataclass
class RunRecord:
    """Record of an in-memory benchmark run."""

    schema_version: int
    status: str
    model_key: str
    model_name: str
    params: float
    train_tokens: int
    eval_tokens: int
    dataset: str
    train_split: str
    eval_split: str
    batch_size: int
    max_length: int
    reduce_seconds: float | None  # Time to collect training gradients
    query_seconds: float | None  # Time to collect query gradients
    score_seconds: float | None  # Time to compute inner products
    total_runtime_seconds: float | None
    start_time: str
    end_time: str
    run_path: str
    notes: str | None
    error: str | None
    num_gpus: int = 1  # Default for backwards compatibility
    token_batch_size: int | None = (
        None  # Auto-determined or configured token batch size
    )


@dataclass
class RunConfig:
    """Configuration for an in-memory benchmark run."""

    model: str = field(positional=True)
    """Key for the model to benchmark (e.g., pythia-14m, pythia-70m)."""

    train_tokens: str = field(positional=True)
    """Target training tokens (e.g., 1M, 10M)."""

    run_root: str = field(positional=True)
    """Root directory for benchmark runs."""

    eval_tokens: int = 1024
    """Target evaluation tokens per sequence. Not
    analogous to train_tokens."""

    eval_sequences: int = 1
    """Target evaluation sequences."""

    batch_size: int = 4
    """Batch size for gradient collection."""

    max_length: int = 1024
    """Maximum sequence length."""

    max_eval_examples: int = 10
    """Maximum number of evaluation examples to score."""

    dataset: str = DEFAULT_DATASET
    """Dataset to use for benchmarking."""

    train_split: str = DEFAULT_TRAIN_SPLIT
    """Dataset split to use for training gradients."""

    eval_split: str = DEFAULT_EVAL_SPLIT
    """Dataset split to use for evaluation gradients."""

    auto_batch_size: bool = True
    """Automatically determine optimal token_batch_size for hardware."""

    starting_batch_size: int = 16384
    """Starting token_batch_size for auto-determination (optimized to power of 2)."""

    run_path: str | None = None
    """Explicit run path (overrides auto-generated path)."""

    tag: str | None = None
    """Tag for the run (used in auto-generated path)."""

    notes: str | None = None
    """Optional notes for the run."""


def load_records(root: Path) -> list[RunRecord]:
    """Load all benchmark records from a directory tree."""
    records: list[RunRecord] = []
    for meta in root.rglob("benchmark.json"):
        try:
            with open(meta, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            records.append(RunRecord(**payload))
        except Exception as exc:
            print(f"Warning: failed to read {meta}: {exc}", file=sys.stderr)
    return records


@dataclass
class Run:
    """Execute a single in-memory Bergson benchmark run."""

    run_cfg: RunConfig

    def execute(self) -> None:
        """Run the benchmark."""
        if self.run_cfg.model not in MODEL_SPECS:
            raise ValueError(f"Unknown model '{self.run_cfg.model}'")

        spec = MODEL_SPECS[self.run_cfg.model]
        train_tokens = parse_tokens(self.run_cfg.train_tokens)
        eval_tokens = self.run_cfg.eval_tokens
        eval_sequences = self.run_cfg.eval_sequences

        print(
            f"Running Bergson benchmark for {self.run_cfg.model} with "
            f"{train_tokens} train and {eval_tokens} eval tokens"
        )

        run_root = Path(self.run_cfg.run_root).resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        run_path = (
            Path(self.run_cfg.run_path).resolve()
            if self.run_cfg.run_path
            else get_run_path(
                run_root,
                spec,
                train_tokens,
                eval_tokens,
                eval_sequences,
                self.run_cfg.tag,
            )
        )

        start_wall = timestamp()
        start = time.perf_counter()
        status = "success"
        error_message: str | None = None
        reduce_time: float | None = None
        query_time: float | None = None
        score_time: float | None = None
        optimal_token_batch_size: int | None = None

        try:
            # Create configs for data loading
            train_data_cfg = DataConfig(
                dataset=self.run_cfg.dataset,
                split=self.run_cfg.train_split,
                prompt_column="text",
            )
            train_index_cfg = IndexConfig(
                run_path="temp",
                model=spec.hf_id,
                data=train_data_cfg,
                token_batch_size=self.run_cfg.max_length,
                max_tokens=train_tokens,
                precision="bf16",
                fsdp=False,
            )

            eval_data_cfg = DataConfig(
                dataset=self.run_cfg.dataset,
                split=self.run_cfg.eval_split,
                prompt_column="text",
            )
            eval_index_cfg = IndexConfig(
                run_path="temp",
                model=spec.hf_id,
                data=eval_data_cfg,
                token_batch_size=self.run_cfg.max_length,
                max_tokens=eval_tokens,
                precision="bf16",
            )

            # Load model using worker utility
            model, _ = setup_model_and_peft(train_index_cfg, device_map_auto=True)

            # Load datasets using setup_data_pipeline
            train_dataset = assert_type(Dataset, setup_data_pipeline(train_index_cfg))
            eval_dataset = assert_type(Dataset, setup_data_pipeline(eval_index_cfg))

            # Determine optimal token_batch_size if requested
            if self.run_cfg.auto_batch_size:
                cache_path = run_path / "batch_size_cache.json"
                tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
                optimal_token_batch_size = get_optimal_batch_size(
                    cache_path=cache_path,
                    model_hf_id=spec.hf_id,
                    fsdp=False,
                    starting_batch_size=self.run_cfg.starting_batch_size,
                    determine_fn=lambda: determine_batch_size_in_memory(
                        model=model,
                        tokenizer=tokenizer,
                        dataset=train_dataset,
                        max_length=self.run_cfg.max_length,
                        starting_batch_size=self.run_cfg.starting_batch_size,
                    ),
                )
            else:
                optimal_token_batch_size = self.run_cfg.max_length

            print(f"Using token_batch_size={optimal_token_batch_size}")

            # Create processor (no normalization, no preconditioners)
            processor = GradientProcessor(
                normalizers={},
                projection_dim=16,
                reshape_to_square=False,
                projection_type="rademacher",
            )

            # Create IndexConfig for using CollectorComputer
            index_cfg = IndexConfig(
                run_path="temp",
                model=spec.hf_id,
                token_batch_size=optimal_token_batch_size,
                loss_fn="ce",
                loss_reduction="mean",
            )

            # QUERY PHASE: Collect query gradients first (needed for scorer)
            print("Collecting query gradients (query phase)...")
            query_start = time.perf_counter()

            # Limit eval examples
            eval_subset = eval_dataset.select(
                range(min(self.run_cfg.max_eval_examples, len(eval_dataset)))
            )

            # Collect all query gradients in one pass
            query_collector = InMemoryCollector(
                model=model.base_model,  # type: ignore
                processor=processor,
                data=eval_subset,
                cfg=index_cfg,
            )

            query_batches = allocate_batches(
                eval_subset["length"], optimal_token_batch_size  # type: ignore
            )

            computer = CollectorComputer(
                model=model,
                data=eval_subset,
                collector=query_collector,
                batches=query_batches,
                cfg=index_cfg,
            )
            computer.run_with_collector_hooks(desc="query gradients")

            # Concatenate query gradients
            query_grads = {
                name: torch.cat(grads, dim=0)
                for name, grads in query_collector.gradients.items()
            }

            query_time = time.perf_counter() - query_start
            print(f"Query phase completed in {query_time:.2f} seconds")
            shapes = [(k, v.shape) for k, v in query_grads.items()]
            print(f"Query gradients shape: {shapes}")

            # Create scorer with query gradients (in-memory)
            modules = list(query_grads.keys())
            num_queries = len(query_grads[modules[0]])
            writer = InMemoryScoreWriter(
                len(train_dataset), num_queries, dtype=torch.bfloat16
            )
            scorer = Scorer(
                query_grads=query_grads,
                modules=modules,
                writer=writer,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )

            # REDUCE PHASE: Collect training gradients with scorer
            print("Collecting training gradients (reduce phase)...")
            reduce_start = time.perf_counter()

            # Create in-memory collector for training gradients with scorer attached
            train_collector = InMemoryCollector(
                model=model.base_model,  # type: ignore
                processor=processor,
                data=train_dataset,
                cfg=index_cfg,
                scorer=scorer,
            )

            # Create batches for CollectorComputer
            batches = allocate_batches(
                train_dataset["length"], optimal_token_batch_size  # type: ignore
            )

            # Use CollectorComputer to process training data
            computer = CollectorComputer(
                model=model,
                data=train_dataset,
                collector=train_collector,
                batches=batches,
                cfg=index_cfg,
            )
            computer.run_with_collector_hooks(desc="training gradients")

            reduce_time = time.perf_counter() - reduce_start
            print(f"Reduce phase completed in {reduce_time:.2f} seconds")

            # SCORE PHASE: Retrieve scores from scorer
            print("Retrieving influence scores (score phase)...")
            score_start = time.perf_counter()

            # Scores are already computed, just transpose to [num_queries, num_train]
            all_scores = writer.scores.T.tolist()

            score_time = time.perf_counter() - score_start
            print(f"Score phase completed in {score_time:.2f} seconds")
            print(f"Computed scores for {len(all_scores)} test examples")

        except Exception as exc:
            status = "error"
            error_message = repr(exc)

            traceback.print_exc()

        runtime = time.perf_counter() - start
        end_wall = timestamp()

        record = RunRecord(
            schema_version=SCHEMA_VERSION,
            status=status,
            model_key=spec.key,
            model_name=spec.hf_id,
            params=spec.params,
            train_tokens=train_tokens,
            eval_tokens=eval_tokens,
            dataset=self.run_cfg.dataset,
            train_split=self.run_cfg.train_split,
            eval_split=self.run_cfg.eval_split,
            batch_size=self.run_cfg.batch_size,
            max_length=self.run_cfg.max_length,
            num_gpus=1,
            reduce_seconds=reduce_time,
            query_seconds=query_time,
            score_seconds=score_time,
            total_runtime_seconds=runtime,
            start_time=start_wall,
            end_time=end_wall,
            run_path=str(run_path),
            notes=self.run_cfg.notes,
            error=error_message,
            token_batch_size=optimal_token_batch_size,
        )
        save_record(run_path, record)

        print(json.dumps(asdict(record), indent=2))

        if status != "success":
            sys.exit(1)


@dataclass
class RunDisk:
    """Execute a disk-based Bergson benchmark using real build/reduce/score."""

    run_cfg: RunConfig

    def _build_phase(
        self,
        spec: "ModelSpec",
        train_tokens: int,
        optimal_token_batch_size: int,
        train_index_path: Path,
    ) -> float:
        """Build training index and return time taken."""
        print("Building training index...")
        build_start = time.perf_counter()

        train_index_cfg = IndexConfig(
            run_path=str(train_index_path),
            model=spec.hf_id,
            data=DataConfig(
                dataset=self.run_cfg.dataset,
                split=self.run_cfg.train_split,
                prompt_column="text",
            ),
            token_batch_size=optimal_token_batch_size,
            projection_dim=16,
            skip_preconditioners=True,  # Skip preconditioners for speed
            max_tokens=train_tokens,
            overwrite=True,
        )

        build(train_index_cfg)
        build_time = time.perf_counter() - build_start
        print(f"Build phase completed in {build_time:.2f} seconds")
        return build_time

    def _reduce_phase(
        self,
        spec: "ModelSpec",
        train_tokens: int,
        optimal_token_batch_size: int,
        reduce_path: Path,
    ) -> float:
        """Reduce training gradients and return time taken."""
        print("Reducing training gradients...")
        reduce_start = time.perf_counter()

        reduce_index_cfg = IndexConfig(
            run_path=str(reduce_path),
            model=spec.hf_id,
            data=DataConfig(
                dataset=self.run_cfg.dataset,
                split=self.run_cfg.train_split,
                prompt_column="text",
            ),
            token_batch_size=optimal_token_batch_size,
            projection_dim=16,
            skip_preconditioners=True,
            max_tokens=train_tokens,
            overwrite=True,
        )

        reduce_cfg = ReduceConfig(method="mean", unit_normalize=False)

        reduce(reduce_index_cfg, reduce_cfg)
        reduce_time = time.perf_counter() - reduce_start
        print(f"Reduce phase completed in {reduce_time:.2f} seconds")
        return reduce_time

    def _score_phase(
        self,
        spec: "ModelSpec",
        train_tokens: int,
        optimal_token_batch_size: int,
        score_path: Path,
        query_index_path: Path,
    ) -> float:
        """Score training data against query gradients and return time taken."""
        print("Setting up score phase...")

        # Setup: Create config for training data scoring
        score_index_cfg = IndexConfig(
            run_path=str(score_path),
            model=spec.hf_id,
            data=DataConfig(
                dataset=self.run_cfg.dataset,
                split=self.run_cfg.train_split,
                prompt_column="text",
            ),
            token_batch_size=optimal_token_batch_size,
            projection_dim=16,
            skip_preconditioners=True,
            max_tokens=train_tokens,
            overwrite=True,
            precision="bf16",
        )

        # Setup: Load model
        model, target_modules = setup_model_and_peft(
            score_index_cfg, device_map_auto=True
        )

        # Setup: Load training data
        train_ds = assert_type(Dataset, setup_data_pipeline(score_index_cfg))

        # Setup: Create processor
        processor = create_processor(model, train_ds, score_index_cfg, target_modules)

        # Setup: Load query gradients from disk
        query_mmap = load_gradients(query_index_path, structured=False)
        with open(query_index_path / "info.json", "r") as f:
            query_info = json.load(f)
            grad_sizes = query_info["grad_sizes"]
            modules = list(grad_sizes.keys())

        # Convert to dict of tensors
        sizes = torch.tensor(list(grad_sizes.values()))
        offsets = torch.tensor([0] + torch.cumsum(sizes, dim=0).tolist())
        query_grads = {
            name: torch.from_numpy(query_mmap[:, offsets[i] : offsets[i + 1]].copy())
            for i, name in enumerate(modules)
        }

        # Setup: Create scorer
        score_cfg = ScoreConfig(
            query_path=str(query_index_path),
            modules=modules,
            score="mean",
            unit_normalize=False,
        )
        scorer = create_scorer(
            score_path,
            len(train_ds),
            query_grads,
            score_cfg,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )

        # Setup: Prepare collect_gradients kwargs
        batches = allocate_batches(train_ds["length"], optimal_token_batch_size)

        # TIMED: Only time the actual gradient collection
        print("Scoring query gradients (timed)...")
        score_start = time.perf_counter()

        collect_gradients(
            model=model,
            data=train_ds,
            processor=processor,
            cfg=score_index_cfg,
            target_modules=target_modules,
            batches=batches,
            scorer=scorer,
        )

        score_time = time.perf_counter() - score_start
        print(f"Score phase completed in {score_time:.2f} seconds")
        return score_time

    def execute(self) -> None:
        """Run the disk-based benchmark."""
        if self.run_cfg.model not in MODEL_SPECS:
            raise ValueError(f"Unknown model '{self.run_cfg.model}'")

        spec = MODEL_SPECS[self.run_cfg.model]
        train_tokens = parse_tokens(self.run_cfg.train_tokens)
        eval_tokens = self.run_cfg.eval_tokens
        eval_sequences = self.run_cfg.eval_sequences

        print(
            f"Running disk-based Bergson benchmark for {self.run_cfg.model} with "
            f"{train_tokens} train and {eval_tokens} eval tokens"
        )

        run_root = Path(self.run_cfg.run_root).resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        run_path = (
            Path(self.run_cfg.run_path).resolve()
            if self.run_cfg.run_path
            else get_run_path(
                run_root,
                spec,
                train_tokens,
                eval_tokens,
                eval_sequences,
                self.run_cfg.tag,
            )
        )

        start_wall = timestamp()
        start = time.perf_counter()
        status = "success"
        error_message: str | None = None
        build_time: float | None = None
        query_time: float | None = None
        reduce_time: float | None = None
        score_time: float | None = None
        optimal_token_batch_size: int | None = None

        # Determine optimal token_batch_size if requested
        if self.run_cfg.auto_batch_size:
            cache_path = run_path / "batch_size_cache.json"
            optimal_token_batch_size = get_optimal_batch_size(
                cache_path=cache_path,
                model_hf_id=spec.hf_id,
                fsdp=False,
                starting_batch_size=self.run_cfg.starting_batch_size,
                determine_fn=lambda: determine_batch_size_disk(
                    model_hf_id=spec.hf_id,
                    dataset_name=self.run_cfg.dataset,
                    dataset_split=self.run_cfg.train_split,
                    max_length=self.run_cfg.max_length,
                    starting_batch_size=self.run_cfg.starting_batch_size,
                    use_fsdp=False,
                ),
            )
        else:
            optimal_token_batch_size = self.run_cfg.max_length

        print(f"Using token_batch_size={optimal_token_batch_size}")

        try:
            # Create paths for different phases
            query_index_path = run_path / "query_index"
            reduce_path = run_path / "reduce"
            score_path = run_path / "score"

            # BUILD PHASE: Build index for training data
            # build_time = self._build_phase(
            #     spec, train_tokens, optimal_token_batch_size, train_index_path
            # )

            # QUERY PHASE: Build query index
            print("Building query index (query phase)...")
            query_start = time.perf_counter()

            query_index_cfg = IndexConfig(
                run_path=str(query_index_path),
                model=spec.hf_id,
                data=DataConfig(
                    dataset=self.run_cfg.dataset,
                    split=self.run_cfg.eval_split,
                    prompt_column="text",
                ),
                token_batch_size=optimal_token_batch_size,
                projection_dim=16,
                skip_preconditioners=True,
                max_tokens=eval_tokens,
                overwrite=True,
            )

            build(query_index_cfg)
            query_time = time.perf_counter() - query_start
            print(f"Query phase completed in {query_time:.2f} seconds")

            # REDUCE PHASE
            reduce_time = self._reduce_phase(
                spec, train_tokens, optimal_token_batch_size, reduce_path
            )

            # SCORE PHASE
            score_time = self._score_phase(
                spec,
                train_tokens,
                optimal_token_batch_size,
                score_path,
                query_index_path,
            )

        except Exception as exc:
            status = "error"
            error_message = repr(exc)
            traceback.print_exc()

        runtime = time.perf_counter() - start
        end_wall = timestamp()

        # Create record with build/reduce/score times
        record = RunRecord(
            schema_version=SCHEMA_VERSION,
            status=status,
            model_key=spec.key,
            model_name=spec.hf_id,
            params=spec.params,
            train_tokens=train_tokens,
            eval_tokens=eval_tokens,
            dataset=self.run_cfg.dataset,
            train_split=self.run_cfg.train_split,
            eval_split=self.run_cfg.eval_split,
            batch_size=self.run_cfg.batch_size,
            max_length=self.run_cfg.max_length,
            num_gpus=1,  # Disk-based always single GPU
            reduce_seconds=reduce_time,
            query_seconds=query_time,
            score_seconds=score_time,
            total_runtime_seconds=runtime,
            start_time=start_wall,
            end_time=end_wall,
            run_path=str(run_path),
            notes=f"disk-based (build={build_time:.2f}s)" if build_time else None,
            error=error_message,
            token_batch_size=optimal_token_batch_size,
        )
        save_record(run_path, record)

        print(json.dumps(asdict(record), indent=2))

        if status != "success":
            sys.exit(1)


@dataclass
class Main:
    """Benchmark Bergson influence analysis scaling."""

    command: Union[Run, RunDisk]

    def execute(self) -> None:
        """Run the selected command."""
        self.command.execute()


def get_parser() -> ArgumentParser:
    """Get the argument parser. Used for documentation generation."""
    parser = ArgumentParser(
        conflict_resolution=ConflictResolution.EXPLICIT,
        description="Benchmark Bergson influence analysis scaling (in-memory)",
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
