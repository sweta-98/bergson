"""Utilities for benchmarking Dattri influence analysis scaling."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset, load_from_disk
from dattri.algorithm.base import BaseInnerProductAttributor
from dattri.func.projection import BasicProjector, ProjectionType
from dattri.task import AttributionTask
from simple_parsing import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.benchmark_utils import (
    MODEL_SPECS,
    get_hardware_details,
    get_run_path,
    parse_tokens,
    prepare_benchmark_ds_path,
    save_record,
    timestamp,
)
from bergson.utils.utils import assert_type


# Dattri implements projections in the TrakAttributor
# but it doesn't work with GPT-NeoX models due to rotary
# embedding compatibility issues with vmap.
class ProjectedInnerProductAttributor(BaseInnerProductAttributor):
    """Inner product attributor with random projection for dimensionality reduction."""

    def __init__(
        self,
        task: "AttributionTask",
        proj_dim: int = 16,
        layer_name: Optional[str] = None,
        device: Optional[str] = "cpu",
    ) -> None:
        """Initialize the projected attributor.

        Args:
            task: The attribution task.
            proj_dim: Dimension to project gradients to.
            layer_name: Optional layer name to restrict gradients to.
            device: Device to run on.
        """
        super().__init__(task, layer_name, device)
        self.proj_dim = proj_dim
        self.projector = None
        self._feature_dim = None

    def _ensure_projector(self, feature_dim: int) -> None:
        """Create projector if needed."""
        if self.projector is None or self._feature_dim != feature_dim:
            self._feature_dim = feature_dim
            self.projector = BasicProjector(
                feature_dim=feature_dim,
                proj_dim=self.proj_dim,
                seed=0,
                proj_type=ProjectionType.rademacher,
                device=torch.device(self.device),
                dtype=torch.float32,
            )

    def transform_train_rep(
        self,
        ckpt_idx: int,
        train_rep: torch.Tensor,
    ) -> torch.Tensor:
        """Project train representations to lower dimension."""
        self._ensure_projector(train_rep.shape[-1])
        return self.projector.project(train_rep, ensemble_id=ckpt_idx)

    def transform_test_rep(
        self,
        ckpt_idx: int,
        test_rep: torch.Tensor,
    ) -> torch.Tensor:
        """Project test representations to lower dimension."""
        self._ensure_projector(test_rep.shape[-1])
        return self.projector.project(test_rep, ensemble_id=ckpt_idx)


def _get_dattri_metadata(
    model_name: str,
    projection_dim: int | None,
    max_length: int,
) -> Dict[str, Any]:
    """Cache key metadata for dattri batch size tuning."""
    gpu_name = "cpu"
    gpu_mem = 0.0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_mem = props.total_memory / 1e9
    return {
        "model": model_name,
        "projection_dim": projection_dim or 0,
        "max_length": max_length,
        "gpu_name": gpu_name,
        "gpu_memory_gb": round(gpu_mem, 1),
    }


def _check_bs_cache(cache_file: Path, meta: Dict[str, Any]) -> int | None:
    """Look up a cached batch size."""
    if not cache_file.exists():
        return None
    with open(cache_file, "r") as f:
        for line in f:
            row = json.loads(line)
            if all(row.get(k) == v for k, v in meta.items()):
                return row.get("batch_size")
    return None


def _append_bs_cache(cache_file: Path, meta: Dict[str, Any], bs: int) -> None:
    """Append a batch size to the cache."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    entry = {**meta, "batch_size": bs}
    with open(cache_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"Cached batch_size {bs} to {cache_file}")


def _try_dattri_batch(
    attributor: BaseInnerProductAttributor,
    batch_size: int,
    seq_len: int,
    collate_fn,
) -> bool:
    """Test whether a given batch size fits in memory.

    Runs attributor.cache() + attribute() on a small dataset
    to test the actual per-sample gradient computation memory.
    """
    gc.collect()

    # Train set: batch_size examples in one batch.
    # Test set: 1 example (minimal query).
    train_ds = Dataset.from_dict(
        {
            "input_ids": [list(range(seq_len)) for _ in range(batch_size)],
            "attention_mask": [[1] * seq_len for _ in range(batch_size)],
        }
    )
    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )
    test_ds = Dataset.from_dict(
        {
            "input_ids": [list(range(seq_len))],
            "attention_mask": [[1] * seq_len],
        }
    )
    test_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,  # type: ignore
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,  # type: ignore
        batch_size=1,
        collate_fn=collate_fn,
    )

    try:
        attributor.cache(train_loader)
        with torch.no_grad():
            attributor.attribute(train_loader, test_loader)
        return True
    except (
        RuntimeError,
        torch.cuda.OutOfMemoryError,
    ):
        return False
    finally:
        if hasattr(attributor, "full_train_rep"):
            attributor.full_train_rep = None
        gc.collect()


def determine_dattri_batch_size(
    cache_root: Path,
    model_name: str,
    projection_dim: int | None,
    max_length: int,
    attributor: BaseInnerProductAttributor,
    collate_fn,
    starting_batch_size: int = 128,
    max_batch_size: int = 1024,
) -> int:
    """Find the largest batch size that fits in memory.

    Binary search: grow until OOM or max_batch_size,
    then use the last working size.
    """
    cache_file = cache_root / "dattri_batch_size_cache.jsonl"
    meta = _get_dattri_metadata(model_name, projection_dim, max_length)

    cached = _check_bs_cache(cache_file, meta)
    if cached is not None:
        print("Loaded optimal batch_size from" f" cache: {cached}")
        return cached

    print("Determining optimal batch size...")

    current = starting_batch_size
    last_working = None

    while True:
        if current < 1:
            if last_working is not None:
                current = last_working
                break
            raise RuntimeError("Could not fit batch_size=1 in memory.")

        print(
            f"Testing batch_size={current}...",
            end=" ",
            flush=True,
        )
        fits = _try_dattri_batch(attributor, current, max_length, collate_fn)

        if fits:
            print("fits", flush=True)
            last_working = current
            if current >= max_batch_size:
                break
            current = min(current * 2, max_batch_size)
        else:
            print("OOM", flush=True)
            if last_working is not None:
                current = last_working
                break
            else:
                current = current // 2

    print(f"Optimal batch_size: {current}", flush=True)
    _append_bs_cache(cache_file, meta, current)
    return current


SCHEMA_VERSION = 1
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_EVAL_SPLIT = "validation"


@dataclass
class RunConfig:
    """Configuration for a Dattri benchmark run."""

    model: str
    """Key for the model to benchmark."""

    train_tokens: str
    """Target training tokens (e.g. 1M, 10M)."""

    eval_tokens: int = 1024
    """Target evaluation tokens per sequence. Not
    analogous to train_tokens."""

    eval_sequences: int = 1
    """Target evaluation sequences."""

    batch_size: int | None = None
    """Batch size for training. Auto-tuned if not set."""

    max_length: int = 512
    """Maximum sequence length."""

    num_gpus: int = 1
    """Number of GPUs to use."""

    dataset: str = ""
    """Dataset to use for benchmarking."""

    train_split: str = DEFAULT_TRAIN_SPLIT
    """Dataset split for training."""

    eval_split: str = DEFAULT_EVAL_SPLIT
    """Dataset split for evaluation."""

    run_root: str = "runs/dattri-scaling"
    """Root directory for benchmark runs."""

    projection_dim: int | None = None
    """If set, use random projection to this dimension (like TRAK/TrackStar)."""

    run_path: str | None = None
    """Explicit run path (overrides auto-generated path)."""

    tag: str | None = None
    """Tag for the run (used in auto-generated path)."""

    notes: str | None = None
    """Optional notes for the run."""


@dataclass
class RunRecord:
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
    runtime_seconds: float | None
    start_time: str
    end_time: str
    run_path: str
    notes: str | None
    error: str | None
    num_gpus: int = 1
    projection_dim: int | None = None
    hardware: str | None = None
    gpu_name: str | None = None
    num_gpus_available: int | None = None
    gpu_vram_gb: float | None = None


@dataclass
class Run:
    """Execute a single Dattri benchmark run."""

    run_cfg: RunConfig

    def execute(self) -> None:
        """Run the benchmark."""
        if not self.run_cfg.dataset:
            self.run_cfg.dataset = str(prepare_benchmark_ds_path())

        if self.run_cfg.model not in MODEL_SPECS:
            raise ValueError(f"Unknown model '{self.run_cfg.model}'")

        spec = MODEL_SPECS[self.run_cfg.model]
        train_tokens = parse_tokens(self.run_cfg.train_tokens)
        eval_tokens = self.run_cfg.eval_tokens
        num_gpus = self.run_cfg.num_gpus

        # Set CUDA_VISIBLE_DEVICES to limit GPU usage
        if num_gpus > 0:
            visible_gpus = ",".join(str(i) for i in range(num_gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
            print(f"Using {num_gpus} GPU(s): CUDA_VISIBLE_DEVICES={visible_gpus}")

        print(
            f"Running Dattri benchmark for {self.run_cfg.model} with {train_tokens} "
            "train "
            f"and {eval_tokens} eval tokens per sequence and "
            f"{self.run_cfg.eval_sequences} "
            f"eval sequences on {num_gpus} GPU(s)"
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
                self.run_cfg.eval_sequences,
                self.run_cfg.tag,
                num_gpus,
            )
        )

        status = "success"
        error_message: str | None = None

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.cuda()

        tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        tokenizer.pad_token = tokenizer.eos_token

        # Load datasets - use load_from_disk for local paths
        dataset_path = Path(self.run_cfg.dataset)
        if dataset_path.exists():
            # Local dataset saved with save_to_disk
            full_dataset = load_from_disk(str(dataset_path))
            if isinstance(full_dataset, dict):
                train_dataset = assert_type(
                    Dataset, full_dataset[self.run_cfg.train_split]
                )
            else:
                train_dataset = assert_type(Dataset, full_dataset)
        else:
            # HuggingFace Hub dataset
            train_dataset = assert_type(
                Dataset,
                load_dataset(self.run_cfg.dataset, split=self.run_cfg.train_split),
            )

        # Estimate examples needed based on token count
        # We'll sample until we have enough tokens
        max_length = self.run_cfg.max_length or 1024
        train_examples_needed = max(1, train_tokens // max_length)
        eval_examples_needed = 1

        # Select enough examples first to avoid mapping entire dataset
        total_needed = train_examples_needed + eval_examples_needed
        train_dataset = train_dataset.select(
            range(min(total_needed, len(train_dataset)))
        )

        # Check if dataset is already tokenized or needs tokenization
        if "input_ids" in train_dataset.column_names:
            # Dataset is pretokenized - add attention_mask if missing
            def add_attention_mask(example):
                input_ids = example["input_ids"]
                # Truncate to max_length
                if len(input_ids) > self.run_cfg.max_length:
                    input_ids = input_ids[: self.run_cfg.max_length]
                attention_mask = [1] * len(input_ids)
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

            train_dataset = train_dataset.map(add_attention_mask)
        else:
            # Dataset needs tokenization
            def tokenize(batch):
                return tokenizer.batch_encode_plus(
                    batch["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.run_cfg.max_length,
                )

            train_dataset = train_dataset.map(tokenize, batched=True)

        eval_dataset = train_dataset.select(
            range(train_examples_needed, train_examples_needed + eval_examples_needed)
        )
        train_dataset = train_dataset.select(range(train_examples_needed))

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        def collate_fn(batch):
            max_len = max(len(item["input_ids"]) for item in batch)
            padded_input_ids = []
            for item in batch:
                ids = item["input_ids"]
                if len(ids) < max_len:
                    padding = torch.zeros(max_len - len(ids), dtype=ids.dtype)
                    ids = torch.cat([ids, padding])
                padded_input_ids.append(ids)
            input_ids = torch.stack(padded_input_ids)
            labels = input_ids.clone()
            return (input_ids, labels)

        # Get model device
        model_device = next(model.parameters()).device

        def loss_func(params, data_target_pair):
            x, y = data_target_pair
            if isinstance(x, torch.Tensor) and x.device != model_device:
                x = x.to(model_device)
            if isinstance(y, torch.Tensor) and y.device != model_device:
                y = y.to(model_device)
            output = torch.func.functional_call(model, params, (x,))
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output.logits if hasattr(output, "logits") else output
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = y[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            return loss

        task = AttributionTask(
            loss_func=loss_func,
            model=model,
            checkpoints=model.state_dict(),
        )

        if self.run_cfg.projection_dim is not None:
            print(
                "Using projected attributor with" f" dim={self.run_cfg.projection_dim}"
            )
            attributor = ProjectedInnerProductAttributor(
                task=task,
                proj_dim=self.run_cfg.projection_dim,
                device="cuda",
            )
        else:
            attributor = BaseInnerProductAttributor(task=task, device="cuda")

        # Determine batch size (not timed)
        if self.run_cfg.batch_size is not None:
            batch_size = self.run_cfg.batch_size
            print(f"Using fixed batch_size: {batch_size}")
        else:
            batch_size = determine_dattri_batch_size(
                cache_root=Path(".cache"),
                model_name=spec.hf_id,
                projection_dim=self.run_cfg.projection_dim,
                max_length=max_length,
                attributor=attributor,
                collate_fn=collate_fn,
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        test_loader = torch.utils.data.DataLoader(
            eval_dataset,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

        start_time = timestamp()
        start = time.perf_counter()
        attributor.cache(train_loader)

        print("Computing attributions...")
        with torch.no_grad():
            attributor.attribute(train_loader, test_loader)

        runtime = time.perf_counter() - start
        end_time = timestamp()

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
            batch_size=batch_size,
            max_length=self.run_cfg.max_length or 1024,
            num_gpus=num_gpus,
            runtime_seconds=runtime,
            start_time=start_time,
            end_time=end_time,
            run_path=str(run_path),
            notes=self.run_cfg.notes,
            error=error_message,
            projection_dim=self.run_cfg.projection_dim,
            **vars(get_hardware_details()),
        )
        save_record(run_path, record)

        print(json.dumps(asdict(record), indent=2), flush=True)

        if status != "success":
            sys.exit(1)


def load_records(root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for meta in root.rglob("benchmark.json"):
        try:
            with open(meta, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            records.append(RunRecord(**payload))
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to read {meta}: {exc}", file=sys.stderr)
    return records


@dataclass
class Main:
    """Benchmark Dattri influence analysis scaling."""

    command: Run

    def execute(self) -> None:
        """Run the selected command."""
        self.command.execute()


def get_parser() -> ArgumentParser:
    """Get the argument parser. Used for documentation generation."""
    parser = ArgumentParser(description="Benchmark Dattri influence analysis scaling")
    parser.add_arguments(Main, dest="prog")
    return parser


def main(args: Optional[list[str]] = None) -> None:
    """Parse CLI arguments and dispatch to the selected subcommand."""
    parser = get_parser()
    prog: Main = parser.parse_args(args=args).prog
    prog.execute()


if __name__ == "__main__":
    main()
