"""Utilities for benchmarking Dattri influence analysis scaling."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from dattri.algorithm.base import BaseInnerProductAttributor
from dattri.task import AttributionTask
from simple_parsing import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from same directory
from benchmarks.benchmark_utils import (
    MODEL_SPECS,
    get_run_path,
    parse_tokens,
    prepare_benchmark_ds_path,
    save_record,
    timestamp,
)
from bergson.utils.utils import assert_type

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

    batch_size: int = 4
    """Batch size for training."""

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

        def tokenize(batch):
            return tokenizer.batch_encode_plus(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.run_cfg.max_length,
            )

        # Load datasets
        train_dataset = assert_type(
            Dataset,
            load_dataset(self.run_cfg.dataset, split=self.run_cfg.train_split),
        )
        train_dataset = train_dataset.map(tokenize, batched=True)

        # Estimate examples needed based on token count
        # We'll sample until we have enough tokens
        max_length = self.run_cfg.max_length or 1024
        train_examples_needed = max(1, train_tokens // max_length)
        eval_examples_needed = 1

        # Select enough examples
        total_needed = train_examples_needed + eval_examples_needed
        train_dataset = train_dataset.select(
            range(min(total_needed, len(train_dataset)))
        )

        eval_dataset = train_dataset.select(
            range(train_examples_needed, train_examples_needed + eval_examples_needed)
        )
        train_dataset = train_dataset.select(range(train_examples_needed))

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        def collate_fn(batch):
            # Dattri expects tuples of (input_ids, labels) where
            # labels = input_ids for language modeling
            # Keep on CPU - dattri will handle device placement
            input_ids = torch.stack([item["input_ids"] for item in batch])
            labels = (
                input_ids.clone()
            )  # For language modeling, labels are the same as input_ids
            return (input_ids, labels)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,  # type: ignore
            batch_size=self.run_cfg.batch_size,
            collate_fn=collate_fn,
        )
        test_loader = torch.utils.data.DataLoader(
            eval_dataset,  # type: ignore
            batch_size=self.run_cfg.batch_size,
            collate_fn=collate_fn,
        )

        # Get model device
        model_device = next(model.parameters()).device

        def loss_func(params, data_target_pair):
            x, y = data_target_pair
            # Ensure data is on the same device as model
            if isinstance(x, torch.Tensor) and x.device != model_device:
                x = x.to(model_device)
            if isinstance(y, torch.Tensor) and y.device != model_device:
                y = y.to(model_device)
            # functional_call returns a tuple for transformers models,
            # extract logits
            output = torch.func.functional_call(model, params, (x,))
            if isinstance(output, tuple):
                logits = output[0]  # First element is logits
            else:
                logits = output.logits if hasattr(output, "logits") else output
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = y[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            return loss

        # Create task
        task = AttributionTask(
            loss_func=loss_func,
            model=model,
            checkpoints=model.state_dict(),
        )

        # Create attributor and cache
        # Try to set device if BaseInnerProductAttributor supports it
        # Remove device if this breaks
        attributor = BaseInnerProductAttributor(task=task, device="cuda")

        start_time = timestamp()
        start = time.perf_counter()
        attributor.cache(train_loader)

        # Compute attributions
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
            batch_size=self.run_cfg.batch_size,
            max_length=self.run_cfg.max_length or 1024,
            num_gpus=num_gpus,
            runtime_seconds=runtime,
            start_time=start_time,
            end_time=end_time,
            run_path=str(run_path),
            notes=self.run_cfg.notes,
            error=error_message,
        )
        save_record(run_path, record)

        print(json.dumps(asdict(record), indent=2))

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
