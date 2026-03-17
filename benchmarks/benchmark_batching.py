"""Benchmark skip_batching (batch_size=1) vs token_batch_size batching.

Uses in-memory gradient collection (no disk I/O) to isolate the
batching overhead. Compares wall time and gradient quality.

Usage::

    python -m benchmarks.benchmark_batching pythia-14m runs/bench_batching
    python -m benchmarks.benchmark_batching pythia-14m runs/bench_batching --token_batch_sizes 1024 2048 4096
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from simple_parsing import ArgumentParser, field

from benchmarks.benchmark_utils import (
    MODEL_SPECS,
    get_hardware_details,
    prepare_benchmark_ds_path,
)
from bergson.collector.collector import CollectorComputer
from bergson.collector.in_memory_collector import InMemoryCollector
from bergson.config import DataConfig, IndexConfig, PreprocessConfig
from bergson.data import allocate_batches
from bergson.gradients import GradientProcessor
from bergson.utils.worker_utils import setup_data_pipeline, setup_model_and_peft


@dataclass
class BatchBenchConfig:
    """Configuration for the batching benchmark."""

    model: str = field(positional=True)
    """Model key (e.g., pythia-14m, pythia-160m)."""

    run_root: str = field(positional=True)
    """Root directory for benchmark outputs."""

    num_examples: int = 100
    """Number of examples to collect gradients for."""

    max_length: int = 1024
    """Maximum sequence length."""

    projection_dim: int = 16
    """Projection dimension per module."""

    token_batch_sizes: list[int] = field(default_factory=lambda: [1024, 2048, 4096])
    """Token batch sizes to benchmark."""

    precision: str = "bf16"
    """Model precision."""

    warmup_steps: int = 5
    """Number of warmup forward/backward passes before timing."""

    dataset: str = ""
    """Dataset path. If empty, uses the default benchmark dataset."""


PREPROCESS = PreprocessConfig(
    unit_normalize=True,
    aggregation="none",
    normalize_aggregated_grad=False,
)


def _collect_grads(model, ds, index_cfg, batches, projection_dim):
    """Collect gradients in-memory and return (flat_grads_np, elapsed_seconds)."""
    proj_dim = projection_dim if projection_dim > 0 else None

    collector = InMemoryCollector(
        model=model.base_model,
        processor=GradientProcessor(projection_dim=proj_dim),
        data=ds,
        cfg=index_cfg,
        preprocess_cfg=PREPROCESS,
    )

    computer = CollectorComputer(
        model=model,
        data=ds,
        collector=collector,
        batches=batches,
        cfg=index_cfg,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    computer.run_with_collector_hooks(desc="benchmark")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    all_grads = []
    for name in sorted(collector.gradients.keys()):
        all_grads.append(collector.gradients[name].float().cpu())
    grads = torch.cat(all_grads, dim=1).numpy()

    return grads, elapsed


def _cosine_sim(a, b):
    na = np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-10)
    nb = np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-10)
    return np.sum((a / na) * (b / nb), axis=1)


def _relative_l2(a, b):
    l2 = np.linalg.norm(a - b, axis=1)
    norms = (np.linalg.norm(a, axis=1) + np.linalg.norm(b, axis=1)) / 2
    return l2 / norms.clip(min=1e-10)


def main():
    parser = ArgumentParser()
    parser.add_arguments(BatchBenchConfig, dest="cfg")
    args = parser.parse_args()
    cfg: BatchBenchConfig = args.cfg

    if cfg.model not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{cfg.model}'. Choose from: {list(MODEL_SPECS)}")

    spec = MODEL_SPECS[cfg.model]
    if not cfg.dataset:
        cfg.dataset = str(prepare_benchmark_ds_path())

    run_root = Path(cfg.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "tmp.part").mkdir(parents=True, exist_ok=True)

    index_cfg = IndexConfig(
        run_path=str(run_root / "tmp"),
        model=spec.hf_id,
        data=DataConfig(
            dataset=cfg.dataset,
            split=f"train[:{cfg.num_examples}]",
            prompt_column="text",
            truncation=True,
        ),
        precision=cfg.precision,
        projection_dim=cfg.projection_dim,
        token_batch_size=cfg.max_length,
        skip_preconditioners=True,
        overwrite=True,
    )

    print(f"Loading model {spec.hf_id}...")
    model, _ = setup_model_and_peft(index_cfg, device_map_auto=True)
    from datasets import Dataset as HFDataset
    ds = setup_data_pipeline(index_cfg)
    assert isinstance(ds, HFDataset)

    lengths = ds["length"][:]
    total_tokens = sum(lengths)
    print(f"Dataset: {len(ds)} examples, {total_tokens} tokens")
    print(f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={total_tokens/len(lengths):.0f}")

    # Warmup
    print(f"\nWarming up ({cfg.warmup_steps} steps)...")
    warmup_ds = ds.select(range(min(cfg.warmup_steps, len(ds))))
    warmup_batches = [[i] for i in range(len(warmup_ds))]
    _collect_grads(model, warmup_ds, index_cfg, warmup_batches, cfg.projection_dim)

    results = []

    # skip_batching (batch_size=1)
    print("\n" + "=" * 70)
    print("skip_batching=True (batch_size=1, no padding)")
    print("=" * 70)
    bs1_batches = [[i] for i in range(len(ds))]
    grads_ref, time_ref = _collect_grads(
        model, ds, index_cfg, bs1_batches, cfg.projection_dim,
    )
    tps_ref = total_tokens / time_ref
    print(f"  Time: {time_ref:.2f}s  ({tps_ref:.0f} tokens/s)  Shape: {grads_ref.shape}")
    results.append({
        "method": "skip_batching",
        "token_batch_size": None,
        "time_seconds": time_ref,
        "tokens_per_second": tps_ref,
    })

    # Token batch sizes
    for tbs in cfg.token_batch_sizes:
        print(f"\n{'=' * 70}")
        print(f"token_batch_size={tbs}")
        print("=" * 70)

        batches = allocate_batches(lengths, tbs)
        grads_tbs, time_tbs = _collect_grads(
            model, ds, index_cfg, batches, cfg.projection_dim,
        )
        tps = total_tokens / time_tbs
        speedup = time_ref / time_tbs

        cosine = _cosine_sim(grads_ref, grads_tbs)
        rel_l2 = _relative_l2(grads_ref, grads_tbs)

        print(f"  Time: {time_tbs:.2f}s  ({tps:.0f} tokens/s)  {speedup:.2f}x speedup")
        print(f"  Batches: {len(batches)}")
        print(f"  vs skip_batching:")
        print(f"    Cosine sim:   min={cosine.min():.8f}  mean={cosine.mean():.8f}")
        print(f"    Relative L2:  max={rel_l2.max():.6e}  mean={rel_l2.mean():.6e}")

        results.append({
            "method": f"token_batch_{tbs}",
            "token_batch_size": tbs,
            "time_seconds": time_tbs,
            "tokens_per_second": tps,
            "speedup_vs_skip_batching": speedup,
            "num_batches": len(batches),
            "cosine_min": float(cosine.min()),
            "cosine_mean": float(cosine.mean()),
            "rel_l2_max": float(rel_l2.max()),
            "rel_l2_mean": float(rel_l2.mean()),
        })

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Method':<20} {'Time':>8} {'Tok/s':>10} {'Speedup':>8} {'Cos min':>10} {'RelL2 max':>10}")
    print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*8} {'─'*10} {'─'*10}")
    for r in results:
        t = f"{r['time_seconds']:.2f}s"
        tps = f"{r['tokens_per_second']:.0f}"
        sp = f"{r.get('speedup_vs_skip_batching', 1.0):.2f}x"
        cos = f"{r.get('cosine_min', 1.0):.8f}"
        rl2 = f"{r.get('rel_l2_max', 0.0):.6e}"
        print(f"  {r['method']:<20} {t:>8} {tps:>10} {sp:>8} {cos:>10} {rl2:>10}")

    # Save
    hw = get_hardware_details()
    output = {
        "model": cfg.model,
        "model_name": spec.hf_id,
        "num_examples": cfg.num_examples,
        "total_tokens": total_tokens,
        "projection_dim": cfg.projection_dim,
        "precision": cfg.precision,
        "hardware": hw.__dict__ if hasattr(hw, '__dict__') else str(hw),
        "results": results,
    }
    out_path = run_root / f"batching_benchmark_{cfg.model}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
