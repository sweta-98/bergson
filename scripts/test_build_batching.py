#!/usr/bin/env python3
"""Measure gradient nondeterminism from batching in bergson.

Compares per-example projected gradients when examples are processed
with token-length batching vs batch size 1 (skip_batching).

Usage::
    # Quick test (20 examples)
    CUDA_VISIBLE_DEVICES=0 python scripts/test_build_batching.py

    # Full reproduction of outlier cosine similarities
    CUDA_VISIBLE_DEVICES=0 python scripts/test_build_batching.py --n 1000 --projection_dim 16

    # Determinism check (batched vs batched)
    CUDA_VISIBLE_DEVICES=0 python scripts/test_build_batching.py --mode batched_vs_batched
"""

import argparse
import os
import tempfile

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.build import allocate_batches
from bergson.collector.collector import CollectorComputer
from bergson.collector.in_memory_collector import InMemoryCollector
from bergson.config import IndexConfig, PreprocessConfig
from bergson.gradients import GradientProcessor

MODEL = "allenai/OLMo-2-1124-7B-Instruct"


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def print_cosine_stats(cosines, label):
    print(f"\n{label}")
    print(f"  min cosine:  {cosines.min():.8f}")
    print(f"  mean cosine: {cosines.mean():.8f}")
    print(f"  max cosine:  {cosines.max():.8f}")

    bins = [0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0001]
    counts, _ = np.histogram(cosines, bins=bins)
    print(f"  Distribution:")
    for i in range(len(counts)):
        if counts[i] > 0:
            print(f"    [{bins[i]:.4f}, {bins[i+1]:.4f}): {counts[i]:>5d} ({counts[i]/len(cosines)*100:.1f}%)")

    worst = np.argsort(cosines)[:5]
    print(f"  Worst 5:")
    for i in worst:
        print(f"    idx={i} cosine={cosines[i]:.6f}")


def collect_grads(model, data, batches, precision="bf16", projection_dim=16):
    """Collect per-example projected gradients via InMemoryCollector."""
    run_path = tempfile.mkdtemp()
    os.makedirs(run_path + ".part", exist_ok=True)

    cfg = IndexConfig(
        run_path=run_path,
        precision=precision,
        projection_dim=projection_dim,
        skip_preconditioners=True,
        skip_index=True,
        overwrite=True,
    )
    preprocess = PreprocessConfig(
        unit_normalize=True,
        aggregation="none",
        normalize_aggregated_grad=False,
    )
    processor = GradientProcessor(projection_dim=projection_dim)
    collector = InMemoryCollector(
        model=model,
        data=data,
        cfg=cfg,
        preprocess_cfg=preprocess,
        processor=processor,
    )
    computer = CollectorComputer(
        model=model,
        data=data,
        collector=collector,
        batches=batches,
        cfg=cfg,
    )
    computer.run_with_collector_hooks()

    grads = []
    for name in sorted(collector.gradients.keys()):
        grads.append(collector.gradients[name].float().cpu())
    return torch.cat(grads, dim=1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32", "fp16"])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--projection_dim", type=int, default=16)
    parser.add_argument("--mode", default="batched_vs_unbatched",
                        choices=["batched_vs_unbatched", "batched_vs_batched"])
    args = parser.parse_args()

    n = args.n
    pdim = args.projection_dim
    print(f"Model: {MODEL}")
    print(f"precision={args.precision}, N={n}, proj_dim={pdim}, mode={args.mode}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32,
        device_map="cuda",
    )
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("NeelNanda/pile-10k", split=f"train[:{n}]")
    all_toks = [tok(ds[i]["text"], truncation=True, max_length=1024)["input_ids"]
                for i in range(n)]
    lengths = [len(t) for t in all_toks]
    data = Dataset.from_dict({"input_ids": all_toks})

    batched_batches = allocate_batches(lengths, 1024)
    unbatched_batches = [[i] for i in range(n)]

    print(f"Batched: {len(batched_batches)} batches, "
          f"sizes {min(len(b) for b in batched_batches)}-{max(len(b) for b in batched_batches)}")

    if args.mode == "batched_vs_unbatched":
        print("\nCollecting with batching...")
        a = collect_grads(model, data, batched_batches, args.precision, pdim)
        print("Collecting without batching...")
        b = collect_grads(model, data, unbatched_batches, args.precision, pdim)
        label = f"{args.precision} batched vs unbatched (N={n}, proj_dim={pdim})"
    else:
        print("\nCollecting with batching (run 1)...")
        a = collect_grads(model, data, batched_batches, args.precision, pdim)
        print("Collecting with batching (run 2)...")
        b = collect_grads(model, data, batched_batches, args.precision, pdim)
        label = f"{args.precision} batched run 1 vs run 2 (N={n}, proj_dim={pdim})"

    cosines = np.array([cosine(a[i], b[i]) for i in range(n)])
    print_cosine_stats(cosines, label)


if __name__ == "__main__":
    main()
