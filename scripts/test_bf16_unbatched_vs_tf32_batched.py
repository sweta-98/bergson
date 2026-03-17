#!/usr/bin/env python3
"""Benchmark bf16 skip_batched vs tf32 batched vs bf16 batched.

Compares speed and accuracy of gradient collection strategies.
All timing uses InMemoryCollector with model already loaded.
Each config gets a warmup run before the timed run.

Usage::
    CUDA_VISIBLE_DEVICES=0 python scripts/test_bf16_unbatched_vs_tf32_batched.py
    CUDA_VISIBLE_DEVICES=0 python scripts/test_bf16_unbatched_vs_tf32_batched.py --n 1000
"""

import argparse
import os
import tempfile
import time

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


def collect_grads(model, data, batches, precision="bf16", projection_dim=16):
    run_path = tempfile.mkdtemp()
    os.makedirs(run_path + ".part", exist_ok=True)
    cfg = IndexConfig(
        run_path=run_path, precision=precision, projection_dim=projection_dim,
        skip_preconditioners=True, skip_index=True, overwrite=True,
    )
    preprocess = PreprocessConfig(
        unit_normalize=True, aggregation="none", normalize_aggregated_grad=False,
    )
    processor = GradientProcessor(projection_dim=projection_dim)
    collector = InMemoryCollector(
        model=model, data=data, cfg=cfg, preprocess_cfg=preprocess, processor=processor,
    )
    computer = CollectorComputer(
        model=model, data=data, collector=collector, batches=batches, cfg=cfg,
    )

    torch.cuda.synchronize()
    t0 = time.monotonic()
    computer.run_with_collector_hooks()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - t0

    grads = []
    for name in sorted(collector.gradients.keys()):
        grads.append(collector.gradients[name].float().cpu())
    return torch.cat(grads, dim=1).numpy(), elapsed


def print_stats(cosines, label):
    print(f"  {label}")
    print(f"    min={cosines.min():.6f}  mean={cosines.mean():.6f}  max={cosines.max():.6f}")
    n_below_99 = (cosines < 0.99).sum()
    n_below_999 = (cosines < 0.999).sum()
    if n_below_99 > 0:
        print(f"    <0.99: {n_below_99} ({n_below_99/len(cosines)*100:.1f}%)")
    if n_below_999 > 0:
        print(f"    <0.999: {n_below_999} ({n_below_999/len(cosines)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--projection_dim", type=int, default=16)
    args = parser.parse_args()

    n = args.n
    pdim = args.projection_dim

    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("NeelNanda/pile-10k", split=f"train[:{n}]")
    all_toks = [tok(ds[i]["text"], truncation=True, max_length=1024)["input_ids"]
                for i in range(n)]
    lengths = [len(t) for t in all_toks]
    data = Dataset.from_dict({"input_ids": all_toks})

    batched = allocate_batches(lengths, 1024)
    unbatched = [[i] for i in range(n)]
    total_tokens = sum(lengths)

    # Small subset for warmup
    warmup_data = Dataset.from_dict({"input_ids": all_toks[:10]})
    warmup_batches = allocate_batches(lengths[:10], 1024)

    print(f"Model: {MODEL}")
    print(f"N={n}, proj_dim={pdim}, total_tokens={total_tokens}")
    print(f"Batched: {len(batched)} batches (sizes {min(len(b) for b in batched)}-{max(len(b) for b in batched)})")

    results = {}
    timings = {}

    # ── bf16 ──
    print("\nLoading bf16 model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")

    print("  warmup...")
    collect_grads(model, warmup_data, warmup_batches, "bf16", pdim)

    print("  bf16 batched...")
    results["bf16_batched"], timings["bf16_batched"] = collect_grads(model, data, batched, "bf16", pdim)

    print("  bf16 unbatched...")
    results["bf16_unbatched"], timings["bf16_unbatched"] = collect_grads(model, data, unbatched, "bf16", pdim)

    del model
    torch.cuda.empty_cache()

    # ── tf32 ──
    print("\nLoading tf32 model...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, device_map="cuda")

    print("  warmup...")
    collect_grads(model, warmup_data, warmup_batches, "fp32", pdim)

    print("  tf32 batched...")
    results["tf32_batched"], timings["tf32_batched"] = collect_grads(model, data, batched, "fp32", pdim)

    print("  tf32 unbatched...")
    results["tf32_unbatched"], timings["tf32_unbatched"] = collect_grads(model, data, unbatched, "fp32", pdim)

    del model
    torch.cuda.empty_cache()

    # ── Results ──
    print(f"\n{'='*60}")
    print("Speed:")
    print(f"  {'config':<20s} {'time':>8s} {'tok/s':>10s} {'vs bf16 ub':>10s}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10}")
    ref_time = timings["bf16_unbatched"]
    for name in ["bf16_batched", "bf16_unbatched", "tf32_batched", "tf32_unbatched"]:
        t = timings[name]
        speedup = ref_time / t
        print(f"  {name:<20s} {t:>7.1f}s {total_tokens/t:>9.0f} {speedup:>9.2f}x")

    print(f"\nAccuracy vs bf16 unbatched (ground truth):")
    for name in ["bf16_batched", "tf32_batched", "tf32_unbatched"]:
        cosines = np.array([cosine(results["bf16_unbatched"][i], results[name][i]) for i in range(n)])
        print_stats(cosines, name)

    print(f"\nSelf-consistency (batched vs unbatched within precision):")
    for prec, a_name, b_name in [
        ("bf16", "bf16_batched", "bf16_unbatched"),
        ("tf32", "tf32_batched", "tf32_unbatched"),
    ]:
        cosines = np.array([cosine(results[a_name][i], results[b_name][i]) for i in range(n)])
        print_stats(cosines, f"{prec}: batched vs unbatched")


if __name__ == "__main__":
    main()
