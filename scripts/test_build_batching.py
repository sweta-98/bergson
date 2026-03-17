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


def collect_logits(model, all_toks, batches, autocast_dtype=None):
    """Collect per-example logits for the given batch structure.

    Returns dict mapping example index -> logits tensor (seq_len, vocab).
    """
    from contextlib import nullcontext

    from bergson.data import pad_and_tensor

    device = next(model.parameters()).device
    ctx = torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype else nullcontext()
    result = {}
    with torch.no_grad(), ctx:
        for batch in batches:
            toks_list = [all_toks[i] for i in batch]
            x, _, _ = pad_and_tensor(toks_list, device=device)
            logits = model(x).logits  # (B, S, V)
            for j, idx in enumerate(batch):
                seq_len = len(all_toks[idx])
                result[idx] = logits[j, :seq_len].float().cpu()
    return result


def collect_grads(model, data, batches, precision="bf16", projection_dim=16, autocast=False):
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
        autocast=autocast,
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
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32", "fp16", "autocast_bf16", "tf32"])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--projection_dim", type=int, default=16)
    parser.add_argument("--mode", default="batched_vs_unbatched",
                        choices=["batched_vs_unbatched", "batched_vs_batched"])
    args = parser.parse_args()

    n = args.n
    pdim = args.projection_dim
    print(f"Model: {MODEL}")
    print(f"precision={args.precision}, N={n}, proj_dim={pdim}, mode={args.mode}")

    if args.precision == "bf16":
        model_dtype = torch.bfloat16
    elif args.precision == "fp16":
        model_dtype = torch.float16
    elif args.precision == "tf32":
        model_dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:  # fp32 and autocast_bf16 both load in fp32
        model_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=model_dtype, device_map="cuda",
    )

    # Map to bergson precision string
    if args.precision == "autocast_bf16":
        build_precision = "bf16"
    elif args.precision == "tf32":
        build_precision = "fp32"
    else:
        build_precision = args.precision
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

    autocast_dtype = torch.bfloat16 if args.precision == "autocast_bf16" else None
    use_autocast = args.precision == "autocast_bf16"

    if args.mode == "batched_vs_unbatched":
        print("\nCollecting with batching...")
        a = collect_grads(model, data, batched_batches, build_precision, pdim, autocast=use_autocast)
        print("Collecting without batching...")
        b = collect_grads(model, data, unbatched_batches, build_precision, pdim, autocast=use_autocast)
        label = f"{args.precision} batched vs unbatched (N={n}, proj_dim={pdim})"
    else:
        print("\nCollecting with batching (run 1)...")
        a = collect_grads(model, data, batched_batches, build_precision, pdim, autocast=use_autocast)
        print("Collecting with batching (run 2)...")
        b = collect_grads(model, data, batched_batches, build_precision, pdim, autocast=use_autocast)
        label = f"{args.precision} batched run 1 vs run 2 (N={n}, proj_dim={pdim})"

    cosines = np.array([cosine(a[i], b[i]) for i in range(n)])
    print_cosine_stats(cosines, label)

    # Logit comparison
    if args.mode == "batched_vs_unbatched":
        print("\nComparing logits (batched vs unbatched)...")
        logits_batched = collect_logits(model, all_toks, batched_batches, autocast_dtype)
        logits_unbatched = collect_logits(model, all_toks, unbatched_batches, autocast_dtype)

        logit_cosines = []
        logit_max_diffs = []
        for i in range(n):
            lb = logits_batched[i].numpy().flatten()
            lu = logits_unbatched[i].numpy().flatten()
            logit_cosines.append(cosine(lb, lu))
            logit_max_diffs.append(np.abs(lb - lu).max())
        logit_cosines = np.array(logit_cosines)
        logit_max_diffs = np.array(logit_max_diffs)

        print_cosine_stats(logit_cosines, "Logit cosine: batched vs unbatched")

        print(f"\n  Logit max abs diff: min={logit_max_diffs.min():.6e} "
              f"mean={logit_max_diffs.mean():.6e} max={logit_max_diffs.max():.6e}")

        # Show correlation between gradient outliers and logit outliers
        grad_worst = np.argsort(cosines)[:10]
        print(f"\n  Gradient worst 10 vs their logit cosine:")
        print(f"  {'idx':>5} {'grad_cos':>10} {'logit_cos':>10} {'logit_maxdiff':>14} {'len':>5}")
        for i in grad_worst:
            print(f"  {i:>5} {cosines[i]:>10.6f} {logit_cosines[i]:>10.6f} "
                  f"{logit_max_diffs[i]:>14.6e} {lengths[i]:>5}")


if __name__ == "__main__":
    main()
