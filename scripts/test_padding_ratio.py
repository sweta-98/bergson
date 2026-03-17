#!/usr/bin/env python3
"""Measure gradient cosine sim as a function of padding ratio.

Creates synthetic batches where one item is always 1024 tokens and the
other varies from 10% to 100% of 1024. Measures cosine between the
shorter item's gradient when batched vs processed alone.

Usage::
    python scripts/test_padding_ratio.py
"""

import os
import tempfile

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.collector.collector import CollectorComputer
from bergson.collector.in_memory_collector import InMemoryCollector
from bergson.config import IndexConfig, PreprocessConfig
from bergson.gradients import GradientProcessor

MODEL = "allenai/OLMo-2-1124-7B-Instruct"
MAX_LEN = 1024
PROJECTION_DIM = 16


def collect_grads(model, data, batches):
    run_path = tempfile.mkdtemp()
    os.makedirs(run_path + ".part", exist_ok=True)

    cfg = IndexConfig(
        run_path=run_path,
        precision="bf16",
        projection_dim=PROJECTION_DIM,
        skip_preconditioners=True,
        skip_index=True,
        overwrite=True,
    )
    preprocess = PreprocessConfig(
        unit_normalize=True,
        aggregation="none",
        normalize_aggregated_grad=False,
    )
    processor = GradientProcessor(projection_dim=PROJECTION_DIM)
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
    computer.run_with_collector_hooks(desc=None)

    all_grads = []
    for name in sorted(collector.gradients.keys()):
        all_grads.append(collector.gradients[name].float().cpu())
    return torch.cat(all_grads, dim=1).numpy()


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def main():
    print(f"Model: {MODEL}, proj_dim={PROJECTION_DIM}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Get a pool of token IDs from real text
    from datasets import load_dataset
    pile = load_dataset("NeelNanda/pile-10k", split="train[:5]")
    all_tokens = []
    for text in pile["text"]:
        ids = tokenizer(text, truncation=False)["input_ids"]
        all_tokens.extend(ids)

    # The "anchor" item is always MAX_LEN tokens
    anchor_tokens = all_tokens[:MAX_LEN]

    ratios = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]

    print(f"\n{'ratio':>6} {'short_len':>10} {'pad_tokens':>11} {'cosine':>10}")
    print("-" * 42)

    for ratio in ratios:
        short_len = max(2, int(MAX_LEN * ratio))
        short_tokens = all_tokens[MAX_LEN:MAX_LEN + short_len]

        data = Dataset.from_dict({"input_ids": [anchor_tokens, short_tokens]})

        # Unbatched: each item alone
        unbatched = collect_grads(model, data, batches=[[0], [1]])

        # Batched: both items together
        batched = collect_grads(model, data, batches=[[0, 1]])

        # Compare the short item (index 1)
        cos = cosine(unbatched[1], batched[1])
        pad_tokens = MAX_LEN - short_len

        print(f"{ratio:>6.2f} {short_len:>10} {pad_tokens:>11} {cos:>10.6f}")


if __name__ == "__main__":
    main()
