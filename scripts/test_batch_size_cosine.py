#!/usr/bin/env python3
"""Measure gradient cosine sim as a function of batch size.

Fixes one target sequence and batches it with N other sequences of the
same length (no padding). Compares the target's gradient when batched
vs processed alone.

Usage::
    python scripts/test_batch_size_cosine.py
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
SEQ_LEN = 512
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
    computer.run_with_collector_hooks()

    all_grads = []
    for name in sorted(collector.gradients.keys()):
        all_grads.append(collector.gradients[name].float().cpu())
    return torch.cat(all_grads, dim=1).numpy()


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def main():
    print(f"Model: {MODEL}, seq_len={SEQ_LEN}, proj_dim={PROJECTION_DIM}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Get token IDs from real text
    from datasets import load_dataset
    pile = load_dataset("NeelNanda/pile-10k", split="train[:50]")
    all_tokens = []
    for text in pile["text"]:
        ids = tokenizer(text, truncation=False)["input_ids"]
        all_tokens.extend(ids)

    # Create fixed-length sequences
    max_seqs = 20
    seqs = []
    for i in range(max_seqs + 1):
        start = i * SEQ_LEN
        seqs.append(all_tokens[start:start + SEQ_LEN])

    # Target is always seqs[0]
    # Baseline: target processed alone
    data_alone = Dataset.from_dict({"input_ids": [seqs[0]]})
    alone_grads = collect_grads(model, data_alone, batches=[[0]])
    target_alone = alone_grads[0]

    batch_sizes = [1, 2, 3, 4, 5, 8, 10, 15, 20]

    print(f"\n{'batch_size':>10} {'cosine':>10}")
    print("-" * 22)

    for n in batch_sizes:
        # Target + (n-1) other sequences, all same length
        batch_seqs = [seqs[0]] + seqs[1:n]
        data = Dataset.from_dict({"input_ids": batch_seqs})

        # All in one batch
        batched_grads = collect_grads(model, data, batches=[list(range(n))])
        target_batched = batched_grads[0]

        cos = cosine(target_alone, target_batched)
        print(f"{n:>10} {cos:>10.6f}")


if __name__ == "__main__":
    main()
