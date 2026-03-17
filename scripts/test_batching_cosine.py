#!/usr/bin/env python3
"""Test per-example gradient cosine similarity: batched vs unbatched.

Runs the same forward+backward as bergson (but without hooks) on two
1024-token sequences, comparing per-example parameter gradients when
processed separately vs together in a batch of 2.

Usage::

    python scripts/test_batching_cosine.py
    python scripts/test_batching_cosine.py --model EleutherAI/pythia-14m
    python scripts/test_batching_cosine.py --precision fp32
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.data import pad_and_tensor


def get_per_example_grads_separate(model, toks):
    """Forward+backward each example individually, return per-example param grads."""
    results = []
    for ids in toks:
        model.zero_grad()
        x, y, _ = pad_and_tensor([ids], device=next(model.parameters()).device)
        logits = model(x).logits[:, :-1]
        masks = y[:, 1:] != -100
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y[:, 1:].flatten(),
            reduction="none",
        ).reshape_as(y[:, 1:])
        loss = (loss * masks).sum(1) / masks.sum(1, dtype=model.dtype)
        loss.sum().backward()
        grad = torch.cat(
            [p.grad.flatten() for p in model.parameters() if p.grad is not None]
        ).float().cpu()
        model.zero_grad()
        results.append(grad.numpy())
    return results


def get_per_example_grads_batched(model, toks):
    """Forward all examples in one batch, backward each loss separately."""
    model.zero_grad()
    x, y, _ = pad_and_tensor(toks, device=next(model.parameters()).device)
    logits = model(x).logits[:, :-1]
    masks = y[:, 1:] != -100
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y[:, 1:].flatten(),
        reduction="none",
    ).reshape_as(y[:, 1:])
    losses = (loss * masks).sum(1) / masks.sum(1, dtype=model.dtype)

    results = []
    for i in range(len(toks)):
        model.zero_grad()
        losses[i].backward(retain_graph=(i < len(toks) - 1))
        grad = torch.cat(
            [p.grad.flatten() for p in model.parameters() if p.grad is not None]
        ).float().cpu()
        results.append(grad.numpy())
    model.zero_grad()
    return results


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-160m")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--num_examples", type=int, default=2)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="cuda",
        attn_implementation=args.attn,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ds = load_dataset("NeelNanda/pile-10k", split=f"train[:{args.num_examples}]")
    toks = [
        tokenizer(t, truncation=True, max_length=1024, return_attention_mask=False)["input_ids"]
        for t in ds["text"]
    ]
    print(f"Model: {args.model}, precision: {args.precision}, attn: {args.attn}")
    print(f"Sequence lengths: {[len(t) for t in toks]}")
    print(f"Param count: {sum(p.numel() for p in model.parameters()):,}")

    sep = get_per_example_grads_separate(model, toks)
    batched = get_per_example_grads_batched(model, toks)

    # Also run sep twice for determinism baseline
    sep2 = get_per_example_grads_separate(model, toks)

    print(f"\n{'Comparison':<35s} ", end="")
    for i in range(len(toks)):
        print(f"{'ex ' + str(i):>10s}", end="")
    print()

    print(f"{'sep vs sep (determinism)':<35s} ", end="")
    for i in range(len(toks)):
        print(f"{cosine(sep[i], sep2[i]):>10.6f}", end="")
    print()

    print(f"{'sep vs batched':<35s} ", end="")
    for i in range(len(toks)):
        print(f"{cosine(sep[i], batched[i]):>10.6f}", end="")
    print()


if __name__ == "__main__":
    main()
