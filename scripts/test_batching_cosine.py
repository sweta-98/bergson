#!/usr/bin/env python3
"""Test per-example gradient cosine similarity: batched vs unbatched.

Runs the same forward+backward as bergson (but without hooks) on two
1024-token sequences, comparing per-example parameter gradients when
processed separately vs together in a batch of 2.

Usage::

    python scripts/test_batching_cosine.py
    python scripts/test_batching_cosine.py --precision fp32
    python scripts/test_batching_cosine.py --precision fp16
    python scripts/test_batching_cosine.py --precision autocast_bf16
"""

import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.data import pad_and_tensor

PRECISIONS = ["bf16", "fp16", "fp32", "autocast_bf16", "autocast_fp16"]


def _get_dtype_and_ctx(precision):
    """Return (model_dtype, autocast_context_factory)."""
    if precision == "bf16":
        return torch.bfloat16, nullcontext
    elif precision == "fp16":
        return torch.float16, nullcontext
    elif precision == "fp32":
        return torch.float32, nullcontext
    elif precision == "autocast_bf16":
        return torch.float32, lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    elif precision == "autocast_fp16":
        return torch.float32, lambda: torch.autocast("cuda", dtype=torch.float16)
    else:
        raise ValueError(f"Unknown precision: {precision}")


def _fwd_bwd(model, toks_list, ctx_factory):
    """Forward+backward, return per-param grad dict."""
    model.zero_grad()
    device = next(model.parameters()).device
    x, y, _ = pad_and_tensor(toks_list, device=device)
    with ctx_factory():
        logits = model(x).logits[:, :-1]
        masks = y[:, 1:] != -100
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y[:, 1:].flatten(),
            reduction="none",
        ).reshape_as(y[:, 1:])
        loss = (loss * masks).sum(1) / masks.sum(1, dtype=torch.float32)
    loss.sum().backward()
    # Collect grads per-module to avoid OOM on large models
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.flatten().float().cpu()
    model.zero_grad()
    return grads


def cosine_of_grad_dicts(a, b):
    """Compute cosine similarity between two per-param gradient dicts."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for name in a:
        if name not in b:
            continue
        ga, gb = a[name].numpy(), b[name].numpy()
        dot += np.dot(ga, gb)
        norm_a += np.dot(ga, ga)
        norm_b += np.dot(gb, gb)
    return dot / (np.sqrt(norm_a) * np.sqrt(norm_b) + 1e-10)


def _collect_bergson_grads(model, data, precision, projection_dim=8, batches=None):
    """Collect per-example projected gradients via InMemoryCollector."""
    import os
    import tempfile

    from bergson.collector.collector import CollectorComputer
    from bergson.collector.in_memory_collector import InMemoryCollector
    from bergson.config import IndexConfig, PreprocessConfig
    from bergson.gradients import GradientProcessor

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
    computer.run_with_collector_hooks(desc="bergson collection")

    all_grads = []
    for name in sorted(collector.gradients.keys()):
        all_grads.append(collector.gradients[name].float().cpu())
    return torch.cat(all_grads, dim=1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    parser.add_argument("--precision", default="bf16", choices=PRECISIONS)
    parser.add_argument("--attn", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable torch deterministic algorithms")
    parser.add_argument("--disable_half_reduction", action="store_true",
                        help="Disable bf16/fp16 reduced precision reduction in matmuls")
    parser.add_argument("--tf32", action="store_true",
                        help="Enable TF32 for matmul and cuDNN")
    args = parser.parse_args()

    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.disable_half_reduction or args.deterministic:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype, ctx_factory = _get_dtype_and_ctx(args.precision)
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
    det_str = " deterministic" if args.deterministic else ""
    print(f"Model: {args.model}, precision: {args.precision}, attn: {args.attn}{det_str}")
    print(f"Sequence lengths: {[len(t) for t in toks]}")

    # ── Raw gradients (no projection) ──
    sep_grads = [_fwd_bwd(model, [t], ctx_factory) for t in toks]
    batched_grads = _fwd_bwd(model, toks, ctx_factory)
    sep0_again = _fwd_bwd(model, [toks[0]], ctx_factory)

    sum_sep = {}
    for name in sep_grads[0]:
        sum_sep[name] = sum(g[name] for g in sep_grads)

    print(f"\n{'Comparison':<50s} {'cosine':>10s}")
    print("-" * 62)
    print(f"{'raw: sep[0] vs sep[0] (determinism)':<50s} {cosine_of_grad_dicts(sep_grads[0], sep0_again):>10.6f}")
    print(f"{'raw: sum(sep) vs batched':<50s} {cosine_of_grad_dicts(sum_sep, batched_grads):>10.6f}")

    # ── Bergson projected gradients ──
    from datasets import Dataset as HFDataset
    from bergson.build import allocate_batches

    hf_data = HFDataset.from_dict({"input_ids": toks})
    lengths = [len(t) for t in toks]

    # Batch size 1 (each example separate)
    bs1_batches = [[i] for i in range(len(toks))]
    bergson_bs1 = _collect_bergson_grads(model, hf_data, args.precision, batches=bs1_batches)

    # Token-length batching
    token_batches = allocate_batches(lengths, 2048)
    bergson_batched = _collect_bergson_grads(model, hf_data, args.precision, batches=token_batches)

    # Batch size 1 again (determinism)
    bergson_bs1_again = _collect_bergson_grads(model, hf_data, args.precision, batches=bs1_batches)

    # Per-example cosine sims
    for i in range(len(toks)):
        a, b = bergson_bs1[i], bergson_batched[i]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        print(f"{'bergson: bs1[%d] vs batched[%d]' % (i, i):<50s} {cos:>10.6f}")

    for i in range(len(toks)):
        a, b = bergson_bs1[i], bergson_bs1_again[i]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        print(f"{'bergson: bs1[%d] vs bs1[%d] (determinism)' % (i, i):<50s} {cos:>10.6f}")


if __name__ == "__main__":
    main()
