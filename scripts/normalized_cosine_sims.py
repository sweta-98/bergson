#!/usr/bin/env python3
"""Compute normalized gradient cosine sims with different normalizers.

Loads per-example gradients from a bergson build, applies normalizers,
and computes cosine similarity against the mean query gradient.
Reports AUROC for forget/retain and retain/pile separation.

Usage:
    python scripts/normalized_cosine_sims.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bergson.data import load_gradients
from bergson.gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    GradientProcessor,
    Normalizer,
)


def load_grad_index(path: str):
    """Load gradient index as numpy memmap."""
    import json
    info = json.load(open(Path(path) / "info.json"))
    grads = load_gradients(path, structured=False)
    return grads, info


def apply_normalizer_to_grads(
    grads: np.ndarray,
    normalizers: dict[str, Normalizer],
    grad_sizes: dict[str, int],
) -> np.ndarray:
    """Apply normalizers to flat gradient array, module by module."""
    result = np.zeros_like(grads)
    offset = 0
    for name, size in grad_sizes.items():
        chunk = torch.from_numpy(grads[:, offset:offset + size].astype(np.float32)).float()
        norm = normalizers.get(name)

        if norm is not None:
            # Reshape to [N, O, I] for weight normalization
            if isinstance(norm, AdafactorNormalizer):
                O, I = norm.row.shape[0], norm.col.shape[0]
            elif isinstance(norm, AdamNormalizer):
                O, I = norm.weight_avg_sq.shape
            else:
                O, I = None, None

            if O is not None and O * I == size:
                chunk_reshaped = chunk.reshape(-1, O, I)
                chunk_reshaped = norm.normalize_weight(chunk_reshaped)
                chunk = chunk_reshaped.reshape(-1, size)

        result[:, offset:offset + size] = chunk.numpy()
        offset += size

    return result


def cosine_sim_scores(value_grads: np.ndarray, query_grad: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between each value grad and the query."""
    # query_grad shape: [1, D] or [D]
    q = query_grad.flatten().astype(np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-10:
        return np.zeros(len(value_grads))
    q = q / q_norm

    v_norms = np.linalg.norm(value_grads, axis=1, keepdims=True).clip(min=1e-10)
    v_normed = value_grads / v_norms
    return (v_normed @ q).astype(np.float32)


def inner_product_scores(value_grads: np.ndarray, query_grad: np.ndarray) -> np.ndarray:
    """Compute raw inner product (no norm division)."""
    q = query_grad.flatten().astype(np.float32)
    return (value_grads @ q).astype(np.float32)


def evaluate_separation(scores: np.ndarray, n_forget: int = 4890, n_retain: int = 4890):
    """Compute AUROC for forget/retain separation."""
    forget = scores[:n_forget]
    retain = scores[n_forget:n_forget + n_retain]
    labels = np.concatenate([np.ones(n_forget), np.zeros(n_retain)])
    all_scores = np.concatenate([forget, retain])
    auroc = roc_auc_score(labels, all_scores)
    sep = abs(forget.mean() - retain.mean())
    return auroc, sep, forget.mean(), retain.mean()


def run_experiment(
    value_grads: np.ndarray,
    query_grad: np.ndarray,
    normalizers: dict[str, Normalizer] | None,
    grad_sizes: dict[str, int],
    label: str,
    use_inner_product: bool = False,
):
    """Run a single normalizer experiment."""
    if normalizers:
        v = apply_normalizer_to_grads(value_grads, normalizers, grad_sizes)
        q = apply_normalizer_to_grads(query_grad.reshape(1, -1), normalizers, grad_sizes)
    else:
        v = value_grads
        q = query_grad.reshape(1, -1)

    if use_inner_product:
        scores = inner_product_scores(v, q)
    else:
        scores = cosine_sim_scores(v, q)

    auroc, sep, f_mean, r_mean = evaluate_separation(scores)
    metric = "inner_prod" if use_inner_product else "cosine"
    print(f"  {label} ({metric}): AUROC={auroc:.4f}  sep={sep:.6f}  forget={f_mean:.6f}  retain={r_mean:.6f}")
    return auroc


def main():
    # Load raw value gradients (9780 examples: 4890 forget + 4890 retain)
    print("Loading gradients...")
    value_path = "runs/olmo_wmdp_raw"
    query_path = "runs/olmo_wmdp_raw_query"

    value_grads, value_info = load_grad_index(value_path)
    query_grads, query_info = load_grad_index(query_path)
    grad_sizes = value_info["grad_sizes"]

    value_grads = np.array(value_grads).astype(np.float32)
    query_grad = np.array(query_grads).astype(np.float32).flatten()

    print(f"Value: {value_grads.shape}, Query: {query_grad.shape}")
    print(f"Modules: {len(grad_sizes)}, proj_dim: {value_info.get('base_dtype')}")

    # Load normalizers
    print("\nLoading normalizers...")

    # 1. Bergson adafactor
    af_proc = GradientProcessor.load(
        "runs/olmo_wmdp/value_preconditioner", skip_preconditioners=True
    )
    # 2. 8-bit training adam
    opt_proc = GradientProcessor.load(
        "runs/optimizer_adam_normalizers", skip_preconditioners=True
    )
    # 3. SFT fp32 adam
    sft_proc = GradientProcessor.load(
        "runs/sft_adam_normalizers", skip_preconditioners=True
    )
    # 4. Bergson-fitted adam
    bergson_adam_proc = GradientProcessor.load(
        "runs/verify_query_grads/adam_normalizers", skip_preconditioners=True
    )

    print(f"  Adafactor: {len(af_proc.normalizers)} modules")
    print(f"  8-bit training Adam: {len(opt_proc.normalizers)} modules")
    print(f"  SFT fp32 Adam: {len(sft_proc.normalizers)} modules")
    print(f"  Bergson-fitted Adam: {len(bergson_adam_proc.normalizers)} modules")

    # Run experiments
    print("\n" + "=" * 80)
    print("FORGET vs RETAIN separation (cosine sim)")
    print("=" * 80)

    run_experiment(value_grads, query_grad, None, grad_sizes, "No normalizer")
    run_experiment(value_grads, query_grad, af_proc.normalizers, grad_sizes, "Adafactor (bergson)")
    run_experiment(value_grads, query_grad, opt_proc.normalizers, grad_sizes, "8-bit training Adam")
    run_experiment(value_grads, query_grad, sft_proc.normalizers, grad_sizes, "SFT fp32 Adam")
    run_experiment(value_grads, query_grad, bergson_adam_proc.normalizers, grad_sizes, "Bergson-fitted Adam")

    print("\n" + "=" * 80)
    print("FORGET vs RETAIN separation (inner product)")
    print("=" * 80)

    run_experiment(value_grads, query_grad, None, grad_sizes, "No normalizer", use_inner_product=True)
    run_experiment(value_grads, query_grad, af_proc.normalizers, grad_sizes, "Adafactor (bergson)", use_inner_product=True)
    run_experiment(value_grads, query_grad, opt_proc.normalizers, grad_sizes, "8-bit training Adam", use_inner_product=True)
    run_experiment(value_grads, query_grad, sft_proc.normalizers, grad_sizes, "SFT fp32 Adam", use_inner_product=True)
    run_experiment(value_grads, query_grad, bergson_adam_proc.normalizers, grad_sizes, "Bergson-fitted Adam", use_inner_product=True)


if __name__ == "__main__":
    main()
