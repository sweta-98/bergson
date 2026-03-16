#!/usr/bin/env python3
"""Compare bergson-fitted adam/adafactor normalizer second moments against
the optimizer buffer from the last OLMo training run.

Usage::

    python scripts/compare_normalizers.py
"""

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from bergson.gradients import AdafactorNormalizer, AdamNormalizer, GradientProcessor
from bergson.utils.optimizer_normalizers import load_from_optimizer

MODEL_NAME = "allenai/OLMo-2-1124-7B-Instruct"
ADAPTER_PATH = "runs/olmo_wmdp_lora/final_adapter"
OPTIMIZER_PATH = "runs/olmo_wmdp_lora/checkpoint-308"

# Bergson normalizer paths from the trackstar run
ADAM_NORMALIZER_PATH = "runs/olmo_wmdp/value_preconditioner"
ADAFACTOR_NORMALIZER_PATH = "runs/olmo_wmdp/value_preconditioner"


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors (flattened)."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return (a_flat @ b_flat / (a_flat.norm() * b_flat.norm())).item()


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative Frobenius error ||a - b|| / ||b||."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return ((a_flat - b_flat).norm() / b_flat.norm()).item()


def main():
    print("Loading model + adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="cpu", torch_dtype=torch.float32
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # Load optimizer second moments as normalizers
    print("\nLoading optimizer second moments...")
    opt_normalizers = load_from_optimizer(model, OPTIMIZER_PATH)

    # Load bergson-fitted normalizers
    print("\nLoading bergson-fitted normalizers...")
    processor = GradientProcessor.load(ADAM_NORMALIZER_PATH, map_location="cpu")
    bergson_normalizers = processor.normalizers

    # Find common modules
    opt_keys = set(opt_normalizers.keys())
    bergson_keys = set(bergson_normalizers.keys())
    common = sorted(opt_keys & bergson_keys)

    print(f"\nOptimizer modules: {len(opt_keys)}")
    print(f"Bergson modules:   {len(bergson_keys)}")
    print(f"Common modules:    {len(common)}")

    if not common:
        # Try stripping prefixes for matching
        print("\nNo direct matches. Optimizer keys sample:")
        for k in sorted(opt_keys)[:5]:
            print(f"  {k}")
        print("Bergson keys sample:")
        for k in sorted(bergson_keys)[:5]:
            print(f"  {k}")
        return

    print(f"\n{'Module':<60s} {'Type':>10s} {'Cosine':>8s} {'RelErr':>8s}")
    print("-" * 90)

    all_cosines_adam = []
    all_cosines_adafactor = []

    for module in common:
        opt_norm = opt_normalizers[module]
        bergson_norm = bergson_normalizers[module]

        # Compare Adam second moments directly
        if isinstance(bergson_norm, AdamNormalizer):
            cos = cosine_sim(opt_norm.weight_avg_sq, bergson_norm.weight_avg_sq)
            rel = relative_error(opt_norm.weight_avg_sq, bergson_norm.weight_avg_sq)
            all_cosines_adam.append(cos)
            short_name = module.split(".")[-2] + "." + module.split(".")[-1]
            layer = module.split(".")[2] if "layers" in module else "?"
            label = f"L{layer}.{short_name}"
            print(f"{label:<60s} {'adam':>10s} {cos:>8.4f} {rel:>8.4f}")

        # Also compare as adafactor
        if isinstance(bergson_norm, AdafactorNormalizer):
            # Convert optimizer's adam normalizer to adafactor for comparison
            opt_as_adafactor = opt_norm.to_adafactor()
            cos_row = cosine_sim(opt_as_adafactor.row, bergson_norm.row)
            cos_col = cosine_sim(opt_as_adafactor.col, bergson_norm.col)
            all_cosines_adafactor.append((cos_row + cos_col) / 2)
            short_name = module.split(".")[-2] + "." + module.split(".")[-1]
            layer = module.split(".")[2] if "layers" in module else "?"
            label = f"L{layer}.{short_name}"
            print(f"{label:<60s} {'adafactor':>10s} "
                  f"row={cos_row:.4f} col={cos_col:.4f}")

        # If bergson is adafactor, also materialize to adam for comparison
        if isinstance(bergson_norm, AdafactorNormalizer):
            bergson_as_adam = bergson_norm.to_adam()
            cos = cosine_sim(opt_norm.weight_avg_sq, bergson_as_adam.weight_avg_sq)
            rel = relative_error(opt_norm.weight_avg_sq, bergson_as_adam.weight_avg_sq)
            all_cosines_adam.append(cos)
            print(f"{'  (materialized adafactor→adam)':<60s} {'adam*':>10s} "
                  f"{cos:>8.4f} {rel:>8.4f}")

    print(f"\n{'='*90}")
    for name, values in [("Adam", all_cosines_adam), ("Adafactor", all_cosines_adafactor)]:
        if not values:
            continue
        v = np.array(values)
        q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"\n{name} cosine similarity (n={len(v)}):")
        print(f"  min={v.min():.4f}  p5={q[0]:.4f}  p25={q[1]:.4f}  "
              f"median={q[2]:.4f}  p75={q[3]:.4f}  p95={q[4]:.4f}  "
              f"max={v.max():.4f}  mean={v.mean():.4f}")


if __name__ == "__main__":
    main()
