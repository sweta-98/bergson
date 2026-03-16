#!/usr/bin/env python3
"""Fit Adam normalizers on WMDP query data and compare to Adafactor.

Also analyzes whether the query preconditioner's negative eigenvalues
need greater damping.

Usage:
    python scripts/fit_adam_normalizer.py
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bergson.config import DataConfig
from bergson.data import pad_and_tensor, tokenize
from bergson.format import apply_format
from bergson.gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    GradientProcessor,
    LayerAdapter,
)


def tensor_stats(t: torch.Tensor, name: str, indent: int = 4):
    prefix = " " * indent
    t_f = t.float().cpu()
    print(f"{prefix}{name}:")
    print(f"{prefix}  shape={list(t.shape)}  dtype={t.dtype}")
    print(f"{prefix}  min={t_f.min().item():.6e}  max={t_f.max().item():.6e}")
    print(f"{prefix}  mean={t_f.mean().item():.6e}  std={t_f.std().item():.6e}")
    n_zero = (t_f == 0).sum().item()
    n_neg = (t_f < 0).sum().item()
    n_nan = t_f.isnan().sum().item()
    n_inf = t_f.isinf().sum().item()
    if n_zero or n_neg or n_nan or n_inf:
        print(f"{prefix}  zeros={n_zero}  negatives={n_neg}  nans={n_nan}  infs={n_inf}")
    pcts = ""
    for p in [1, 5, 25, 50, 75, 95, 99]:
        val = torch.quantile(t_f, p / 100).item()
        pcts += f"p{p}={val:.4e} "
    print(f"{prefix}  {pcts}")


# ── Part 1: Damping analysis on existing query preconditioner ────────────────

def analyze_damping(run_path: Path):
    """Check if the current damping is sufficient for negative eigenvalues."""
    query_precond_path = run_path / "query_preconditioner"
    if not (query_precond_path / "preconditioners.pth").exists():
        print("No query preconditioner found, skipping damping analysis.")
        return

    proc = GradientProcessor.load(query_precond_path)

    print("=" * 80)
    print("DAMPING ANALYSIS — query_preconditioner")
    print("=" * 80)

    total_neg = 0
    total_eigvals = 0
    worst_neg = 0.0
    worst_module = ""

    for name, H in sorted(proc.preconditioners.items()):
        H = H.float().cpu()
        eigvals = torch.linalg.eigvalsh(H.double()).float()

        neg_eigvals = eigvals[eigvals < 0]
        total_eigvals += len(eigvals)
        total_neg += len(neg_eigvals)

        if len(neg_eigvals) > 0 and neg_eigvals.min().item() < worst_neg:
            worst_neg = neg_eigvals.min().item()
            worst_module = name

        # The actual damping applied in damped_psd_power:
        # damping_val = 0.1 * H.abs().mean()
        damping_val = 0.1 * H.abs().mean().item()

        if len(neg_eigvals) > 0:
            most_neg = neg_eigvals.min().item()
            ratio = damping_val / abs(most_neg) if most_neg != 0 else float("inf")
            # After damping: eigenvalue becomes eigval + damping_val
            # If damping_val > |most_neg|, the damped eigenvalue is positive
            is_fixed = damping_val > abs(most_neg)

            short = ".".join(name.split(".")[-3:])
            if not is_fixed:
                print(
                    f"  *** {short}: most_neg={most_neg:.4e} "
                    f"damping={damping_val:.4e} ratio={ratio:.1f} "
                    f"DAMPING INSUFFICIENT ***"
                )

    print(f"\n  Total eigenvalues: {total_eigvals}")
    print(f"  Negative eigenvalues: {total_neg} ({100*total_neg/total_eigvals:.1f}%)")
    print(f"  Most negative: {worst_neg:.6e} (in {worst_module})")

    # Check with different damping factors
    print(f"\n  Damping factor analysis:")
    for factor in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        n_still_neg = 0
        for name, H in proc.preconditioners.items():
            H = H.float().cpu()
            eigvals = torch.linalg.eigvalsh(H.double()).float()
            damping_val = factor * H.abs().mean().item()
            damped = eigvals + damping_val
            n_still_neg += (damped < 0).sum().item()
        print(f"    factor={factor:.2f}: {n_still_neg} eigenvalues still negative after damping")


# ── Part 2: Adafactor→Adam conversion check ─────────────────────────────────

def compare_adafactor_adam_conversion(run_path: Path):
    """Convert existing Adafactor normalizers to Adam and compare."""
    query_precond_path = run_path / "query_preconditioner"
    proc = GradientProcessor.load(query_precond_path, skip_preconditioners=True)

    print("\n" + "=" * 80)
    print("ADAFACTOR → ADAM CONVERSION (rank-one materialization)")
    print("=" * 80)

    module_names = sorted(proc.normalizers.keys())
    # Sample a few
    if len(module_names) > 8:
        sample = (
            module_names[:2]
            + module_names[len(module_names) // 2 - 1 : len(module_names) // 2 + 1]
            + module_names[-2:]
        )
    else:
        sample = module_names

    for name in sample:
        norm = proc.normalizers[name]
        if not isinstance(norm, AdafactorNormalizer):
            continue

        adam = norm.to_adam()
        short = ".".join(name.split(".")[-3:])
        print(f"\n  [{short}]")
        print(f"    Adafactor row: [{norm.row.shape[0]}] col: [{norm.col.shape[0]}]")

        # The materialized weight_avg_sq = outer(row, col) / row.mean()
        tensor_stats(adam.weight_avg_sq, "Adam weight_avg_sq [O, I]")

        # Check for zeros/near-zeros that would blow up normalization
        denom = adam.weight_avg_sq.sqrt().add(1e-8)
        tensor_stats(1.0 / denom, "Adam normalization factor (1/sqrt(avg_sq + eps))")


# ── Part 3: Fit fresh Adam normalizer from scratch ───────────────────────────

def fit_adam_normalizer(run_path: Path, max_examples: int = 200):
    """Fit Adam normalizers from scratch on WMDP query data."""
    model_path = "runs/olmo_wmdp_lora/final_adapter"
    subset = "wmdp-bio"
    split = "test"
    format_template = str(
        Path(__file__).resolve().parents[1] / "bergson" / "templates" / "mcqa.yaml"
    )

    # Load and format data
    print("\n" + "=" * 80)
    print(f"FITTING ADAM NORMALIZER FROM SCRATCH ({max_examples} examples)")
    print("=" * 80)

    print("Loading WMDP dataset...")
    ds = load_dataset("cais/wmdp", subset, split=split)
    ds = ds.select(range(min(max_examples, len(ds))))
    ds_formatted = apply_format(ds, format_template)

    peft_cfg = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_cfg.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    data_cfg = DataConfig(
        prompt_column="prompt",
        completion_column="completion",
        truncation=True,
    )
    ds_tok = ds_formatted.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=data_cfg, tokenizer=tokenizer, max_length=2048),
    )

    # Load model
    print("Loading model...")
    device = torch.device("cuda:0")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        base_model, model_path,
        device_map="auto",
        autocast_adapter_dtype=False,
    )
    model.requires_grad_(False)
    model.get_input_embeddings().requires_grad_(True)
    model.eval()

    # Find LoRA linear modules
    target_modules = {}
    from peft import PeftModel as PM
    from peft.utils import get_peft_model_state_dict
    peft_state = get_peft_model_state_dict(model=model)
    for peft_name in peft_state.keys():
        prefix = peft_name.removesuffix(".weight")
        processed = f"{prefix}.default".removeprefix("base_model.")
        try:
            mod = model.get_submodule(processed)
            if isinstance(mod, LayerAdapter.supported_modules):
                target_modules[processed] = mod
        except AttributeError:
            pass

    module_names = sorted(target_modules.keys())
    print(f"Tracking {len(module_names)} LoRA modules")

    # Accumulators: Adam = E[grad^2] as full [O, I] matrices
    adam_accum = {}
    adafactor_accum = {}

    hooks = []
    activation_cache = {}

    def make_fwd_hook(mod_name):
        def hook(module, input, output):
            activation_cache[mod_name] = input[0].detach()
        return hook

    def make_bwd_hook(mod_name):
        def hook(module, grad_input, grad_output):
            g = grad_output[0].detach()  # [N, S, O]
            a = activation_cache.get(mod_name)
            if a is None:
                return
            P = g.mT @ a  # [N, O, I]

            # Adam: accumulate E[P^2] elementwise
            sq = P.float().square().sum(0)  # [O, I]
            if mod_name not in adam_accum:
                adam_accum[mod_name] = torch.zeros_like(sq)
                adafactor_accum[mod_name] = {
                    "row": torch.zeros(sq.shape[0], device=sq.device),
                    "col": torch.zeros(sq.shape[1], device=sq.device),
                }
            adam_accum[mod_name].add_(sq)
            adafactor_accum[mod_name]["row"].add_(sq.mean(dim=1))
            adafactor_accum[mod_name]["col"].add_(sq.mean(dim=0))
        return hook

    for name in module_names:
        mod = target_modules[name]
        hooks.append(mod.register_forward_hook(make_fwd_hook(name)))
        hooks.append(mod.register_full_backward_hook(make_bwd_hook(name)))

    # Process
    n_processed = 0
    batch_size = 2
    for start in range(0, len(ds_tok), batch_size):
        end = min(start + batch_size, len(ds_tok))
        batch_indices = list(range(start, end))

        input_ids_list = [ds_tok[i]["input_ids"] for i in batch_indices]
        labels_list = [
            ds_tok[i].get("labels", ds_tok[i]["input_ids"]) for i in batch_indices
        ]

        x, y, _ = pad_and_tensor(input_ids_list, labels=labels_list, device=device)
        model.zero_grad()
        logits = model(x).logits[:, :-1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y[:, 1:].flatten(),
            reduction="sum",
        )
        if loss.item() > 0:
            loss.backward()
        n_processed += len(batch_indices)
        activation_cache.clear()

        if (start // batch_size) % 20 == 0:
            print(f"  {end}/{len(ds_tok)} examples")

    for h in hooks:
        h.remove()

    # Average
    for name in adam_accum:
        adam_accum[name].div_(n_processed)
        adafactor_accum[name]["row"].div_(n_processed)
        adafactor_accum[name]["col"].div_(n_processed)

    # Report Adam vs Adafactor
    print(f"\n  Processed {n_processed} examples")

    # Sample modules for reporting
    if len(module_names) > 10:
        sample = (
            module_names[:3]
            + module_names[len(module_names) // 2 - 1 : len(module_names) // 2 + 1]
            + module_names[-3:]
        )
    else:
        sample = module_names

    print("\n  --- Adam vs Adafactor comparison ---")
    for name in sample:
        if name not in adam_accum:
            continue
        short = ".".join(name.split(".")[-3:])
        adam_sq = adam_accum[name].cpu()
        af_row = adafactor_accum[name]["row"].cpu()
        af_col = adafactor_accum[name]["col"].cpu()

        # Reconstruct Adafactor's rank-1 approximation
        af_approx = torch.outer(af_row, af_col) / af_row.mean()

        # How well does Adafactor approximate Adam?
        # Relative error
        denom = adam_sq.clamp(min=1e-30)
        rel_error = ((adam_sq - af_approx).abs() / denom)

        print(f"\n  [{short}]  Adam shape={list(adam_sq.shape)}")
        tensor_stats(adam_sq, "Adam E[grad^2]")
        tensor_stats(af_approx, "Adafactor rank-1 approximation")
        tensor_stats(rel_error, "Relative error |Adam - AF| / Adam")

        # Check zeros in Adam
        n_zero = (adam_sq == 0).sum().item()
        n_near_zero = (adam_sq < 1e-20).sum().item()
        total = adam_sq.numel()
        if n_zero > 0 or n_near_zero > 0:
            print(f"    *** Adam zeros: {n_zero}/{total}  near-zero(<1e-20): {n_near_zero}/{total} ***")

        # Adam normalization factor distribution
        adam_factor = adam_sq.sqrt().add(1e-8).reciprocal()
        tensor_stats(adam_factor, "Adam norm factor 1/sqrt(E[g^2]+eps)")

        # Condition: ratio of max/min normalization factor
        cond = adam_factor.max().item() / adam_factor.min().item()
        print(f"    Adam normalization condition: {cond:.2e}")

    # Save the Adam normalizers
    output_path = Path("runs/verify_query_grads/adam_normalizers")
    output_path.mkdir(parents=True, exist_ok=True)
    adam_normalizers = {}
    for name in adam_accum:
        adam_normalizers[name] = AdamNormalizer(
            weight_avg_sq=adam_accum[name].cpu()
        )

    proc = GradientProcessor(normalizers=adam_normalizers)
    proc.save(output_path)
    print(f"\n  Adam normalizers saved to {output_path}")


def main():
    run_path = Path("runs/olmo_wmdp")

    # Part 3: Fit fresh Adam normalizer (needs GPU)
    fit_adam_normalizer(run_path, max_examples=200)


if __name__ == "__main__":
    main()
