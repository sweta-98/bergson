#!/usr/bin/env python3
"""Normalizer ablation: raw cosine sim with a single normalizer.

Normalizers are applied PRE-projection via processor_path or normalizer arg,
so each normalizer requires its own bergson build. Run one normalizer per
invocation; launch multiple in parallel via separate sbatch jobs.

After all normalizer builds complete, run with --score-only to compute AUROCs.

Usage:
    # Build indices for one normalizer (parallelizable)
    python scripts/normalizer_ablation.py --model lora --normalizer none
    python scripts/normalizer_ablation.py --model lora --normalizer bergson_adafactor
    python scripts/normalizer_ablation.py --model lora --normalizer opt_8bit_adam

    # Score all completed normalizers
    python scripts/normalizer_ablation.py --model lora --score-only
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bergson.build import build
from bergson.config import DataConfig, IndexConfig, PreprocessConfig
from bergson.data import load_gradients
from bergson.gradients import GradientProcessor
from bergson.utils.optimizer_normalizers import load_from_optimizer


MODEL_CONFIGS = {
    "lora": {
        "model": "runs/olmo_wmdp_lora/final_adapter",
        "type": "lora",
        "opt_ckpts": {
            "opt_8bit_adam": "runs/olmo_wmdp_lora/checkpoint-308",
            "opt_fp32_adam": "runs/olmo_wmdp_lora/20260316_092716/checkpoint-612",
        },
    },
    "sft": {
        "model": "runs/olmo_wmdp_sft/20260316_105701/final_model",
        "type": "sft",
        "opt_ckpts": {
            "opt_fp32_adam": "runs/olmo_wmdp_sft/20260316_105701/checkpoint-612",
        },
    },
    "lora_rp": {
        "model": "runs/olmo_retain_pile_lora/20260316_233724/final_adapter",
        "type": "lora",
        "opt_ckpts": {
            "opt_fp32_adam": "runs/olmo_retain_pile_lora/20260316_233724/checkpoint-932",
        },
    },
    "sft_rp": {
        "model": "runs/olmo_retain_pile_sft/20260316_233724/final_model",
        "type": "sft",
        "opt_ckpts": {
            "opt_fp32_adam": "runs/olmo_retain_pile_sft/20260316_233724/checkpoint-932",
        },
    },
}

DATASETS = {
    "mixed": DataConfig(dataset="data/wmdp_mixed", split="train", truncation=True),
    "pile": DataConfig(dataset="NeelNanda/pile-10k", split="train", truncation=True),
    "retain": DataConfig(dataset="data/wmdp_retain", split="train", truncation=True),
}

QUERY_DATA = DataConfig(
    dataset="cais/wmdp",
    split="test",
    subset="wmdp-bio",
    format_template="bergson/templates/mcqa.yaml",
    truncation=True,
)


def build_index(
    run_path: str,
    model: str,
    data: DataConfig,
    precision: str = "bf16",
    processor_path: str | None = None,
    normalizer: str = "none",
    stats_sample_size: int | None = None,
):
    """Build a gradient index with normalizer applied pre-projection."""
    if Path(run_path, "gradients.bin").exists():
        print(f"  Index exists at {run_path}, skipping.")
        return

    preprocess = PreprocessConfig(
        unit_normalize=True,
        aggregation="none",
        normalize_aggregated_grad=False,
    )

    kwargs = {}
    if processor_path:
        kwargs["processor_path"] = processor_path
    if stats_sample_size:
        kwargs["stats_sample_size"] = stats_sample_size

    cfg = IndexConfig(
        run_path=run_path,
        model=model,
        data=data,
        normalizer=normalizer,
        precision=precision,
        token_batch_size=1024,
        projection_dim=32,
        fsdp=False,
        overwrite=True,
        skip_preconditioners=True,
        **kwargs,
    )
    build(cfg, preprocess)


def load_grads(path: str) -> np.ndarray:
    return np.array(load_gradients(path, structured=False)).astype(np.float32)


def cosine_sim(value_grads: np.ndarray, query_grad: np.ndarray) -> np.ndarray:
    q = query_grad.flatten().astype(np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-10:
        return np.zeros(len(value_grads))
    q = q / q_norm
    v_norms = np.linalg.norm(value_grads, axis=1, keepdims=True).clip(min=1e-10)
    return (value_grads / v_norms) @ q


def save_optimizer_normalizers(
    name: str, model_type: str, model_path: str,
    optimizer_ckpt: str, output_dir: Path,
) -> Path:
    out = output_dir / f"{name}_normalizers"
    if (out / "processor_config.json").exists():
        print(f"  {name}: already saved at {out}")
        return out

    print(f"  Extracting {name}...")
    if model_type == "lora":
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM
        peft_cfg = PeftConfig.from_pretrained(model_path)
        model_obj = AutoModelForCausalLM.from_pretrained(
            peft_cfg.base_model_name_or_path, dtype=torch.bfloat16, device_map="cpu"
        )
        model_obj = PeftModel.from_pretrained(model_obj, model_path, device_map="cpu")
    else:
        from transformers import AutoModelForCausalLM
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cpu"
        )
    normalizer_dict = load_from_optimizer(model_obj, optimizer_ckpt)
    proc = GradientProcessor(
        normalizers=normalizer_dict, projection_dim=32, projection_type="rademacher",
    )
    proc.save(out)
    del model_obj
    return out


def get_normalizer_config(norm_name, model_key, model_path, prefix):
    """Return (normalizer_arg, processor_path, stats_sample_size) for a normalizer."""
    cfg = MODEL_CONFIGS[model_key]
    if norm_name == "none":
        return "none", None, None
    elif norm_name == "bergson_adafactor":
        return "adafactor", None, 10000
    elif norm_name == "bergson_adam":
        return "adam", None, 10000
    elif norm_name.startswith("opt_"):
        ckpt = cfg["opt_ckpts"][norm_name]
        proc_path = save_optimizer_normalizers(
            norm_name, cfg["type"], model_path, ckpt, prefix,
        )
        return "none", str(proc_path), None
    else:
        raise ValueError(f"Unknown normalizer: {norm_name}")


def build_for_normalizer(norm_name, model_key, model_path, precision, prefix):
    """Build query + all value indices for a single normalizer."""
    norm_arg, proc_path, stats_size = get_normalizer_config(
        norm_name, model_key, model_path, prefix,
    )

    # Query
    query_path = str(prefix / f"{norm_name}_query")
    print(f"\n[{norm_name}] Building query index...")
    build_index(
        query_path, model_path, QUERY_DATA, precision,
        processor_path=proc_path, normalizer=norm_arg,
        stats_sample_size=stats_size,
    )

    # Value datasets
    for ds_name, ds_cfg in DATASETS.items():
        vp = str(prefix / f"{norm_name}_{ds_name}")
        print(f"[{norm_name}] Building {ds_name} index...")
        build_index(
            vp, model_path, ds_cfg, precision,
            processor_path=proc_path, normalizer=norm_arg,
            stats_sample_size=stats_size,
        )


def score_all(model_key, prefix):
    """Score all normalizers that have completed indices. Returns results list."""
    from datasets import load_from_disk
    ds = load_from_disk("data/wmdp_mixed")
    sources = np.array(ds["source"])
    forget_labels = (sources == "forget").astype(int)

    cfg = MODEL_CONFIGS[model_key]
    norms = ["none", "bergson_adafactor", "bergson_adam"] + list(cfg["opt_ckpts"].keys())

    results = []

    print(f"\n{'='*90}")
    print(f"RAW COSINE SIM — {model_key.upper()} MODEL")
    print(f"{'='*90}")

    print("\n--- Forget vs Retain ---")
    for norm_name in norms:
        query_path = prefix / f"{norm_name}_query"
        mixed_path = prefix / f"{norm_name}_mixed"
        if not (query_path / "gradients.bin").exists() or not (mixed_path / "gradients.bin").exists():
            print(f"  {norm_name:25s}  PENDING")
            continue
        query_grads = load_grads(str(query_path))
        query_grad = query_grads.mean(axis=0)
        mixed_grads = load_grads(str(mixed_path))
        scores = cosine_sim(mixed_grads, query_grad)
        auroc = roc_auc_score(forget_labels, scores)
        f_mean = float(scores[forget_labels == 1].mean())
        r_mean = float(scores[forget_labels == 0].mean())
        print(f"  {norm_name:25s}  AUROC={auroc:.4f}  forget={f_mean:.6f}  retain={r_mean:.6f}")
        results.append({"comparison": "forget_vs_retain", "normalizer": norm_name,
                        "auroc": float(auroc), "forget_mean": f_mean, "retain_mean": r_mean})

    print("\n--- Pile vs Retain ---")
    for norm_name in norms:
        query_path = prefix / f"{norm_name}_query"
        pile_path = prefix / f"{norm_name}_pile"
        retain_path = prefix / f"{norm_name}_retain"
        if not all((p / "gradients.bin").exists() for p in [query_path, pile_path, retain_path]):
            print(f"  {norm_name:25s}  PENDING")
            continue
        query_grads = load_grads(str(query_path))
        query_grad = query_grads.mean(axis=0)
        pile_s = cosine_sim(load_grads(str(pile_path)), query_grad)
        retain_s = cosine_sim(load_grads(str(retain_path)), query_grad)
        combined = np.concatenate([pile_s, retain_s])
        lbl = np.concatenate([np.zeros(len(pile_s)), np.ones(len(retain_s))])
        auroc = roc_auc_score(lbl, combined)
        d = (pile_s.mean() - retain_s.mean()) / np.sqrt((pile_s.std()**2 + retain_s.std()**2) / 2)
        print(f"  {norm_name:25s}  AUROC={auroc:.4f}  d={d:.4f}  pile={pile_s.mean():.6f}  retain={retain_s.mean():.6f}")
        results.append({"comparison": "pile_vs_retain", "normalizer": norm_name,
                        "auroc": float(auroc), "cohen_d": float(d),
                        "pile_mean": float(pile_s.mean()), "retain_mean": float(retain_s.mean())})

    out_path = prefix / "results.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nResults saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), required=True)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--normalizer", type=str, default=None,
                        help="Build indices for this normalizer only")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip building, just score all completed indices")
    args = parser.parse_args()

    model_path = MODEL_CONFIGS[args.model]["model"]
    prefix = Path(f"runs/ablation_{args.precision}_{args.model}")
    prefix.mkdir(parents=True, exist_ok=True)

    if args.score_only:
        score_all(args.model, prefix)
        return

    if args.normalizer is None:
        parser.error("--normalizer is required when not using --score-only")

    build_for_normalizer(
        args.normalizer, args.model, model_path, args.precision, prefix,
    )

    # After building, score everything that's available
    score_all(args.model, prefix)


if __name__ == "__main__":
    main()
