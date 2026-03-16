#!/usr/bin/env python3
"""Normalizer ablation: bergson build + score with different normalizers and proj dims.

Runs bergson build to collect gradients with normalizer applied PRE-projection,
then scores against the WMDP query. Tests forget/retain and retain/pile separation.

Usage:
    python scripts/normalizer_ablation.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bergson.gradients import GradientProcessor
from bergson.utils.optimizer_normalizers import load_from_optimizer


def save_normalizer(name: str, checkpoint_path: str, model_path: str) -> Path:
    """Save optimizer normalizer as a GradientProcessor."""
    output = Path(f"runs/normalizer_ablation/{name}")
    if (output / "processor_config.json").exists():
        print(f"  {name}: already saved at {output}")
        return output

    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM

    peft_cfg = PeftConfig.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_cfg.base_model_name_or_path,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base_model, model_path, device_map="cpu")
    normalizers = load_from_optimizer(model, checkpoint_path)

    # Save without projection_dim — that gets set per-experiment
    proc = GradientProcessor(
        normalizers=normalizers,
        projection_type="rademacher",
    )
    proc.save(output)
    print(f"  {name}: saved {len(normalizers)} normalizers to {output}")

    del model, base_model
    return output


def run_bergson_build(run_path: str, processor_path: str,
                      proj_dim: int,
                      data_args: list[str],
                      model: str = "runs/olmo_wmdp_lora/final_adapter",
                      ) -> bool:
    """Run bergson build with specific normalizer and projection dim."""
    cmd = [
        "bergson", "build", run_path,
        "--model", model,
        "--projection_dim", str(proj_dim),
        "--precision", "bf16",
        "--token_batch_size", "1024",
        "--skip_preconditioners",
        "--normalizer", "none",
        "--overwrite",
        "--nproc_per_node", "4",
    ]
    if processor_path:
        cmd.extend(["--processor_path", processor_path])
    cmd.extend(data_args)

    print(f"  CMD: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        print(f"  FAILED: {proc.stderr[-500:]}")
        return False
    return True


def run_bergson_score(query_path: str, value_path: str, scores_path: str,
                      data_args: list[str]) -> bool:
    """Run bergson score."""
    cmd = [
        "bergson", "score", scores_path,
        "--model", "runs/olmo_wmdp_lora/final_adapter",
        "--query_path", query_path,
        "--processor_path", value_path,
        "--skip_preconditioners",
        "--precision", "bf16",
        "--token_batch_size", "1024",
        "--overwrite",
        "--nproc_per_node", "4",
    ]
    cmd.extend(data_args)

    print(f"  CMD: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        print(f"  FAILED: {proc.stderr[-500:]}")
        return False
    return True


def load_scores(scores_path: str) -> np.ndarray | None:
    """Load scores from a bergson score run."""
    score_file = Path(scores_path) / "scores.bin"
    if not score_file.exists():
        # Try .part subdir
        score_file = Path(scores_path + ".part") / "scores.bin"
    if not score_file.exists():
        return None

    info_file = score_file.parent / "info.json"
    if info_file.exists():
        info = json.load(open(info_file))
        n = info.get("num_items", info.get("num_grads"))
    else:
        n = None

    dtype = np.dtype({
        "names": ["score_0", "written_0"],
        "formats": ["float32", "bool"],
        "offsets": [0, 4],
        "itemsize": 8,
    })
    scores = np.memmap(str(score_file), dtype=dtype, mode="r")
    return scores["score_0"]


def evaluate(scores: np.ndarray, n_forget: int = 4890, n_retain: int = 4890) -> dict:
    """Compute separation metrics."""
    from sklearn.metrics import roc_auc_score

    forget = scores[:n_forget]
    retain = scores[n_forget:n_forget + n_retain]
    labels = np.concatenate([np.ones(n_forget), np.zeros(n_retain)])
    all_s = np.concatenate([forget, retain])
    auroc = roc_auc_score(labels, all_s)

    return {
        "auroc": auroc,
        "forget_mean": float(forget.mean()),
        "retain_mean": float(retain.mean()),
        "separation": float(abs(forget.mean() - retain.mean())),
    }


def main():
    model_path = "runs/olmo_wmdp_lora/final_adapter"
    base_config = "ablations/olmo_wmdp_lora.yaml"

    # ── Step 1: Save normalizers from different training checkpoints ──────
    print("=" * 80)
    print("SAVING NORMALIZERS FROM TRAINING CHECKPOINTS")
    print("=" * 80)

    normalizer_configs = {
        "adam_8bit_step308": ("runs/olmo_wmdp_lora/checkpoint-308", model_path),
        "adam_8bit_step500": ("runs/olmo_wmdp_lora/20260316_053024/checkpoint-500", model_path),
        "adam_fp32_epoch1": ("runs/olmo_wmdp_lora_frequent/20260316_115027/checkpoint-153", model_path),
        "adam_fp32_epoch2": ("runs/olmo_wmdp_lora_frequent/20260316_115027/checkpoint-306", model_path),
        "adam_fp32_epoch3": ("runs/olmo_wmdp_lora_frequent/20260316_115027/checkpoint-459", model_path),
        "adam_fp32_epoch4": ("runs/olmo_wmdp_lora_frequent/20260316_115027/checkpoint-612", model_path),
    }

    normalizer_paths = {}
    for name, (ckpt, mpath) in normalizer_configs.items():
        normalizer_paths[name] = str(save_normalizer(name, ckpt, mpath))

    # Also include bergson-fitted normalizers
    normalizer_paths["bergson_adafactor"] = "runs/olmo_wmdp/value_preconditioner"
    normalizer_paths["none"] = ""  # empty string = no normalizer

    # ── Step 2: Run build + score ablations ──────────────────────────────
    print("\n" + "=" * 80)
    print("RUNNING ABLATIONS")
    print("=" * 80)

    proj_dims = [16, 32, 64]
    results = []

    value_data_args = [
        "--dataset", "data/wmdp_mixed", "--split", "train",
        "--truncation",
    ]
    query_data_args = [
        "--dataset", "cais/wmdp", "--split", "test",
        "--subset", "wmdp-bio",
        "--format_template", "bergson/templates/mcqa.yaml",
        "--truncation",
    ]
    score_data_args = [
        "--dataset", "data/wmdp_mixed", "--split", "train",
        "--truncation",
    ]

    for norm_name, norm_path in normalizer_paths.items():
        for proj_dim in proj_dims:
            exp_name = f"{norm_name}_proj{proj_dim}"
            run_base = f"runs/normalizer_ablation/{exp_name}"

            value_path = f"{run_base}/value"
            query_path = f"{run_base}/query"
            scores_path = f"{run_base}/scores"

            print(f"\n--- {exp_name} ---")

            # Build value gradients
            build_ok = run_bergson_build(
                value_path, norm_path, proj_dim, value_data_args)
            if not build_ok:
                print(f"  SKIPPING {exp_name} (value build failed)")
                continue

            # Build query gradients
            build_ok = run_bergson_build(
                query_path, norm_path, proj_dim, query_data_args)
            if not build_ok:
                print(f"  SKIPPING {exp_name} (query build failed)")
                continue

            # Score
            score_ok = run_bergson_score(
                query_path, value_path, scores_path, score_data_args)
            if not score_ok:
                print(f"  SKIPPING {exp_name} (score failed)")
                continue

            # Evaluate
            scores = load_scores(scores_path)
            if scores is None:
                print(f"  SKIPPING {exp_name} (no scores found)")
                continue

            metrics = evaluate(scores)
            results.append({"experiment": exp_name, "normalizer": norm_name,
                            "proj_dim": proj_dim, **metrics})
            print(f"  RESULT: AUROC={metrics['auroc']:.4f}  sep={metrics['separation']:.6f}")

    # ── Step 3: Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<40} {'AUROC':>8} {'Separation':>12}")
    print("-" * 62)
    for r in sorted(results, key=lambda x: -x["auroc"]):
        print(f"{r['experiment']:<40} {r['auroc']:>8.4f} {r['separation']:>12.6f}")

    # Save results
    out_path = Path("runs/normalizer_ablation/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
