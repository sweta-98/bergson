#!/usr/bin/env python3
"""Run trackstar using the SFT training run's full Adam second moments as normalizers.

Usage:
    python scripts/trackstar_sft_optimizer_norm.py
"""

import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SFT_MODEL_PATH = "runs/olmo_wmdp_sft/20260316_105701/final_model"
SFT_CHECKPOINT_PATH = "runs/olmo_wmdp_sft/20260316_105701/checkpoint-612"
NORMALIZER_OUTPUT_PATH = Path("runs/sft_full_adam_normalizers")
TRACKSTAR_RUN_PATH = "runs/olmo_sft_optimizer_norm_trackstar"


def save_optimizer_normalizers():
    """Load optimizer state from the SFT run and save as a GradientProcessor."""
    from transformers import AutoModelForCausalLM

    from bergson.gradients import GradientProcessor
    from bergson.utils.optimizer_normalizers import load_from_optimizer

    if (NORMALIZER_OUTPUT_PATH / "processor_config.json").exists():
        print(f"SFT normalizers already saved at {NORMALIZER_OUTPUT_PATH}, skipping.")
        return NORMALIZER_OUTPUT_PATH

    print("Loading model for parameter names...")
    model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"Loading optimizer state from {SFT_CHECKPOINT_PATH}...")
    normalizers = load_from_optimizer(model, SFT_CHECKPOINT_PATH)

    proc = GradientProcessor(
        normalizers=normalizers,
        projection_dim=32,
        projection_type="rademacher",
    )
    proc.save(NORMALIZER_OUTPUT_PATH)
    print(f"Saved SFT normalizers to {NORMALIZER_OUTPUT_PATH}")

    del model
    return NORMALIZER_OUTPUT_PATH


def run_trackstar(processor_path: Path):
    """Run bergson trackstar with the SFT optimizer normalizers."""
    cmd = [
        "python", "-m", "bergson", "trackstar",
        "--config", "ablations/olmo_sft_optimizer_norm.yaml",
        "--processor_path", str(processor_path),
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    print()

    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)

    proc.wait()
    elapsed = time.monotonic() - t0
    mins, secs = divmod(elapsed, 60)
    print(f"\nTotal time: {int(mins)}m {secs:.1f}s")

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    return TRACKSTAR_RUN_PATH


def main():
    processor_path = save_optimizer_normalizers()
    run_trackstar(processor_path)


if __name__ == "__main__":
    main()
