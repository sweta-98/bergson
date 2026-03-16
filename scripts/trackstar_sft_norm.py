#!/usr/bin/env python3
"""Run trackstar using the SFT training run's full Adam second moments as normalizers.

Usage:
    python scripts/trackstar_sft_norm.py
"""

import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def save_optimizer_normalizers():
    """Load optimizer state from the latest SFT run and save as a GradientProcessor."""
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM

    from bergson.gradients import GradientProcessor
    from bergson.utils.optimizer_normalizers import load_from_optimizer

    output_path = Path("runs/sft_adam_normalizers")
    if (output_path / "processor_config.json").exists():
        print(f"SFT normalizers already saved at {output_path}, skipping.")
        return output_path

    # Use the latest SFT run's final checkpoint
    checkpoint_path = "runs/olmo_wmdp_lora/20260316_094031/checkpoint-612"
    model_path = "runs/olmo_wmdp_lora/final_adapter"

    print(f"Loading model for parameter names...")
    peft_cfg = PeftConfig.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_cfg.base_model_name_or_path,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base_model, model_path, device_map="cpu")

    print(f"Loading optimizer state from {checkpoint_path}...")
    normalizers = load_from_optimizer(model, checkpoint_path)

    proc = GradientProcessor(
        normalizers=normalizers,
        projection_dim=32,
        projection_type="rademacher",
    )
    proc.save(output_path)
    print(f"Saved SFT normalizers to {output_path}")

    del model, base_model
    return output_path


def run_trackstar(processor_path: Path):
    """Run bergson trackstar with the SFT optimizer normalizers."""
    run_path = "runs/olmo_wmdp_sft_norm_trackstar"

    cmd = [
        "bergson", "trackstar", run_path,
        "--config", "ablations/olmo_wmdp_lora.yaml",
        "--processor_path", str(processor_path),
        "--overwrite",
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

    return run_path


def main():
    processor_path = save_optimizer_normalizers()
    run_trackstar(processor_path)


if __name__ == "__main__":
    main()
