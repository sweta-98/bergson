#!/usr/bin/env python3
"""Run trackstar using optimizer Adam second moments as normalizers.

Step 1: Loads the 8-bit Adam exp_avg_sq from the training checkpoint and
        saves it as a GradientProcessor.
Step 2: Runs bergson trackstar with processor_path pointing to it.

Usage:
    python scripts/trackstar_optimizer_norm.py
"""

import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def save_optimizer_normalizers():
    """Load optimizer state and save as a GradientProcessor."""
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM

    from bergson.gradients import GradientProcessor
    from bergson.utils.optimizer_normalizers import load_from_optimizer

    output_path = Path("runs/optimizer_adam_normalizers")
    if (output_path / "processor_config.json").exists():
        print(f"Optimizer normalizers already saved at {output_path}, skipping.")
        return output_path

    model_path = "runs/olmo_wmdp_lora/final_adapter"
    checkpoint_path = "runs/olmo_wmdp_lora/checkpoint-308"

    print("Loading model for parameter names...")
    peft_cfg = PeftConfig.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_cfg.base_model_name_or_path,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base_model, model_path, device_map="cpu")

    print("Loading optimizer state...")
    normalizers = load_from_optimizer(model, checkpoint_path)

    # Save as a GradientProcessor with projection settings matching the config
    proc = GradientProcessor(
        normalizers=normalizers,
        projection_dim=32,
        projection_type="rademacher",
    )
    proc.save(output_path)
    print(f"Saved optimizer normalizers to {output_path}")

    # Free memory
    del model, base_model
    return output_path


def run_trackstar(processor_path: Path):
    """Run bergson trackstar with the optimizer normalizers."""
    run_path = "runs/olmo_wmdp_optimizer_norm_trackstar"

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
