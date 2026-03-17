#!/usr/bin/env python3
"""Create filtered datasets and launch training runs based on attribution scores.

For each scoring method (trackstar, raw cosine sim), selects the top 10% and
bottom 10% of examples by score, creates filtered datasets, and launches
LoRA fine-tuning runs that mimic the original training setup.

Usage::
    # Create datasets and print sbatch commands
    python scripts/filtered_finetune.py

    # Also submit the jobs
    python scripts/filtered_finetune.py --submit
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIXED_DATASET = "data/wmdp_mixed"
TOP_FRAC = 0.10
BOT_FRAC = 0.10

SCORE_CONFIGS = {
    "trackstar_nonorm": {
        "scores_path": "runs/olmo_wmdp_lora_trackstar_nonorm/scores/scores.bin",
        "description": "TrackStar (no normalizer)",
    },
    "trackstar_adafactor": {
        "scores_path": "runs/olmo_wmdp/scores/scores.bin",
        "description": "TrackStar (adafactor normalizer)",
    },
    "raw_cosine_none": {
        "type": "raw_cosine",
        "query_path": "runs/ablation_bf16_lora/none_query",
        "value_path": "runs/ablation_bf16_lora/none_mixed",
        "description": "Raw cosine sim (no normalizer)",
    },
    "trackstar_opt_fp32": {
        "scores_path": "runs/olmo_wmdp_sft_norm_trackstar/scores/scores.bin",
        "description": "TrackStar (fp32 adam optimizer buffer)",
    },
    "raw_cosine_opt_fp32": {
        "type": "raw_cosine",
        "query_path": "runs/ablation_bf16_lora/opt_fp32_adam_query",
        "value_path": "runs/ablation_bf16_lora/opt_fp32_adam_mixed",
        "description": "Raw cosine sim (fp32 adam optimizer buffer)",
    },
}


def load_trackstar_scores(scores_path):
    dtype = np.dtype({
        "names": ["score_0", "written_0"],
        "formats": ["float32", "bool"],
        "offsets": [0, 4],
        "itemsize": 8,
    })
    mmap = np.memmap(scores_path, dtype=dtype, mode="r")
    return mmap["score_0"].copy()


def compute_raw_cosine_scores(query_path, value_path):
    from bergson.data import load_gradients

    query_grads = np.array(load_gradients(query_path, structured=False)).astype(np.float32)
    value_grads = np.array(load_gradients(value_path, structured=False)).astype(np.float32)

    query_mean = query_grads.mean(axis=0)
    q_norm = np.linalg.norm(query_mean)
    if q_norm < 1e-10:
        return np.zeros(len(value_grads))
    q = query_mean / q_norm

    v_norms = np.linalg.norm(value_grads, axis=1, keepdims=True).clip(min=1e-10)
    return ((value_grads / v_norms) @ q).astype(np.float32)


def create_filtered_dataset(ds, indices, out_path):
    filtered = ds.select(indices)
    filtered.save_to_disk(out_path)
    print(f"  Saved {len(filtered)} examples to {out_path}")
    return out_path


def create_sbatch(job_name, dataset_path, output_dir):
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=runs/{job_name}-%j.out

module load PrgEnv-cray
module load cuda/12.6

export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/home/a6a/lucia.a6a/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/home/a6a/lucia.a6a/.cache/torch_inductor

cd /lus/lfs1aip2/projects/public/a6a/lucia/bergson3
source .venv/bin/activate
set -a && source .env && set +a

python -c "
import os
from datetime import datetime
import torch
import torch.distributed as dist
from datasets import Dataset, load_from_disk
from peft import LoraConfig
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

class NoShuffleSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, train_dataset):
        return SequentialSampler(train_dataset)

rank = int(os.environ.get('LOCAL_RANK', 0))
if 'LOCAL_RANK' in os.environ:
    dist.init_process_group('nccl', device_id=torch.device(f'cuda:{{rank}}'))

ds = load_from_disk('{dataset_path}')
print(f'Loaded {{len(ds)}} examples from {dataset_path}')

model = AutoModelForCausalLM.from_pretrained(
    'allenai/OLMo-2-1124-7B-Instruct', device_map={{'': f'cuda:{{rank}}'}},
)
tokenizer = AutoTokenizer.from_pretrained('allenai/OLMo-2-1124-7B-Instruct')

peft_config = LoraConfig(
    r=128, lora_alpha=256,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj'],
    lora_dropout=0.1, use_rslora=True, bias='none', task_type='CAUSAL_LM',
)

trainer = NoShuffleSFTTrainer(
    model=model, train_dataset=ds,
    args=SFTConfig(
        ddp_find_unused_parameters=False, bf16=True,
        gradient_accumulation_steps=1, learning_rate=1e-4,
        logging_steps=1, lr_scheduler_type='cosine',
        max_length=1024, max_steps=-1, num_train_epochs=4,
        optim='adamw_torch', output_dir='{output_dir}',
        per_device_train_batch_size=16, report_to='wandb',
        run_name='{job_name}', save_steps=500,
        warmup_steps=50, weight_decay=0.01,
    ),
    peft_config=peft_config,
)
trainer.train()

if rank == 0:
    adapter_path = os.path.join('{output_dir}', 'final_adapter')
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f'Adapter saved to {{adapter_path}}')

if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
"
"""
    sbatch_path = f"scripts/{job_name}.sbatch"
    with open(sbatch_path, "w") as f:
        f.write(script)
    return sbatch_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    ds = load_from_disk(MIXED_DATASET)
    n = len(ds)
    top_k = int(n * TOP_FRAC)
    bot_k = int(n * BOT_FRAC)
    sources = np.array(ds["source"])

    print(f"Dataset: {MIXED_DATASET} ({n} examples)")
    print(f"Top {TOP_FRAC*100:.0f}% = {top_k}, Bottom {BOT_FRAC*100:.0f}% = {bot_k}")

    sbatch_paths = []

    for config_name, config in SCORE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"{config['description']} ({config_name})")
        print(f"{'='*60}")

        # Load or compute scores
        if config.get("type") == "raw_cosine":
            print("  Computing raw cosine scores...")
            scores = compute_raw_cosine_scores(config["query_path"], config["value_path"])
        else:
            scores = load_trackstar_scores(config["scores_path"])

        print(f"  Scores: min={scores.min():.6f} mean={scores.mean():.6f} max={scores.max():.6f}")

        sorted_indices = np.argsort(scores)
        top_indices = sorted_indices[-top_k:][::-1]  # highest scores
        bot_indices = sorted_indices[:bot_k]  # lowest scores

        # Stats
        for label, indices in [("top", top_indices), ("bottom", bot_indices)]:
            s = scores[indices]
            forget_frac = (sources[indices] == "forget").mean()
            print(f"  {label} {len(indices)}: score [{s.min():.6f}, {s.max():.6f}], "
                  f"forget={forget_frac*100:.1f}%")

        # Create filtered datasets
        for label, indices in [("top", top_indices), ("bottom", bot_indices)]:
            ds_path = f"data/filtered_{config_name}_{label}"
            if Path(ds_path).exists():
                print(f"  {ds_path} already exists, skipping")
            else:
                create_filtered_dataset(ds, indices.tolist(), ds_path)

            output_dir = f"runs/filtered_{config_name}_{label}"
            job_name = f"filt-{config_name}-{label}"
            sbatch_path = create_sbatch(job_name, ds_path, output_dir)
            sbatch_paths.append(sbatch_path)
            print(f"  Created {sbatch_path}")

    # Submit or print commands
    print(f"\n{'='*60}")
    if args.submit:
        for path in sbatch_paths:
            result = subprocess.run(["sbatch", path], capture_output=True, text=True)
            print(f"  sbatch {path}: {result.stdout.strip()}")
    else:
        print("To submit all jobs:")
        for path in sbatch_paths:
            print(f"  sbatch {path}")
        print(f"\nOr run with --submit")


if __name__ == "__main__":
    main()
