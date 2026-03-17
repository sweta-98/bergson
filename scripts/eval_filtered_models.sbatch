#!/bin/bash
#SBATCH --job-name=eval-filtered
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=runs/eval-filtered-%j.out

module load PrgEnv-cray
module load cuda/12.6

export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/home/a6a/lucia.a6a/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/home/a6a/lucia.a6a/.cache/torch_inductor

cd /lus/lfs1aip2/projects/public/a6a/lucia/bergson3
source .venv/bin/activate
set -a && source .env && set +a

echo "=== Baseline (no adapter) ==="
python scripts/eval_wmdp_bio.py --model allenai/OLMo-2-1124-7B-Instruct

echo "=== Baseline (original adapter, full mixed) ==="
python scripts/eval_wmdp_bio.py --model runs/olmo_wmdp_lora/final_adapter

for scoring in trackstar_nonorm trackstar_adafactor raw_cosine_none trackstar_opt_fp32 raw_cosine_opt_fp32; do
    for subset in top bottom; do
        run_dir="runs/filtered_${scoring}_${subset}"
        adapter="${run_dir}/final_adapter"
        if [ -d "$adapter" ]; then
            echo "=== ${scoring}_${subset} ==="
            python scripts/eval_wmdp_bio.py --model "$adapter"
        else
            echo "=== ${scoring}_${subset}: NOT READY ==="
        fi
    done
done
