#!/bin/bash
#SBATCH --job-name=ablation-lora
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=runs/ablation-lora-%j.out

module load PrgEnv-cray
module load cuda/12.6

export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/home/a6a/lucia.a6a/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/home/a6a/lucia.a6a/.cache/torch_inductor
export TORCHDYNAMO_DISABLE=1

cd /lus/lfs1aip2/projects/public/a6a/lucia/bergson3
source .venv/bin/activate
set -a && source .env && set +a

python scripts/normalizer_ablation.py --model lora
