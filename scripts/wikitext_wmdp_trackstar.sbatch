#!/bin/bash
#SBATCH --job-name=wikitext-wmdp-ts
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=/lus/lfs1aip2/projects/public/a6a/lucia/bergson3/runs/wikitext-wmdp-ts-%j.out

module load PrgEnv-cray
module load cuda/12.6

export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/home/a6a/lucia.a6a/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/home/a6a/lucia.a6a/.cache/torch_inductor

cd /lus/lfs1aip2/projects/public/a6a/lucia/bergson3
source .venv/bin/activate

if [ -z "$1" ]; then
    echo "Usage: sbatch scripts/wikitext_wmdp_trackstar.sbatch <yaml_config>"
    exit 1
fi

python -u scripts/wikitext_wmdp_trackstar.py "$1"
