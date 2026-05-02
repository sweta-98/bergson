#!/bin/bash
#SBATCH --job-name=less_pipeline
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=18:00:00
#SBATCH --output=logs/less_pipeline_%A_%N.out
#SBATCH --error=logs/less_pipeline_%A_%N.err

# LESS replication on a 4-node × 4-H200 slurm cluster.
# Submit from the repo root:   sbatch examples/slurm/less_pipeline.sh
#
# Phase 0 (head node): download LESS data + warmup checkpoints from HF hub.
# Phase 1 (4 nodes parallel): each node builds eval + train indices for one
#   of the 4 warmup checkpoints (106/212/318/424).
# Phase 2 (head node): scoring + filtered SFT + MMLU eval via examples.less.less,
#   which skips already-built indices.
#
# Requires: bergson installed in the active env; HF_TOKEN set or `hf auth login`
# already done; shared filesystem visible to all nodes at $SLURM_SUBMIT_DIR.

set -euo pipefail
mkdir -p logs

cd "$SLURM_SUBMIT_DIR"

# Run config — must match between phases so paths line up.
export PROJECTION_DIM=8192
export SEED=42
export LR=2e-5
export PRECISION=fp32
export PDBS=4

export RUN_NAME="Llama-2-7b-hfp${PROJECTION_DIM}_s${SEED}_lr${LR}"
export RUN_PATH="runs/less/${RUN_NAME}"
export WARMUP_PATH="${RUN_PATH}/warmup"
export EVAL_INDEX="${RUN_PATH}/eval_index"
export TRAIN_INDEX="${RUN_PATH}/train_index"
export EVAL_DATA="runs/less/extracted_data/data/eval/mmlu_processed"
export TRAIN_DATA="runs/less/extracted_data/data/train/processed"
export WARMUP_REPO="EleutherAI/less-replication-7b-warmup"
export PARALLEL_EVAL=4

# Phase 0: prep. Download LESS data + the 4 warmup checkpoints (epochs 1-4).
echo "[$(date)] Phase 0: prep"
python -m examples.less.download_less

mkdir -p "$WARMUP_PATH"
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo = os.environ["WARMUP_REPO"]
warmup_path = Path(os.environ["WARMUP_PATH"])
for epoch in range(1, 5):
    target = warmup_path / f"checkpoint-{106 * epoch}"
    if (target / "adapter_config.json").exists():
        print(f"[skip] {target} already on disk")
        continue
    print(f"[download] {repo}@epoch-{epoch} -> {target}")
    snapshot_download(repo_id=repo, revision=f"epoch-{epoch}", local_dir=str(target))
PY

# The MMLU subjects need to be processed into the eval-data layout that
# build_subset_indices expects. less.py's load_ds() does this on the fly,
# so prime it once on the head node before fanning out.
python -c "from examples.less.less import LESSConfig, load_ds; load_ds(LESSConfig())" >/dev/null

# Phase 1: build indices on 4 nodes in parallel (1 checkpoint per node).
echo "[$(date)] Phase 1: build indices on $SLURM_NNODES nodes"
srun --kill-on-bad-exit=1 \
    --output=logs/less_worker_%A_%t.out \
    --error=logs/less_worker_%A_%t.err \
    bash examples/slurm/less_index_worker.sh

# Phase 2: scoring + filtered SFT + MMLU eval on the head node.
# less.py finds existing indices and skips straight to scoring.
echo "[$(date)] Phase 2: finalize"
WANDB_MODE=disabled torchrun --nproc_per_node 4 -m examples.less.less \
    --pdbs "$PDBS" \
    --projection_dim "$PROJECTION_DIM" \
    --seed "$SEED" \
    --learning_rate "$LR" \
    --precision "$PRECISION"

echo "[$(date)] Pipeline complete"
