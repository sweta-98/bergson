#!/bin/bash

# Dattri benchmark WITH random projection (1 GPU)
# Note: Dattri still computes full gradients then projects, so projection
# doesn't provide the same speedup as Bergson's hook-based projection.

set -e

source .venv/bin/activate

TOKEN_SCALES=("10K" "100K" "1M")
MODELS=("pythia-14m" "pythia-70m" "pythia-160m")
DATASET="data/EleutherAI/SmolLM2-135M-10B-tokenized"
PROJECTION_DIM=16

mkdir -p runs/benchmarks

echo "=========================================="
echo "DATTRI BENCHMARK (WITH PROJECTION)"
echo "projection_dim=${PROJECTION_DIM}"
echo "=========================================="

for model in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Model: $model"
    echo "=========================================="

    for tokens in "${TOKEN_SCALES[@]}"; do
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running $model with $tokens tokens..."

        python -m benchmarks.benchmark_dattri \
            --model "$model" \
            --train_tokens "$tokens" \
            --run_root "runs/dattri_proj" \
            --dataset "$DATASET" \
            --max_length 1024 \
            --projection_dim "$PROJECTION_DIM" \
            2>&1 | tee "runs/benchmarks/dattri_proj_${model}_${tokens}.log"

        echo ""
    done
done

echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
