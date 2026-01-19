#!/bin/bash

# Bergson in-memory benchmark WITHOUT random projection (1 GPU)
# Uses full gradients - will be slower and use more memory

set -e

source .venv/bin/activate

TOKEN_SCALES=("10K" "100K")  # Smaller scales due to memory
MODELS=("pythia-14m" "pythia-70m")  # Smaller models due to memory
DATASET="data/EleutherAI/SmolLM2-135M-10B-tokenized"

mkdir -p runs/benchmarks

echo "=========================================="
echo "BERGSON IN-MEMORY BENCHMARK (NO PROJECTION)"
echo "WARNING: Uses full gradients - high memory usage"
echo "=========================================="

for model in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Model: $model"
    echo "=========================================="

    for tokens in "${TOKEN_SCALES[@]}"; do
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running $model with $tokens tokens..."

        python -m benchmarks.benchmark_bergson \
            "$model" \
            "$tokens" \
            "runs/bergson_inmem_noproj" \
            --dataset "$DATASET" \
            --projection_dim 0 \
            2>&1 | tee "runs/benchmarks/inmem_noproj_${model}_${tokens}.log"

        echo ""
    done
done

echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
