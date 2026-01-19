#!/bin/bash

# Bergson in-memory benchmark WITH random projection (1 GPU)

set -e

source .venv/bin/activate

TOKEN_SCALES=("10K" "100K" "1M")
MODELS=("pythia-14m" "pythia-70m" "pythia-160m" "pythia-1b")
DATASET="data/EleutherAI/SmolLM2-135M-10B-tokenized"
PROJECTION_DIM=16

mkdir -p runs/benchmarks

echo "=========================================="
echo "BERGSON IN-MEMORY BENCHMARK (WITH PROJECTION)"
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

        python -m benchmarks.benchmark_bergson \
            "$model" \
            "$tokens" \
            "runs/bergson_inmem_proj" \
            --dataset "$DATASET" \
            --projection_dim "$PROJECTION_DIM" \
            2>&1 | tee "runs/benchmarks/inmem_proj_${model}_${tokens}.log"

        echo ""
    done
done

echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
