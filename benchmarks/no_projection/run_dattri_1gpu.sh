#!/bin/bash

# Dattri benchmark WITHOUT random projection (1 GPU)
# Uses full gradients - this is the default dattri behavior

set -e

source .venv/bin/activate

TOKEN_SCALES=("10K" "100K" "1M")
MODELS=("pythia-14m" "pythia-70m" "pythia-160m")
DATASET="data/EleutherAI/SmolLM2-135M-10B-tokenized"

mkdir -p runs/benchmarks

echo "=========================================="
echo "DATTRI BENCHMARK (NO PROJECTION)"
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
            --run_root "runs/dattri_noproj" \
            --dataset "$DATASET" \
            --max_length 1024 \
            2>&1 | tee "runs/benchmarks/dattri_noproj_${model}_${tokens}.log"

        echo ""
    done
done

echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
