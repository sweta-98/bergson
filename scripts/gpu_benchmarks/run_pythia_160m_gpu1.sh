#!/bin/bash

# Run pythia-160m benchmark on GPU 1
# Based on run_small_models_cli_1gpu.sh

set -e
export CUDA_VISIBLE_DEVICES=1

source .venv/bin/activate

TOKEN_SCALES=("10K" "100K" "1M" "10M" "100M")
MODEL="pythia-160m"

# Create runs/benchmarks directory if it doesn't exist
mkdir -p runs/benchmarks

echo "=========================================="
echo "CLI BENCHMARK FOR $MODEL on GPU 1"
echo "=========================================="
echo "Dataset: EleutherAI/SmolLM2-135M-10B (default)"
echo "Token scales: ${TOKEN_SCALES[@]}"
echo ""

for tokens in "${TOKEN_SCALES[@]}"; do
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running $MODEL with $tokens tokens on GPU 1..."

    START_TIME=$(date +%s)

    python -m benchmarks.benchmark_bergson_cli \
        "$MODEL" \
        "$tokens" \
        "runs/bergson_cli_benchmark" \
        2>&1 | tee "runs/benchmarks/${MODEL}_${tokens}_gpu1.log"

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Success: $MODEL with $tokens tokens on GPU 1 (${DURATION}s)"
    else
        echo "✗ Failed: $MODEL with $tokens tokens on GPU 1 (after ${DURATION}s)"
    fi

    echo ""
done

echo "Completed $MODEL on GPU 1"
