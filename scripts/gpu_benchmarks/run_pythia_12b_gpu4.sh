#!/bin/bash

# Run pythia-12b benchmark on GPU 4 (Large model with FSDP)
# Based on run_large_models_cli_1gpu.sh

set -e
export CUDA_VISIBLE_DEVICES=4

source .venv/bin/activate

TOKEN_SCALES=("10K" "100K" "1M" "10M" "100M")
MODEL="pythia-12b"
BATCH_SIZE=32768

echo "=========================================="
echo "CLI BENCHMARK FOR $MODEL on GPU 4 (LARGE MODEL)"
echo "=========================================="
echo "Dataset: EleutherAI/SmolLM2-135M-10B (default)"
echo "Token scales: ${TOKEN_SCALES[@]}"
echo "Note: FSDP will be automatically enabled"
echo ""

# Create runs/benchmarks directory if it doesn't exist
mkdir -p runs/benchmarks

for tokens in "${TOKEN_SCALES[@]}"; do
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running $MODEL with $tokens tokens on GPU 4..."

    START_TIME=$(date +%s)

    python -m benchmarks.benchmark_bergson_cli \
        "$MODEL" \
        "$tokens" \
        "runs/bergson_cli_benchmark" \
        --token_batch_size "$BATCH_SIZE" \
        2>&1 | tee "runs/benchmarks/${MODEL}_${tokens}_gpu4.log"

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Success: $MODEL with $tokens tokens on GPU 4 (${DURATION}s)"
    else
        echo "✗ Failed: $MODEL with $tokens tokens on GPU 4 (after ${DURATION}s)"
    fi

    echo ""

    # Brief pause to let GPU cool down
    echo "Pausing for 5 seconds..."
    sleep 5
done

echo "Completed $MODEL on GPU 4"
