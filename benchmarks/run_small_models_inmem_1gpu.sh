#!/bin/bash

# In-memory Bergson benchmark for small models (1 GPU)
# Tests in-memory reduce + score pipeline

set -e

source .venv/bin/activate

TOKEN_SCALES=("10K" "100K" "1M") # capped at 1M for speed
MODELS=("pythia-70m" "pythia-160m" "pythia-1b")
DATASET="data/EleutherAI/SmolLM2-135M-10B-tokenized"

# Create runs/benchmarks directory if it doesn't exist
mkdir -p runs/benchmarks

echo "=========================================="
echo "IN-MEMORY BENCHMARK FOR SMALL MODELS"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Models: ${MODELS[@]}"
echo "Token scales: ${TOKEN_SCALES[@]}"
echo ""
echo "=========================================="
echo ""

for model in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Model: $model"
    echo "=========================================="

    for tokens in "${TOKEN_SCALES[@]}"; do
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running $model with $tokens tokens..."

        START_TIME=$(date +%s)

        python -m benchmarks.benchmark_bergson \
            "$model" \
            "$tokens" \
            "runs/bergson_inmem_benchmark" \
            --dataset "$DATASET" \
            2>&1 | tee "runs/benchmarks/inmem_${model}_${tokens}.log"

        EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        if [ $EXIT_CODE -eq 0 ]; then
            echo "Success: $model with $tokens tokens (${DURATION}s)"
        else
            echo "Failed: $model with $tokens tokens (after ${DURATION}s)"
        fi

        echo ""

        # Update plot after each completion
        echo "Updating plot..."
        python -m benchmarks.plot_inmem_benchmark \
            --run_root "runs/bergson_inmem_benchmark" \
            --output_csv "runs/benchmarks/inmem_benchmark_1gpu.csv" \
            --output_plot "figures/inmem_benchmark_1gpu.png"
    done

    echo ""
    echo "Completed $model"
    echo ""
done

echo "=========================================="
echo "COMPLETE!"
echo "=========================================="

python -m benchmarks.plot_inmem_benchmark \
    --run_root "runs/bergson_inmem_benchmark" \
    --output_csv "runs/benchmarks/inmem_benchmark_1gpu.csv" \
    --output_plot "figures/inmem_benchmark_1gpu.png"
echo "Final plot saved to figures/inmem_benchmark_1gpu.png"
