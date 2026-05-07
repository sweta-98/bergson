#!/bin/bash

set -e

TOKEN_SCALES=("100M")
#"pythia-14m" "pythia-70m" "pythia-1b""pythia-160m"

MODELS=( "pythia-160m" )
DATASET="EleutherAI/SmolLM2-135M-10B"

# Create runs/benchmarks directory if it doesn't exist
mkdir -p runs/benchmarks

echo "=========================================="
echo "CLI BENCHMARK FOR SMALL MODELS"
echo "=========================================="
echo "Dataset: $DATASET (streaming, unlimited tokens)"
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

        CUDA_VISIBLE_DEVICES=3 python -m benchmarks.benchmark_magic \
            "$model" \
            "$tokens" \
            "runs/bergson_magic" \
            --dataset "$DATASET" \
            2>&1 | tee "runs/benchmarks/magic_1gpu_${model}_${tokens}.log"

        EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ Success: $model with $tokens tokens (${DURATION}s)"
        else
            echo "✗ Failed: $model with $tokens tokens (after ${DURATION}s)"
        fi

        echo ""

        # Update plot after each completion
        echo "Updating plot..."
        #python -m benchmarks.plot_cli_benchmark --run_root "runs/bergson_cli_benchmark" --output_path "figures"
    done

    echo ""
    echo "Completed $model"
    echo ""
done

echo "=========================================="
echo "COMPLETE!"
echo "=========================================="

#cd /home/luciarosequirke/bergson
#python -m benchmarks.plot_cli_benchmark --run_root "runs/bergson_cli_benchmark" --output_path "figures"
