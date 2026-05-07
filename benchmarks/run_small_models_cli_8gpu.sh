#!/bin/bash

set -e

source .venv/bin/activate

TOKEN_SCALES=("10K" ) # "1B" commented out - too slow
#"100K" "1M" "10M" "100M"
MODELS=("pythia-70m" "pythia-160m" "pythia-1b") #)
DATASET="EleutherAI/SmolLM2-135M-10B"
NUM_GPUS=8

# Create runs/benchmarks directory if it doesn't exist
mkdir -p runs/benchmarks

echo "=========================================="
echo "CLI BENCHMARK FOR SMALL MODELS (8 GPUs)"
echo "=========================================="
echo "Dataset: $DATASET (streaming, unlimited tokens)"
echo "Models: ${MODELS[@]}"
echo "Token scales: ${TOKEN_SCALES[@]}"
echo "GPUs: $NUM_GPUS"
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
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running $model with $tokens tokens on $NUM_GPUS GPUs..."

        START_TIME=$(date +%s)

        python -m benchmarks.benchmark_bergson_cli \
            "$model" \
            "$tokens" \
            "runs/bergson_cli_benchmark" \
            --dataset "$DATASET" \
            --num_gpus $NUM_GPUS \
            2>&1 | tee "runs/benchmarks/small_models_8gpu_${model}_${tokens}.log"

        EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ Success: $model with $tokens tokens on $NUM_GPUS GPUs (${DURATION}s)"
        else
            echo "✗ Failed: $model with $tokens tokens on $NUM_GPUS GPUs (after ${DURATION}s)"
        fi

        echo ""

        # Update plot after each completion
        echo "Updating plot..."
       # python -m benchmarks.plot_cli_benchmark --run_root "runs/bergson_cli_benchmark" --output_path "figures"
    done

    echo ""
    echo "Completed $model"
    echo ""
done

echo "=========================================="
echo "COMPLETE!"
echo "=========================================="

#python -m benchmarks.plot_cli_benchmark --run_root "runs/bergson_cli_benchmark" --output_path "figures"
