#!/bin/bash

# Large models CLI benchmark script (pythia-6.9b and pythia-12b)
# These models automatically use FSDP for models >= 1B parameters
# Running with 8 GPUs for maximum throughput

set -e

# Prevent CUDA memory fragmentation (use new variable name)
export PYTORCH_ALLOC_CONF=expandable_segments:True

source .venv/bin/activate

TOKEN_SCALES=("10K" "100K" "1M" "10M" "100M") #  "1B"
MODELS=("pythia-6.9b" "pythia-12b")
DATASET="data/EleutherAI/SmolLM2-135M-10B-tokenized"
BATCH_SIZE=32768
NUM_GPUS=8

echo "=========================================="
echo "Starting LARGE MODEL CLI benchmark suite (8 GPUs)"
echo "=========================================="
echo "Models: ${MODELS[@]}"
echo "Token scales: ${TOKEN_SCALES[@]}"
echo "Dataset: $DATASET"
echo "GPUs: $NUM_GPUS"
echo "Note: FSDP will be automatically enabled"
echo ""

# Create runs/benchmarks directory if it doesn't exist
mkdir -p runs/benchmarks

# Run benchmarks for large models
for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Starting benchmarks for $model (LARGE MODEL, 8 GPUs)"
    echo "=========================================="
    echo ""

    for tokens in "${TOKEN_SCALES[@]}"; do
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running $model with $tokens tokens on $NUM_GPUS GPUs..."

        START_TIME=$(date +%s)

        python -m benchmarks.benchmark_bergson_cli \
            "$model" \
            "$tokens" \
            "runs/bergson_cli_benchmark" \
            --dataset "$DATASET" \
            --token_batch_size "$BATCH_SIZE" \
            --num_gpus $NUM_GPUS \
            --fsdp \
            2>&1 | tee -a "runs/benchmarks/large_models_8gpu_${model}_${tokens}.log"

        EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ Success: $model with $tokens tokens on $NUM_GPUS GPUs (took ${DURATION}s)"

            # Generate updated plot after each successful benchmark
            echo "Generating updated plot..."
            python -m benchmarks.plot_cli_benchmark \
                --run_root "runs/bergson_cli_benchmark" \
                --output_path "figures" \
                --filter_num_gpus ${NUM_GPUS}
        else
            echo "✗ Failed: $model with $tokens tokens on $NUM_GPUS GPUs (after ${DURATION}s)"
            echo "Continuing to next token scale..."
        fi

        # Brief pause to let GPU cool down
        echo "Pausing for 5 seconds..."
        sleep 5
    done

    echo ""
    echo "=========================================="
    echo "Completed benchmarks for $model on $NUM_GPUS GPUs"
    echo "=========================================="
    echo ""
done

echo "=========================================="
echo "ALL LARGE MODEL BENCHMARKS COMPLETE (8 GPUs)!"
echo "=========================================="

# Generate final plot
echo "Generating final comprehensive plot..."
python -m benchmarks.plot_cli_benchmark \
    --run_root "runs/bergson_cli_benchmark" \
    --output_path "figures" \
    --filter_num_gpus ${NUM_GPUS}
