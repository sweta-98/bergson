#!/bin/bash

# Large models CLI benchmark script (pythia-6.9b and pythia-12b)
# These models automatically use FSDP for models >= 1B parameters
# Running one by one - war of attrition style

set -e

source .venv/bin/activate

# Token scales to test (up to 100M, 1B commented out - too slow)
TOKEN_SCALES=("10K" "100K" "1M") # capped at 1M for speed
MODELS=("pythia-6.9b" "pythia-12b")
DATASET="data/EleutherAI/SmolLM2-135M-10B-tokenized"
BATCH_SIZE=32768

echo "=========================================="
echo "Starting LARGE MODEL CLI benchmark suite"
echo "=========================================="
echo "Models: ${MODELS[@]}"
echo "Token scales: ${TOKEN_SCALES[@]}"
echo "Dataset: $DATASET"
echo "Note: FSDP will be automatically enabled"
echo ""

# Create runs/benchmarks directory if it doesn't exist
mkdir -p runs/benchmarks

# Run benchmarks for large models
for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Starting benchmarks for $model (LARGE MODEL)"
    echo "=========================================="
    echo "This will take a while..."
    echo ""

    for tokens in "${TOKEN_SCALES[@]}"; do
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running $model with $tokens tokens..."

        START_TIME=$(date +%s)

        python -m benchmarks.benchmark_bergson_cli \
            "$model" \
            "$tokens" \
            "runs/bergson_cli_benchmark" \
            --dataset "$DATASET" \
            --token_batch_size "$BATCH_SIZE" \
            2>&1 | tee -a "runs/benchmarks/large_models_cli_benchmark_${model}_${tokens}.log"

        EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ Success: $model with $tokens tokens (took ${DURATION}s)"
        else
            echo "✗ Failed: $model with $tokens tokens (after ${DURATION}s)"
            echo "Continuing to next token scale..."
        fi

        echo ""

        # Brief pause to let GPU cool down
        echo "Pausing for 5 seconds..."
        sleep 5
    done

    echo ""
    echo "=========================================="
    echo "Completed benchmarks for $model"
    echo "=========================================="
    echo ""

    # Generate intermediate plot
    echo "Generating updated plot..."
    python -m benchmarks.plot_cli_benchmark --run_root "runs/bergson_cli_benchmark" --output_path "figures"
    echo "Plot updated at figures/cli_benchmark.png"
    echo ""
done

echo "=========================================="
echo "ALL LARGE MODEL BENCHMARKS COMPLETE!"
echo "=========================================="

# Generate final plot
echo "Generating final comprehensive plot..."
python -m benchmarks.plot_cli_benchmark --run_root "runs/bergson_cli_benchmark" --output_path "figures"
