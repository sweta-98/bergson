#!/bin/bash

# Benchmark pythia-6.9b on GPU 3 (LARGE MODEL - FSDP enabled)
# Part of parallel benchmark execution across 8 GPUs

set -e

export CUDA_VISIBLE_DEVICES=3

source .venv/bin/activate

MODEL="pythia-6.9b"
GPU_ID=3
TOKEN_SCALES=("10K" "100K" "1M" "10M" "100M")
DATASET="data/EleutherAI/SmolLM2-135M-10B-tokenized"
BATCH_SIZE=32768
OUTPUT_DIR="runs/parallel_benchmarks/gpu${GPU_ID}_${MODEL}"

# Create output directory
mkdir -p runs/parallel_benchmarks
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "CLI BENCHMARK FOR $MODEL on GPU $GPU_ID (LARGE MODEL)"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Token scales: ${TOKEN_SCALES[@]}"
echo "Batch size: $BATCH_SIZE"
echo "Output: $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Note: FSDP will be automatically enabled"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Log start
echo "Starting benchmark for $MODEL on GPU $GPU_ID at $(date)" > "$OUTPUT_DIR/benchmark.log"

for tokens in "${TOKEN_SCALES[@]}"; do
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running $MODEL with $tokens tokens on GPU $GPU_ID..."

    START_TIME=$(date +%s)

    # Run benchmark with unique output directory and batch size
    python -m benchmarks.benchmark_bergson_cli \
        "$MODEL" \
        "$tokens" \
        "$OUTPUT_DIR" \
        --dataset "$DATASET" \
        --token_batch_size "$BATCH_SIZE" \
        2>&1 | tee -a "$OUTPUT_DIR/benchmark_${tokens}.log"

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Success: $MODEL with $tokens tokens on GPU $GPU_ID (${DURATION}s)"
        echo "SUCCESS: $MODEL $tokens tokens completed in ${DURATION}s at $(date)" >> "$OUTPUT_DIR/benchmark.log"
    else
        echo "✗ Failed: $MODEL with $tokens tokens on GPU $GPU_ID (after ${DURATION}s)"
        echo "FAILED: $MODEL $tokens tokens failed after ${DURATION}s at $(date)" >> "$OUTPUT_DIR/benchmark.log"
    fi

    echo ""

    # Brief pause for GPU cooling
    echo "Pausing for 5 seconds..."
    sleep 5
done

echo ""
echo "=========================================="
echo "COMPLETED $MODEL on GPU $GPU_ID"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Log completion
echo "Completed benchmark for $MODEL on GPU $GPU_ID at $(date)" >> "$OUTPUT_DIR/benchmark.log"

# Generate individual plot
python -m benchmarks.plot_cli_benchmark \
    --run_root "$OUTPUT_DIR" \
    --output_csv "$OUTPUT_DIR/results.csv" \
    --output_plot "$OUTPUT_DIR/results.png"

echo "Results saved to: $OUTPUT_DIR"
echo "Plot saved to: $OUTPUT_DIR/results.png"
