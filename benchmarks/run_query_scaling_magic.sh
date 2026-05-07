#!/bin/bash

set -e

# Fixed index: one model and one token scale
MODEL="pythia-160m"
TRAIN_TOKENS="100K"
DATASET="EleutherAI/SmolLM2-135M-10B"

# Vary the number of query examples
QUERY_COUNTS=(1 10 100 1000)

mkdir -p runs/benchmarks

echo "=========================================="
echo "QUERY SCALING BENCHMARK (MAGIC)"
echo "=========================================="
echo "Model: $MODEL"
echo "Train tokens: $TRAIN_TOKENS (fixed)"
echo "Dataset: $DATASET"
echo "Query counts: ${QUERY_COUNTS[@]}"
echo "=========================================="
echo ""

for num_queries in "${QUERY_COUNTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "num_queries: $num_queries"
    echo "=========================================="

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running $MODEL with $TRAIN_TOKENS tokens and $num_queries queries..."

    START_TIME=$(date +%s)

    python -m benchmarks.benchmark_magic \
        "$MODEL" \
        "$TRAIN_TOKENS" \
        "runs/bergson_magic_query_scaling" \
        --dataset "$DATASET" \
        --num_queries "$num_queries" \
        2>&1 | tee "runs/benchmarks/magic_query_scaling_${MODEL}_${TRAIN_TOKENS}_q${num_queries}.log"

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Success: $num_queries queries (${DURATION}s)"
    else
        echo "✗ Failed: $num_queries queries (after ${DURATION}s)"
    fi

    echo ""
done

echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
