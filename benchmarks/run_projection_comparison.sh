#!/bin/bash

# Run all benchmarks for projection comparison:
# - Bergson in-memory with projection (proj_dim=16)
# - Bergson in-memory without projection (proj_dim=0)
# - Dattri with projection (proj_dim=16)
# - Dattri without projection (default)

set -e

TOKEN_SCALES=("10K" "100K" "1M" "10M" "100M")
MODELS=("pythia-14m" "pythia-70m" "pythia-160m" "pythia-410m")
DATASET="data/EleutherAI/SmolLM2-135M-10B-tokenized"
PROJECTION_DIM=16

mkdir -p runs/benchmarks

echo "=========================================="
echo "PROJECTION COMPARISON BENCHMARK"
echo "=========================================="
echo "Models: ${MODELS[@]}"
echo "Token scales: ${TOKEN_SCALES[@]}"
echo "Projection dim: ${PROJECTION_DIM}"
echo ""

# 1. Bergson in-memory WITH projection
echo "=========================================="
echo "1/4: BERGSON IN-MEMORY WITH PROJECTION"
echo "=========================================="
for model in "${MODELS[@]}"; do
    for tokens in "${TOKEN_SCALES[@]}"; do
        echo "[$(date '+%H:%M:%S')] bergson-proj: $model @ $tokens"
        python -m benchmarks.benchmark_bergson \
            "$model" "$tokens" "runs/proj_comparison/bergson_proj" \
            --dataset "$DATASET" --projection_dim "$PROJECTION_DIM" \
            2>&1 | grep -E "(runtime|seconds|success)" || true
    done
done

# 2. Bergson in-memory WITHOUT projection
echo ""
echo "=========================================="
echo "2/4: BERGSON IN-MEMORY WITHOUT PROJECTION"
echo "=========================================="
for model in "${MODELS[@]}"; do
    for tokens in "${TOKEN_SCALES[@]}"; do
        echo "[$(date '+%H:%M:%S')] bergson-noproj: $model @ $tokens"
        python -m benchmarks.benchmark_bergson \
            "$model" "$tokens" "runs/proj_comparison/bergson_noproj" \
            --dataset "$DATASET" --projection_dim 0 \
            2>&1 | grep -E "(runtime|seconds|success)" || true
    done
done

# 3. Dattri WITH projection
echo ""
echo "=========================================="
echo "3/4: DATTRI WITH PROJECTION"
echo "=========================================="
for model in "${MODELS[@]}"; do
    for tokens in "${TOKEN_SCALES[@]}"; do
        echo "[$(date '+%H:%M:%S')] dattri-proj: $model @ $tokens"
        python -m benchmarks.benchmark_dattri \
            --model "$model" --train_tokens "$tokens" \
            --run_root "runs/proj_comparison/dattri_proj" \
            --dataset "$DATASET" --max_length 1024 \
            --projection_dim "$PROJECTION_DIM" \
            2>&1 | grep -E "(runtime|seconds|success)" || true
    done
done

# 4. Dattri WITHOUT projection
echo ""
echo "=========================================="
echo "4/4: DATTRI WITHOUT PROJECTION"
echo "=========================================="
for model in "${MODELS[@]}"; do
    for tokens in "${TOKEN_SCALES[@]}"; do
        echo "[$(date '+%H:%M:%S')] dattri-noproj: $model @ $tokens"
        python -m benchmarks.benchmark_dattri \
            --model "$model" --train_tokens "$tokens" \
            --run_root "runs/proj_comparison/dattri_noproj" \
            --dataset "$DATASET" --max_length 1024 \
            2>&1 | grep -E "(runtime|seconds|success)" || true
    done
done

echo ""
echo "=========================================="
echo "BENCHMARKS COMPLETE - GENERATING PLOTS"
echo "=========================================="

python -m benchmarks.plot_projection_comparison \
    --output_csv "runs/benchmarks/projection_comparison.csv" \
    --output_plot "figures/projection_comparison.png"

echo ""
echo "Results saved to:"
echo "  - runs/benchmarks/projection_comparison.csv"
echo "  - figures/projection_comparison.png"
