#!/bin/bash
# Run bergson CLI benchmark with 8 GPUs and VRAM monitoring.
# Tries each model without FSDP first; retries with FSDP on failure.

set -e

cd "$(dirname "$0")/.."

RUN_ROOT="runs/bergson_vram_8gpu_benchmark"
CSV_OUT="runs/benchmarks/bergson_vram_benchmark_8gpu.csv"
NUM_GPUS=8
# 1M tokens is enough for VRAM to stabilize
TOKEN_SCALE="1M"
MODELS=("pythia-14m" "pythia-70m" "pythia-160m" "pythia-1b" "pythia-6.9b" "pythia-12b")

mkdir -p runs/benchmarks

echo "=========================================="
echo "BERGSON 8-GPU VRAM BENCHMARK"
echo "=========================================="
echo "Models: ${MODELS[*]}"
echo "Tokens: $TOKEN_SCALE"
echo "GPUs:   $NUM_GPUS"
echo ""

for model in "${MODELS[@]}"; do
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $model ($TOKEN_SCALE tokens, ${NUM_GPUS} GPUs)..."

    CMD="python -m benchmarks.benchmark_bergson_cli $model $TOKEN_SCALE $RUN_ROOT --num_gpus $NUM_GPUS --skip_existing False"
    echo "$CMD"

    if $CMD; then
        echo "OK: $model without FSDP"
    else
        echo "FAILED without FSDP, retrying with FSDP..."
        CMD_FSDP="python -m benchmarks.benchmark_bergson_cli $model $TOKEN_SCALE $RUN_ROOT --num_gpus $NUM_GPUS --fsdp True --skip_existing False"
        echo "$CMD_FSDP"
        $CMD_FSDP
        echo "OK: $model with FSDP"
    fi

    echo ""
done

echo ""
echo "=========================================="
echo "Collecting results into CSV"
echo "=========================================="

python -c "
import csv, json, sys
from pathlib import Path

root = Path('$RUN_ROOT')
rows = []
for jf in sorted(root.rglob('benchmark_cli.json')):
    with open(jf) as f:
        r = json.load(f)
    if r.get('status') != 'success':
        continue
    rows.append(r)

if not rows:
    print('No successful runs found.', file=sys.stderr)
    sys.exit(1)

cols = [
    'model_key', 'model_name', 'params', 'train_tokens',
    'eval_tokens', 'dataset', 'batch_size',
    'build_seconds', 'score_seconds',
    'build_peak_vram_mb', 'score_peak_vram_mb',
    'num_gpus', 'token_batch_size', 'projection_dim',
    'run_path', 'hardware',
]
with open('$CSV_OUT', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f'Wrote {len(rows)} records to $CSV_OUT')
"

echo "Done."
