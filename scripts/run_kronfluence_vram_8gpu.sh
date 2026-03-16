#!/bin/bash
# Run kronfluence benchmark with 8 GPUs via torchrun and
# monitor VRAM from outside via nvidia-smi polling.

set -e

cd "$(dirname "$0")/.."

RUN_ROOT="runs/kronfluence_vram_8gpu_benchmark"
CSV_OUT="runs/benchmarks/kronfluence_vram_benchmark_8gpu.csv"
NUM_GPUS=8
EXAMPLES="200"
EVAL_EXAMPLES="1"
MODELS=("pythia-14m" "pythia-70m" "pythia-160m" "pythia-1b")

mkdir -p runs/benchmarks

# --- VRAM monitoring helpers ---
VRAM_PEAK_FILE=$(mktemp)

start_vram_monitor() {
    echo 0 > "$VRAM_PEAK_FILE"
    (
        while true; do
            # Query all GPUs, take max (integer MiB)
            MAX_MB=$(nvidia-smi \
                --query-gpu=memory.used \
                --format=csv,noheader,nounits \
                | sort -rn | head -1 | tr -d ' ')
            PREV=$(cat "$VRAM_PEAK_FILE" | tr -d ' ')
            if [ "$MAX_MB" -gt "$PREV" ] 2>/dev/null; then
                echo "$MAX_MB" > "$VRAM_PEAK_FILE"
            fi
            sleep 0.25
        done
    ) &
    VRAM_PID=$!
}

stop_vram_monitor() {
    kill "$VRAM_PID" 2>/dev/null || true
    wait "$VRAM_PID" 2>/dev/null || true
    cat "$VRAM_PEAK_FILE"
}

echo "=========================================="
echo "KRONFLUENCE 8-GPU VRAM BENCHMARK"
echo "=========================================="
echo "Models:   ${MODELS[*]}"
echo "Examples: $EXAMPLES"
echo "GPUs:     $NUM_GPUS"
echo ""

declare -A RESULTS

for model in "${MODELS[@]}"; do
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $model ($EXAMPLES examples, ${NUM_GPUS} GPUs)..."

    RUN_PATH="${RUN_ROOT}/${model}/${EXAMPLES}-${EVAL_EXAMPLES}-1-${NUM_GPUS}gpu"

    CMD="torchrun --nproc_per_node=$NUM_GPUS -m benchmarks.kronfluence_benchmark run $model $EXAMPLES $EVAL_EXAMPLES $RUN_ROOT"
    echo "$CMD"

    start_vram_monitor

    START_TIME=$(date +%s)
    if $CMD; then
        END_TIME=$(date +%s)
        PEAK_VRAM=$(stop_vram_monitor)
        DURATION=$((END_TIME - START_TIME))
        echo "OK: $model (${DURATION}s, peak VRAM: ${PEAK_VRAM} MB)"
        RESULTS[$model]="$PEAK_VRAM"
    else
        END_TIME=$(date +%s)
        PEAK_VRAM=$(stop_vram_monitor)
        DURATION=$((END_TIME - START_TIME))
        echo "FAILED: $model (${DURATION}s, peak VRAM: ${PEAK_VRAM} MB)"
        RESULTS[$model]="FAILED"
    fi

    echo ""
done

rm -f "$VRAM_PEAK_FILE"

echo ""
echo "=========================================="
echo "Collecting results into CSV"
echo "=========================================="

python -c "
import csv, json, sys
from pathlib import Path

root = Path('$RUN_ROOT')
rows = []
for jf in sorted(root.rglob('benchmark.json')):
    with open(jf) as f:
        r = json.load(f)
    if r.get('status') != 'success':
        continue
    rows.append(r)

if not rows:
    print('No successful runs found.', file=sys.stderr)
    sys.exit(1)

cols = [
    'model_key', 'model_name', 'params',
    'train_examples', 'eval_examples', 'dataset',
    'strategy', 'per_device_batch_size',
    'runtime_seconds', 'peak_vram_mb',
    'run_path', 'hardware',
]
with open('$CSV_OUT', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f'Wrote {len(rows)} records to $CSV_OUT')
"

echo ""
echo "Summary of peak VRAM (nvidia-smi, max across all GPUs):"
for model in "${MODELS[@]}"; do
    echo "  $model: ${RESULTS[$model]} MB"
done

echo "Done."
