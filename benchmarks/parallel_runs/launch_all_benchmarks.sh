#!/bin/bash

# Master launcher for parallel benchmark execution across all 8 GPUs
# Runs 5 different models on separate GPUs to avoid contention

set -e

echo "=========================================="
echo "LAUNCHING PARALLEL BENCHMARK SUITE"
echo "=========================================="
echo "Starting time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "GPU Allocation:"
echo "  GPU 0: pythia-70m (small model)"
echo "  GPU 1: pythia-160m (small model)"
echo "  GPU 2: pythia-1b (small model)"
echo "  GPU 3: pythia-6.9b (large model)"
echo "  GPU 4: pythia-12b (large model)"
echo "  GPU 5-7: Available for future use"
echo ""
echo "Each model will run through 5 token scales: 10K, 100K, 1M, 10M, 100M"
echo ""

# Create master log directory
mkdir -p runs/parallel_benchmarks/logs

# Check that we have the required dataset
DATASET="data/EleutherAI/SmolLM2-135M-10B-tokenized"
if [ ! -d "$DATASET" ]; then
    echo "ERROR: Dataset not found at $DATASET"
    echo "Please ensure the tokenized dataset is available before running benchmarks."
    exit 1
fi

# Verify NVIDIA GPUs are available
echo "Checking GPU availability..."
nvidia-smi --list-gpus | head -8
echo ""

# Launch all benchmarks in parallel
echo "=========================================="
echo "LAUNCHING BENCHMARK PROCESSES..."
echo "=========================================="

# Array to store process PIDs
declare -a PIDS

# Launch pythia-70m on GPU 0
echo "Launching pythia-70m on GPU 0..."
nohup ./benchmarks/parallel_runs/run_pythia_70m_gpu0.sh > runs/parallel_benchmarks/logs/gpu0_pythia-70m.out 2>&1 &
PIDS[0]=$!
echo "  Started with PID: ${PIDS[0]}"

# Launch pythia-160m on GPU 1
echo "Launching pythia-160m on GPU 1..."
nohup ./benchmarks/parallel_runs/run_pythia_160m_gpu1.sh > runs/parallel_benchmarks/logs/gpu1_pythia-160m.out 2>&1 &
PIDS[1]=$!
echo "  Started with PID: ${PIDS[1]}"

# Launch pythia-1b on GPU 2
echo "Launching pythia-1b on GPU 2..."
nohup ./benchmarks/parallel_runs/run_pythia_1b_gpu2.sh > runs/parallel_benchmarks/logs/gpu2_pythia-1b.out 2>&1 &
PIDS[2]=$!
echo "  Started with PID: ${PIDS[2]}"

# Launch pythia-6.9b on GPU 3
echo "Launching pythia-6.9b on GPU 3..."
nohup ./benchmarks/parallel_runs/run_pythia_6.9b_gpu3.sh > runs/parallel_benchmarks/logs/gpu3_pythia-6.9b.out 2>&1 &
PIDS[3]=$!
echo "  Started with PID: ${PIDS[3]}"

# Launch pythia-12b on GPU 4
echo "Launching pythia-12b on GPU 4..."
nohup ./benchmarks/parallel_runs/run_pythia_12b_gpu4.sh > runs/parallel_benchmarks/logs/gpu4_pythia-12b.out 2>&1 &
PIDS[4]=$!
echo "  Started with PID: ${PIDS[4]}"

echo ""
echo "All benchmark processes launched!"
echo ""

# Save PID information
echo "Process PIDs:" > runs/parallel_benchmarks/pids.txt
echo "GPU 0 (pythia-70m): ${PIDS[0]}" >> runs/parallel_benchmarks/pids.txt
echo "GPU 1 (pythia-160m): ${PIDS[1]}" >> runs/parallel_benchmarks/pids.txt
echo "GPU 2 (pythia-1b): ${PIDS[2]}" >> runs/parallel_benchmarks/pids.txt
echo "GPU 3 (pythia-6.9b): ${PIDS[3]}" >> runs/parallel_benchmarks/pids.txt
echo "GPU 4 (pythia-12b): ${PIDS[4]}" >> runs/parallel_benchmarks/pids.txt

echo "=========================================="
echo "MONITORING PROGRESS"
echo "=========================================="
echo ""
echo "Monitor individual GPU logs:"
echo "  tail -f runs/parallel_benchmarks/logs/gpu0_pythia-70m.out"
echo "  tail -f runs/parallel_benchmarks/logs/gpu1_pythia-160m.out"
echo "  tail -f runs/parallel_benchmarks/logs/gpu2_pythia-1b.out"
echo "  tail -f runs/parallel_benchmarks/logs/gpu3_pythia-6.9b.out"
echo "  tail -f runs/parallel_benchmarks/logs/gpu4_pythia-12b.out"
echo ""
echo "Monitor GPU usage:"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "Check process status:"
echo "  ps aux | grep 'run_pythia'"
echo ""

# Function to check process status
check_process_status() {
    local pid=$1
    local name=$2
    if kill -0 "$pid" 2>/dev/null; then
        echo "  ✓ $name (PID: $pid) - RUNNING"
        return 0
    else
        echo "  ✗ $name (PID: $pid) - FINISHED/FAILED"
        return 1
    fi
}

# Monitor progress periodically
echo "Starting periodic progress checks (every 5 minutes)..."
echo "Press Ctrl+C to stop monitoring (processes will continue running)"
echo ""

MONITOR_START=$(date +%s)

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Process Status Check:"

    ACTIVE_COUNT=0
    check_process_status "${PIDS[0]}" "GPU 0 (pythia-70m)" && ((ACTIVE_COUNT++))
    check_process_status "${PIDS[1]}" "GPU 1 (pythia-160m)" && ((ACTIVE_COUNT++))
    check_process_status "${PIDS[2]}" "GPU 2 (pythia-1b)" && ((ACTIVE_COUNT++))
    check_process_status "${PIDS[3]}" "GPU 3 (pythia-6.9b)" && ((ACTIVE_COUNT++))
    check_process_status "${PIDS[4]}" "GPU 4 (pythia-12b)" && ((ACTIVE_COUNT++))

    echo "  Active processes: $ACTIVE_COUNT/5"

    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - MONITOR_START))
    echo "  Elapsed time: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"

    if [ $ACTIVE_COUNT -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "ALL BENCHMARKS COMPLETED!"
        echo "Total runtime: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
        echo "=========================================="
        break
    fi

    echo ""
    sleep 300  # Check every 5 minutes
done

echo ""
echo "Results available in:"
echo "  runs/parallel_benchmarks/gpu0_pythia-70m/"
echo "  runs/parallel_benchmarks/gpu1_pythia-160m/"
echo "  runs/parallel_benchmarks/gpu2_pythia-1b/"
echo "  runs/parallel_benchmarks/gpu3_pythia-6.9b/"
echo "  runs/parallel_benchmarks/gpu4_pythia-12b/"
echo ""

# Final status report
echo "Creating final status report..."
echo "Benchmark Suite Completion Report" > runs/parallel_benchmarks/final_report.txt
echo "Generated: $(date)" >> runs/parallel_benchmarks/final_report.txt
echo "" >> runs/parallel_benchmarks/final_report.txt

for i in {0..4}; do
    case $i in
        0) MODEL="pythia-70m" ;;
        1) MODEL="pythia-160m" ;;
        2) MODEL="pythia-1b" ;;
        3) MODEL="pythia-6.9b" ;;
        4) MODEL="pythia-12b" ;;
    esac

    LOG_FILE="runs/parallel_benchmarks/gpu${i}_${MODEL}/benchmark.log"
    if [ -f "$LOG_FILE" ]; then
        echo "GPU $i ($MODEL):" >> runs/parallel_benchmarks/final_report.txt
        tail -5 "$LOG_FILE" >> runs/parallel_benchmarks/final_report.txt
        echo "" >> runs/parallel_benchmarks/final_report.txt
    fi
done

echo "Final report saved to: runs/parallel_benchmarks/final_report.txt"
