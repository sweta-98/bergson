#!/bin/bash
# Per-node worker for less_pipeline.sh.
# Picks one of the 4 warmup checkpoints by SLURM_NODEID, builds:
#   - eval indices for all 57 MMLU subjects, parallelizing PARALLEL_EVAL
#     subjects at a time (one GPU each)
#   - train indices for {cot, dolly, flan_v2, oasst1} sequentially, each
#     using all GPUs on the node via DDP
# Idempotent: skips any index directory that already exists.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

CHECKPOINTS=(checkpoint-106 checkpoint-212 checkpoint-318 checkpoint-424)
ckpt_name=${CHECKPOINTS[$SLURM_NODEID]}
ckpt="${WARMUP_PATH}/${ckpt_name}"

local_gpus=${SLURM_GPUS_PER_NODE:-4}
parallel_eval=${PARALLEL_EVAL:-$local_gpus}

epoch_eval_index="${EVAL_INDEX}/${ckpt_name}"
epoch_train_index="${TRAIN_INDEX}/${ckpt_name}"

echo "[$(date)] Node $SLURM_NODEID -> $ckpt_name (gpus=$local_gpus, parallel_eval=$parallel_eval)"
mkdir -p "$epoch_eval_index" "$epoch_train_index"

build_args=(
    --truncation
    --projection_dim "$PROJECTION_DIM"
    --projection_target global
    --token_batch_size "$TOKEN_BATCH_SIZE"
    --precision "$PRECISION"
    --overwrite
)

# Eval indices: small datasets (~100 examples each), one GPU per subject,
# PARALLEL_EVAL subjects in flight at once.
echo "[$(date)] Building eval indices for $ckpt_name"
in_flight=0
for subject_dir in "$EVAL_DATA"/*/; do
    subject=$(basename "$subject_dir")
    sub_index="${epoch_eval_index}/${subject}"
    if [ -d "$sub_index" ]; then
        echo "[skip] $subject already built"
        continue
    fi

    gpu=$in_flight
    cmd=(python -m bergson build "$sub_index"
        --model "$ckpt"
        --dataset "$subject_dir"
        "${build_args[@]}"
        --format_template bergson/templates/mcqa.yaml
        --nproc_per_node 1)
    echo "[build] $subject on GPU $gpu: ${cmd[*]}"
    CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" &

    in_flight=$((in_flight + 1))
    if [ "$in_flight" -ge "$parallel_eval" ]; then
        wait
        in_flight=0
    fi
done
wait

# Train indices: large datasets, run sequentially with all GPUs via DDP.
echo "[$(date)] Building train indices for $ckpt_name"
for subset in cot dolly flan_v2 oasst1; do
    sub_index="${epoch_train_index}/${subset}"
    if [ -d "$sub_index" ]; then
        echo "[skip] $subset already built"
        continue
    fi

    subset_dir="${TRAIN_DATA}/${subset}"
    if [ -f "${subset_dir}/dataset_info.json" ]; then
        dataset_arg="$subset_dir"
    else
        dataset_arg=$(ls "$subset_dir"/*.jsonl | head -1)
    fi
    if [ -z "$dataset_arg" ]; then
        echo "[error] no dataset found under $subset_dir" >&2
        exit 1
    fi

    cmd=(python -m bergson build "$sub_index"
        --model "$ckpt"
        --dataset "$dataset_arg"
        "${build_args[@]}"
        --conversation_column messages
        --optimizer_state "$ckpt"
        --nproc_per_node "$local_gpus")
    echo "[build] $subset (DDP x$local_gpus): ${cmd[*]}"
    "${cmd[@]}"
done

echo "[$(date)] Node $SLURM_NODEID done"
