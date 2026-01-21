#!/bin/bash

# Hyperparameter tuning for remove_coef in ckpt_deep.py
# Usage: ./scripts/tune_remove_coef.sh <remove_coef> [retain_coef] [num_train_examples]

set -e

REMOVE_COEF=${1:-50}
RETAIN_COEF=${2:-2}
NUM_TRAIN_EXAMPLES=${3:-2048}

export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export WANDB_MODE=offline
export MASTER_PORT=$((29000 + RANDOM % 1000))
export CUBLAS_WORKSPACE_CONFIG=:16:8

if [ "$RETAIN_COEF" == "2" ]; then
    SAVE_NAME="remove_coef_${REMOVE_COEF}"
else
    SAVE_NAME="remove_coef_${REMOVE_COEF}_retain_${RETAIN_COEF}"
fi
MODEL_PATH="./models/EleutherAI/deep-ignorance-unfiltered_${SAVE_NAME}"

echo "============================================="
echo "Running with remove_coef=${REMOVE_COEF}, retain_coef=${RETAIN_COEF}"
echo "num_train_examples=${NUM_TRAIN_EXAMPLES}"
echo "Model will be saved to: ${MODEL_PATH}"
echo "============================================="

CMD="python /home/luciarosequirke/lucia/bergson/bergson/unlearn/ckpt_deep.py \
    --num_train_examples ${NUM_TRAIN_EXAMPLES} \
    --remove_coef ${REMOVE_COEF} \
    --retain_coef ${RETAIN_COEF} \
    --lr 1e-3 \
    --pdbs 4 \
    --layers 3 6 9 12 15 18 21 24 27 30 \
    --save_name ${SAVE_NAME}"

echo "Executing: $CMD"
echo ""

$CMD

echo ""
echo "============================================="
echo "Training completed. Running evaluations..."
echo "============================================="

echo ""
echo "Running MMLU STEM evaluation..."
python scripts/eval_mmlu_stem.py --model_path $MODEL_PATH --batch_size 8

echo ""
echo "Running WMDP Robust evaluation..."
python scripts/eval_wmdp_robust.py --model_path $MODEL_PATH --batch_size 8

echo ""
echo "============================================="
echo "Completed run with remove_coef=${REMOVE_COEF}"
echo "============================================="
