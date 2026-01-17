#!/bin/bash

# Circuit Breaker (LoRRA) Hyperparameter Sweep for Deep Ignorance
# Usage: ./lorra_sweep_deep.sh <lorra_alpha>

if [ -z "$1" ]; then
    echo "Usage: $0 <lorra_alpha>"
    echo "Example: $0 10"
    exit 1
fi

lorra_alpha=$1
PYTHON=/home/luciarosequirke/bergson/venv/bin/python

export WANDB_MODE=offline
export MASTER_PORT=$((29000 + RANDOM % 1000))
export CUBLAS_WORKSPACE_CONFIG=:16:8

CIRCUIT_BREAKER_CUDA_HOME=/home/luciarosequirke/bergson/.fake_cuda
CIRCUIT_BREAKER_PATH=$CIRCUIT_BREAKER_CUDA_HOME/bin:$PATH

model_name_or_path=EleutherAI/deep-ignorance-unfiltered
# Use more layers like lorra_ckpt_deep.sh
layers="5,10,15,20"
transform_layers="-1"

output_dir="./runs/deep_ignorance_CB_alpha${lorra_alpha}"

echo "============================================================"
echo "Circuit Breaker (LoRRA) Sweep"
echo "============================================================"
echo "model_name_or_path=$model_name_or_path"
echo "lorra_alpha=$lorra_alpha"
echo "layers=$layers"
echo "output_dir=$output_dir"
echo "============================================================"

mkdir -p $output_dir

CUDA_HOME=$CIRCUIT_BREAKER_CUDA_HOME \
PATH=$CIRCUIT_BREAKER_PATH \
DS_SKIP_CUDA_CHECK=1 \
DS_BUILD_OPS=0 \
DS_BUILD_FUSED_ADAM=0 \
DS_BUILD_CPU_ADAM=0 \
DS_BUILD_UTILS=0 \
$PYTHON bergson/unlearn/circuit_breaker/lorra_deep.py \
    --model_name_or_path $model_name_or_path \
    --target_layers $layers \
    --transform_layers $transform_layers \
    --lorra_alpha $lorra_alpha \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules query_key_value dense dense_h_to_4h dense_4h_to_h \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --max_steps 150 \
    --bf16 True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --use_refusal_retain \
    --do_eval \
    --eval_steps 1000 \
    --save_total_limit 0 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 8192 \
    --q_lora False \
    --gradient_checkpointing False \
    --report_to none \
    --log_every 1 \
    --coeff_schedule linear_converge \
    --sc_loss_type orig_act_dotprod \
    --sc_train_seq_type all_text

echo ""
echo "Running evaluations..."
echo ""

echo "Running MMLU STEM evaluation..."
$PYTHON scripts/eval_mmlu_stem.py --model_path $output_dir --batch_size 8

echo ""
echo "Running WMDP Robust evaluation..."
$PYTHON scripts/eval_wmdp_robust.py --model_path $output_dir --batch_size 8
