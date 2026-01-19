#!/bin/bash

# Set your HuggingFace token here (get it from https://huggingface.co/settings/tokens)
# export HF_TOKEN=your_token_here

PYTHON=/home/luciarosequirke/bergson/venv/bin/python

export WANDB_MODE=offline
export MASTER_PORT=$((29000 + RANDOM % 1000))
export CUBLAS_WORKSPACE_CONFIG=:16:8

# Set directory for fake NVCC script to provide
# compilation information for deepspeed.
CIRCUIT_BREAKER_CUDA_HOME=/home/luciarosequirke/bergson/.fake_cuda
CIRCUIT_BREAKER_PATH=$CIRCUIT_BREAKER_CUDA_HOME/bin:$PATH

### Deep Ignorance Config with Cosine Loss ###
model_name_or_path=EleutherAI/deep-ignorance-unfiltered
lorra_alpha=100  # Higher alpha for stronger intervention
layers="10,20"
transform_layers="-1"
learning_rate=3e-4
cb_loss_scale=2000  # Target the promising loss_scale=2000

output_dir="./out/DeepIgnorance_CB_cosine_alpha100_scale2000"

echo "model_name_or_path=$model_name_or_path"
echo "output_dir=$output_dir"

# Create output directory if it doesn't exist
mkdir -p $output_dir

# Print the command that will be executed
echo "Executing command:"
echo "python bergson/unlearn/circuit_breaker/lorra_deep.py \\"

# Run with localized CUDA environment and paper hyperparameters
# same lr as llama
# python bergson/unlearn/circuit_breaker/lorra.py \
# PYTORCH_ALLOC_CONF=expandable_segments:True \
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
    --output_dir  $output_dir \
    --overwrite_output_dir \
    --max_steps 150 \
    --bf16 True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --use_refusal_retain \
    --do_eval \
    --eval_steps 1000  \
    --save_total_limit 0 \
    --learning_rate $learning_rate \
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
    --sc_train_seq_type all_text \
    --cb_loss_scale $cb_loss_scale

# Run evaluations
echo "Running MMLU STEM evaluation..."
$PYTHON scripts/eval_mmlu_stem.py --model_path $output_dir --batch_size 8

echo "Running WMDP Robust evaluation..."
$PYTHON scripts/eval_wmdp_robust.py --model_path $output_dir --batch_size 8
