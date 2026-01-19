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

### Deep Ignorance Config
model_name_or_path=EleutherAI/deep-ignorance-unfiltered
# Weighting placed on forget loss
# Right pane 110, 
lorra_alpha=0.1 # 110  # Llama uses 10, scale by 11x for activation difference (10 * 11)
# layers="10,20"
layers="10,20"
transform_layers="-1"
learning_rate=1e-4

# output_dir="./out/DeepIgnorance_CB"
output_dir="./out/DeepIgnorance_CB_la110"


echo "model_name_or_path=$model_name_or_path"
echo "output_dir=$output_dir"

# Create output directory if it doesn't exist
mkdir -p $output_dir

# Print the command that will be executed
echo "python bergson/unlearn/circuit_breaker/lorra.py \\"

# Run with localized CUDA environment and paper hyperparameters
CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7" \
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
    --sc_train_seq_type all_text

# Run evaluations
echo "Running MMLU STEM evaluation..."
$PYTHON scripts/eval_mmlu_stem.py --model_path $output_dir --batch_size 8

echo "Running WMDP Robust evaluation..."
$PYTHON scripts/eval_wmdp_robust.py --model_path $output_dir --batch_size 8
