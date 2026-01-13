#!/bin/bash

# Set your HuggingFace token here (get it from https://huggingface.co/settings/tokens)
# export HF_TOKEN=your_token_here

export WANDB_MODE=offline
export MASTER_PORT=$((29000 + RANDOM % 1000))
export CUBLAS_WORKSPACE_CONFIG=:16:8

# Circuit breaker specific CUDA workaround - only for this script
CIRCUIT_BREAKER_CUDA_HOME=/home/luciarosequirke/bergson/.fake_cuda
CIRCUIT_BREAKER_PATH=$CIRCUIT_BREAKER_CUDA_HOME/bin:$PATH

model_name_or_path=EleutherAI/deep-ignorance-unfiltered

### Circuit Breakers Paper Config with Deep Ignorance Unfiltered Model ###
lorra_alpha=10 # Same as llama, mistral was 5
layers="10,20"  # Standard circuit breaker target layers
transform_layers="-1"

output_dir="./out/DeepIgnorance_CB_paper"

echo "Running Circuit Breaker with Deep Ignorance Unfiltered Model (Paper Settings)"
echo "model_name_or_path=$model_name_or_path"
echo "output_dir=$output_dir"

# Create output directory if it doesn't exist
mkdir -p $output_dir

# Print the command that will be executed (following project conventions)
echo "Executing command:"
echo "python bergson/unlearn/circuit_breaker/lorra_prototype.py \\"

# Run with localized CUDA environment and paper hyperparameters
# same lr as llama
CUDA_HOME=$CIRCUIT_BREAKER_CUDA_HOME \
PATH=$CIRCUIT_BREAKER_PATH \
PYTORCH_ALLOC_CONF=expandable_segments:True \
DS_SKIP_CUDA_CHECK=1 \
DS_BUILD_OPS=0 \
DS_BUILD_FUSED_ADAM=0 \
DS_BUILD_CPU_ADAM=0 \
DS_BUILD_UTILS=0 \
python bergson/unlearn/circuit_breaker/lorra_prototype.py \
    --model_name_or_path $model_name_or_path \
    --target_layers $layers \
    --transform_layers $transform_layers \
    --lorra_alpha $lorra_alpha \
    --lora_r 8 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --output_dir  $output_dir \
    --overwrite_output_dir \
    --max_steps 150 \
    --bf16 True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --use_refusal_retain \
    --do_eval \
    --eval_steps 500 \
    --save_total_limit 0 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 2048 \
    --q_lora False \
    --gradient_checkpointing True \
    --report_to none \
    --log_every 1