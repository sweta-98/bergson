#!/bin/bash

# Set your HuggingFace token here (get it from https://huggingface.co/settings/tokens)
# export HF_TOKEN=your_token_here

export WANDB_MODE=offline
export MASTER_PORT=$((29000 + RANDOM % 1000))
export CUBLAS_WORKSPACE_CONFIG=:16:8

# Set directory for fake NVCC script to provide
# compilation information for deepspeed.
CIRCUIT_BREAKER_CUDA_HOME=/home/luciarosequirke/bergson/.fake_cuda
CIRCUIT_BREAKER_PATH=$CIRCUIT_BREAKER_CUDA_HOME/bin:$PATH

### Matches Llama-3-8B Config ###
model_name_or_path=EleutherAI/deep-ignorance-unfiltered
lorra_alpha=10 # Same as llama, mistral was 5
layers="10,20" # 2,4,
transform_layers="-1"

output_dir="./out/DeepIgnorance_CB"

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
python bergson/unlearn/circuit_breaker/lorra_deep.py \
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
    --learning_rate 1e-3 \
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
