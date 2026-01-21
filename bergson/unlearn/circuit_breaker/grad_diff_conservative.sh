#!/bin/bash

# Conservative Gradient Difference Unlearning
# Uses KL regularization to prevent capability collapse

export WANDB_MODE=offline
export MASTER_PORT=$((29000 + RANDOM % 1000))
export CUBLAS_WORKSPACE_CONFIG=:16:8

CIRCUIT_BREAKER_CUDA_HOME=/home/luciarosequirke/bergson/.fake_cuda
CIRCUIT_BREAKER_PATH=$CIRCUIT_BREAKER_CUDA_HOME/bin:$PATH

model_name_or_path=EleutherAI/deep-ignorance-unfiltered

alpha=0.9
gamma=1.0
num_forget=5000
num_retain=5000

output_dir="./runs/deep_ignorance_GradDiff_conservative_alpha${alpha}_gamma${gamma}"

echo "============================================================"
echo "Conservative Gradient Difference Unlearning"
echo "============================================================"
echo "model_name_or_path=$model_name_or_path"
echo "alpha=$alpha (low - gentle unlearning)"
echo "gamma=$gamma (high - stay close to original)"
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
python bergson/unlearn/circuit_breaker/grad_diff.py \
    --model_name_or_path $model_name_or_path \
    --alpha $alpha \
    --gamma $gamma \
    --num_forget_examples $num_forget \
    --num_retain_examples $num_retain \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --max_steps 200 \
    --bf16 True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --do_eval \
    --eval_steps 50 \
    --save_total_limit 0 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --report_to none

echo ""
echo "Running evaluations..."
echo ""

echo "Running MMLU STEM evaluation..."
python scripts/eval_mmlu_stem.py --model_path $output_dir --batch_size 8

echo ""
echo "Running WMDP Robust evaluation..."
python scripts/eval_wmdp_robust.py --model_path $output_dir --batch_size 8
