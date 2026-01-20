#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --output=logs/eval.log
#SBATCH --error=logs/eval.err

# sbatch /home/a6a/lucia.a6a/bergson/bergson/unlearn/circuit_breaker/cas_fast.sh

accelerate launch --num_processes 4 -m bergson.unlearn.circuit_breaker.cas_faster \
    --model_name EleutherAI/deep-ignorance-unfiltered \
    --num_train_examples 256 --save_name fast

OUTPUT_DIR="./models/EleutherAI/deep-ignorance-unfiltered_fast"

echo "Running MMLU STEM evaluation..."
python scripts/eval_mmlu_stem.py --model_path $OUTPUT_DIR --batch_size 8

echo "Running WMDP Robust evaluation..."
python scripts/eval_wmdp_robust.py --model_path $OUTPUT_DIR --batch_size 8