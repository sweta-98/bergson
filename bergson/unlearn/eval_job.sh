#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --output=logs/eval.log
#SBATCH --error=logs/eval.err

accelerate launch --num_processes=4 -m lm_eval \
    --model hf \
    --model_args pretrained=${CHECKPOINT_PATH},dtype=bfloat16 \
    --tasks ${TASKS} \
    --batch_size auto \
    --output_path ${RESULTS_PATH} \
    --include_path ${INCLUDE_PATH}

python /home/a5k/lucia.a5k/bergson/bergson/unlearn/log_eval_to_wandb.py \
    --results ${RESULTS_PATH} \
    --step ${STEP} \
    --wandb_run_id ${WANDB_RUN_ID} \
    --wandb_project ${WANDB_PROJECT}