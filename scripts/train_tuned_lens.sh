#!/bin/bash
# Train tuned lens for EleutherAI/deep-ignorance-unfiltered

export CUDA_VISIBLE_DEVICES=1

cd /home/luciarosequirke/lucia/bergson

python -m bergson.tuned_lens.train \
    --model_name "EleutherAI/deep-ignorance-unfiltered" \
    --bio_forget_path "/home/luciarosequirke/lucia/bergson/data/bio_forget_ds" \
    --output_dir "runs/tuned_lens" \
    --num_epochs 3 \
    --batch_size 8 \
    --lr 1e-3 \
    --wandb_project "tuned-lens" \
    --wandb_run_name "deep-ignorance-unfiltered-lens"
