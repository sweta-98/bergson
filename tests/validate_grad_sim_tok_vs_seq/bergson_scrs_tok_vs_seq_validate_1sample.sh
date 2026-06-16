#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES="0"

# QUERY STEP (animal-query)
bergson build "./teacher_number_scorings/build_op.part" \
    --model unsloth/Llama-3.2-1B-Instruct \
    --dataset "./data/elephant_query_1sample.jsonl" \
    --prompt_column "prompt" \
    --completion_column "completion" \
    --aggregation mean \
    --projection_dim 16 \
    --token_batch_size 2048 \
    --overwrite \
    --truncation \
    --filter_modules "*vision*"


# DATASET STEP (teacher data)
bergson score "./teacher_number_scorings_tok/score" \
    --model unsloth/Llama-3.2-1B-Instruct \
    --dataset "./data/elephant_teacher_numbers_1sample.jsonl" \
    --prompt_column "prompt" \
    --completion_column "completion" \
    --query_path "./teacher_number_scorings/build_op.part" \
    --projection_dim 16 \
    --token_batch_size 2048 \
    --overwrite \
    --truncation \
    --filter_modules "*vision*"\
    --attribute_tokens

bergson score "./teacher_number_scorings_seq/score" \
    --model unsloth/Llama-3.2-1B-Instruct \
    --dataset "./data/elephant_teacher_numbers_1sample.jsonl" \
    --prompt_column "prompt" \
    --completion_column "completion" \
    --query_path "./teacher_number_scorings/build_op.part" \
    --projection_dim 16 \
    --token_batch_size 2048 \
    --overwrite \
    --truncation \
    --filter_modules "*vision*"

python check_tok_sum_vs_seq.py
