#!/usr/bin/env python3
"""Evaluate a model on WMDP Bio (multiple choice QA accuracy).

Usage::
    python scripts/eval_wmdp_bio.py --model allenai/OLMo-2-1124-7B-Instruct
    python scripts/eval_wmdp_bio.py --model runs/filtered_trackstar_nonorm_top/final_adapter
"""

import argparse

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_mcqa(model, tokenizer, dataset, device):
    """Evaluate multiple-choice accuracy by comparing log-probs of each choice."""
    correct = 0
    total = 0

    for example in tqdm(dataset, desc="Evaluating"):
        question = example["question"]
        choices = example["choices"]
        answer_idx = example["answer"]

        choice_logprobs = []
        for choice in choices:
            text = f"{question}\n{choice}"
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            # Log-prob of the choice tokens (everything after the question)
            q_len = len(tokenizer(question, return_tensors="pt")["input_ids"][0])
            choice_logits = logits[0, q_len - 1:-1]
            choice_ids = inputs["input_ids"][0, q_len:]
            logprobs = torch.log_softmax(choice_logits, dim=-1)
            choice_logprob = logprobs.gather(1, choice_ids.unsqueeze(1)).sum().item()
            choice_logprobs.append(choice_logprob)

        predicted = max(range(len(choice_logprobs)), key=lambda i: choice_logprobs[i])
        if predicted == answer_idx:
            correct += 1
        total += 1

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    print(f"Model: {args.model}")

    # Load model (handle PEFT adapters)
    try:
        from peft import PeftConfig, PeftModel

        peft_cfg = PeftConfig.from_pretrained(args.model)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_cfg.base_model_name_or_path,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        model = PeftModel.from_pretrained(base_model, args.model)
        tokenizer = AutoTokenizer.from_pretrained(peft_cfg.base_model_name_or_path)
        print(f"Loaded PEFT adapter from {args.model}")
    except (ValueError, OSError):
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        print(f"Loaded base model from {args.model}")

    model.eval()
    device = next(model.parameters()).device

    # Load WMDP Bio
    ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
    print(f"WMDP Bio: {len(ds)} questions")

    accuracy = evaluate_mcqa(model, tokenizer, ds, device)
    print(f"\nWMDP Bio Accuracy: {accuracy:.4f} ({int(accuracy * len(ds))}/{len(ds)})")


if __name__ == "__main__":
    main()
