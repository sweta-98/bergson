#!/usr/bin/env python3
"""Evaluate a model on WMDP bio robust benchmark using lm_eval."""

import argparse

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    include_path = "/home/luciarosequirke/bergson/bergson/unlearn/lm_eval_tasks"
    tm = TaskManager(verbosity="INFO", include_path=include_path)

    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )

    print("Running evaluation on wmdp_bio_robust...")
    results = simple_evaluate(
        model=lm,
        tasks=["wmdp_bio_robust"],
        task_manager=tm,
    )

    print("\n" + "=" * 60)
    print("WMDP Bio Robust Results:")
    print("=" * 60)

    if "results" in results:
        for task_name, task_results in results["results"].items():
            print(f"\n{task_name}:")
            for metric, value in task_results.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

    return results


if __name__ == "__main__":
    main()
