#!/usr/bin/env python3
"""
Score training data with Trackstar using induction head queries, then retrain
on the top-scoring examples to see whether induction heads form faster.

This script:
1. Saves the synthetic induction query dataset to disk
2. Runs Trackstar scoring via CLI (streams training data, no local cache)
3. Filters the top-k training examples by induction-influence score
4. Retrains a fresh 2-layer transformer on only the filtered data
5. Trains a random-sample baseline of the same size for comparison
"""

import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from bergson.data import load_scores

# Register the custom model type with transformers
from examples.induction.attn_only_transformer import AttnOnlyForCausalLM  # noqa: F401
from examples.induction.setup_utils import (
    create_induction_ds,
    create_model,
)


def save_induction_dataset(tokenizer, seed, num_prompts, output_path):
    """Save the synthetic induction dataset to disk for bergson."""
    path = Path(output_path)
    if path.exists():
        print(f"Induction dataset already saved at {path}")
        return

    ds = create_induction_ds(tokenizer, seed=seed, num_prompts=num_prompts)

    # bergson expects a 'length' column for batch allocation
    ds = ds.map(lambda ex: {"length": len(ex["input_ids"])})
    ds.save_to_disk(str(path))
    print(f"Saved induction dataset ({len(ds)} examples) to {path}")


def run_trackstar_scoring(
    run_path: str,
    model_path: str,
    dataset_name: str,
    query_data_path: str,
):
    """Run Trackstar scoring via the bergson CLI."""
    cmd = [
        sys.executable,
        "-m",
        "bergson",
        "trackstar",
        run_path,
        "--model",
        model_path,
        "--normalizer",
        "adafactor",
        "--token_batch_size",
        "8192",
        "--data.dataset",
        dataset_name,
        "--data.truncation",
        "--query.dataset",
        query_data_path,
        "--unit_normalize",
        "--overwrite",
    ]

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return f"{run_path}/scores"


def filter_by_scores(
    scores_path: str,
    dataset_name: str,
    num_examples: int,
    output_path: str,
):
    """Select the top-k training examples by mean induction-influence score."""
    path = Path(output_path)
    if path.exists():
        print(f"Filtered dataset already saved at {path}")
        return

    scores = load_scores(Path(scores_path))
    print(f"Loaded scores: {len(scores)} items, {scores.num_scores} queries")

    # Mean score across all induction queries
    all_scores = scores[:].mean(axis=1)
    print(
        f"Score stats: mean={all_scores.mean():.4f}, "
        f"std={all_scores.std():.4f}, "
        f"min={all_scores.min():.4f}, max={all_scores.max():.4f}"
    )

    # Select top-k indices
    top_indices = np.argsort(all_scores)[-num_examples:]
    top_indices.sort()  # Keep original dataset order
    print(
        f"Selected top {num_examples} examples (score range: "
        f"{all_scores[top_indices[0]]:.4f} to {all_scores[top_indices[-1]]:.4f})"
    )

    # Load full dataset and select
    full_ds = load_dataset(dataset_name, split="train")
    filtered_ds = full_ds.select(top_indices.tolist())
    filtered_ds.save_to_disk(str(path))
    print(f"Saved filtered dataset ({len(filtered_ds)} examples) to {path}")

    return top_indices


def train_model(
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    special_pos_embed: bool = True,
    run_name: str = "induction-filter",
):
    """Train a fresh 2-layer transformer from scratch."""
    model_path = Path(output_dir) / "model.safetensors"
    if model_path.exists():
        print(f"Model already trained at {output_dir}")
        return

    print(f"Training on {len(train_dataset)} examples for 1 epoch")
    model = create_model(tokenizer, special_pos_embed=special_pos_embed)

    pad_id = -100

    def compute_metrics(eval_preds):
        preds = eval_preds.predictions
        input_ids = eval_preds.label_ids
        correct = 0
        total = 0
        for i in range(input_ids.shape[0]):
            seq = input_ids[i]
            non_pad = np.where(seq != pad_id)[0]
            if len(non_pad) == 0:
                continue
            j = non_pad[-1]
            if j == 0:
                continue
            pred_tok = preds[i, j - 1].argmax(-1)
            tgt_tok = seq[j]
            correct += int(pred_tok == tgt_tok)
            total += 1
        acc = (correct / total) if total > 0 else 0.0
        return {"accuracy": acc}

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=1,
        warmup_steps=1000,
        learning_rate=5e-4,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=10_000,
        report_to=None,
        run_name=run_name,
        seed=42,
        fp16=False,
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def tokenize_for_training(dataset, tokenizer, max_length=512):
    """Tokenize a raw text dataset for causal LM training."""

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,
        )

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )


def main():
    parser = ArgumentParser(description="Induction head data filtering experiment")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="runs/induction_filter",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="runs/two_layer_transformer",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="EleutherAI/SmolLM2-135M-10B",
    )
    parser.add_argument("--projection_dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_induction_prompts", type=int, default=100)
    parser.add_argument(
        "--filter_top_k",
        type=int,
        default=5_000_000,
        help="Number of top-scoring examples to keep for retraining",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    induction_data_path = str(base_dir / "induction_dataset")
    trackstar_path = str(base_dir / "trackstar")
    scores_path = str(base_dir / "trackstar" / "scores")
    filtered_data_path = str(base_dir / "filtered_dataset")
    filtered_model_dir = str(base_dir / "filtered_model")
    baseline_model_dir = str(base_dir / "baseline_model")

    # Tokenizer (same as original training)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 1: Save induction query dataset to disk
    print("\n=== Step 1: Preparing induction query dataset ===")
    save_induction_dataset(
        tokenizer, args.seed, args.num_induction_prompts, induction_data_path
    )

    # Step 2: Score full training data with Trackstar (streams, no local cache)
    print("\n=== Step 2: Scoring training data with Trackstar ===")
    scores_path = run_trackstar_scoring(
        run_path=trackstar_path,
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        query_data_path=induction_data_path,
    )

    # Step 3: Filter top-k from scores
    print("\n=== Step 3: Filtering training data ===")
    filter_by_scores(
        scores_path, args.dataset_name, args.filter_top_k, filtered_data_path
    )

    # Step 4: Create induction eval dataset (different seed from query)
    induction_eval = create_induction_ds(tokenizer, seed=args.seed + 1, num_prompts=100)

    # Step 5: Tokenize filtered data and train
    print("\n=== Step 4: Training on filtered data ===")
    filtered_ds = Dataset.load_from_disk(filtered_data_path)
    filtered_tok = tokenize_for_training(filtered_ds, tokenizer)
    train_model(
        tokenizer,
        filtered_tok,
        induction_eval,
        filtered_model_dir,
        run_name="induction-filtered",
    )

    # Step 6: Train baseline on same-sized random sample
    print("\n=== Step 5: Training baseline on random sample ===")
    baseline_data_path = str(base_dir / "baseline_dataset")
    if not Path(baseline_data_path).exists():
        full_ds = load_dataset(args.dataset_name, split="train")
        random_ds = full_ds.shuffle(seed=args.seed).select(range(args.filter_top_k))
        random_ds.save_to_disk(baseline_data_path)
        print(f"Saved random baseline dataset ({len(random_ds)} examples)")

    baseline_ds = Dataset.load_from_disk(baseline_data_path)
    baseline_tok = tokenize_for_training(baseline_ds, tokenizer)
    train_model(
        tokenizer,
        baseline_tok,
        induction_eval,
        baseline_model_dir,
        run_name="induction-baseline",
    )

    print("\n=== Done! ===")
    print("Compare training logs:")
    print(f"  Filtered: {filtered_model_dir}/logs/")
    print(f"  Baseline: {baseline_model_dir}/logs/")


if __name__ == "__main__":
    main()
