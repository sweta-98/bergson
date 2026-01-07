import os
import fnmatch
import json

import pandas as pd

from datasets import load_dataset, load_from_disk, Dataset
from pathlib import Path
from transformers import AutoTokenizer
from huggingface_hub import list_repo_files, hf_hub_download

from bergson.utils.utils import assert_type


location = "mnt"
# location = "mnt"
if location == "mnt":
    # BIO_FORGET_PATH = "/mnt/ssd-1/lucia/bergson/rmu/bio-forget"
    # WMDP_REWRITTEN_PATH = "/mnt/ssd-1/lucia/bergson/rmu/wmdp-lie-o-rewritten"
    # BIO_RETAIN_PATH = "/mnt/ssd-1/lucia/bergson/rmu/bio-retain"
    BIO_RETAIN_PATH = "/home/lucia/bio_retain"
    WMDP_REWRITTEN_PATH = "/home/lucia/wmdp-lie-o-rewritten"
    BIO_FORGET_PATH = "/home/lucia/bio-forget"

    # OUTPUT_DIR = "/mnt/ssd-1/lucia/bergson/runs/bio_transfer"
    OUTPUT_DIR = "/home/lucia/bio_tmp"
    EVAL_INCLUDE_PATH = "/home/lucia/bergson/bergson/unlearn/lm_eval_tasks"
    # EVAL_INCLUDE_PATH = "/mnt/ssd-1/lucia/bergson/lm-eval-tasks"
else:
    BIO_FORGET_PATH = "/projects/a5k/public/lucia/rmu/bio-forget"
    WMDP_REWRITTEN_PATH = "/projects/a5k/public/lucia/rmu/wmdp-lie-o-rewritten"
    BIO_RETAIN_PATH = "/projects/a5k/public/lucia/rmu/bio-retain"

    # DO NOT CHANGE EVER
    OUTPUT_DIR = "/projects/a5k/public/lucia/runs/bio_transfer_test"
    EVAL_INCLUDE_PATH = "/home/a5k/lucia.a5k/bergson/bergson/unlearn/lm_eval_tasks"


def is_debug():
    return True

def find_and_load_rmu_file(dataset_name, target_pattern):
    """Find and load a specific file from the rmu-training-data dataset.
    
    Args:
        dataset_name: Name of the HuggingFace dataset (e.g., "Unlearning/rmu-training-data")
        target_pattern: Pattern to match the file name (e.g., "bio-forget-corpus")
    
    Returns:
        Dataset loaded from the matching file
    """
    # List all files in the Hugging Face repository
    all_files = list_repo_files(repo_id=dataset_name, repo_type="dataset")
    data_files_list = [f for f in all_files if f.endswith('.json') or f.endswith('.jsonl') or f.endswith('.parquet')]
    
    if is_debug():
        print(f"Found data files in {dataset_name}:")
        for f in data_files_list:
            print(f"  - {f}")
    
    # Try to find exact match first
    target_file = None
    for f in data_files_list:
        if target_pattern in f:
            target_file = f
            break
    
    if target_file is None:
        # Fallback: try to find a file that matches the pattern
        matches = fnmatch.filter(data_files_list, f"*{target_pattern}*")
        if matches:
            target_file = matches[0]
        else:
            raise ValueError(f"Could not find a file matching '{target_pattern}' in {dataset_name}")
    
    if is_debug():
        print(f"Loading specific data file: {target_file}")
    
    # Download the file and read line by line to handle schema inconsistencies
    # Download the file to a temporary location
    local_file = hf_hub_download(
        repo_id=dataset_name,
        filename=target_file,
        repo_type="dataset"
    )
        
    if is_debug():
        print(f"Downloaded file to: {local_file}")
    
    # Read JSONL file line by line to handle schema inconsistencies
    dataset_list = []
    with open(local_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip():  # Skip empty lines
                try:
                    example = json.loads(line)
                    
                    # FIX: Handle cases where the JSON object is a simple string
                    if isinstance(example, str):
                        example = {"text": example}
                        
                    dataset_list.append(example)
                    if is_debug() and (i + 1) % 1000 == 0:
                        print(f"Loaded {i + 1} examples...", flush=True)
                except json.JSONDecodeError as e:
                    if is_debug():
                        print(f"Warning: Skipping invalid JSON on line {i + 1}: {e}")
                    continue
    
    if is_debug():
        print(f"Successfully loaded {len(dataset_list)} examples")
    
    # Create dataset from list
    dataset = Dataset.from_list(dataset_list)
    return dataset


def tokenize_ds(datasets, tokenizer, max_length):
    """Tokenize datasets if not already tokenized."""

    def is_tokenized(example):
        return "input_ids" in example

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
        )

    for key in datasets:
        sample = datasets[key][0]
        if not is_tokenized(sample):
            datasets[key] = datasets[key].map(
                tokenize_function,
                batched=False,
                remove_columns=["text"],
            )

    return datasets


def load_datasets():
    """Load all required datasets."""
    # Try to resolve paths - check if relative or absolute
    project_root = Path(__file__).parent.parent.parent

    def resolve_path(path):
        if os.path.isabs(path):
            return path
        full_path = project_root / path
        if full_path.exists():
            return str(full_path)
        return path

    bio_forget_path = resolve_path(BIO_FORGET_PATH)
    rewritten_path = resolve_path(WMDP_REWRITTEN_PATH)
    retain_path = resolve_path(BIO_RETAIN_PATH)

    print(f"Loading bio-forget from: {bio_forget_path}")
    print(f"Loading wmdp-lie-o-rewritten from: {rewritten_path}")
    print(f"Loading bio-retain from: {retain_path}")

    try:
        bio_forget = Dataset.load_from_disk(bio_forget_path)
        rewritten = Dataset.load_from_disk(rewritten_path)
        retain = Dataset.load_from_disk(retain_path)
    except Exception as e:
        print(f"Error loading from disk: {e}")
        print("Trying to load from HuggingFace Hub...")
        # Use the helper function to find and load the correct files
        bio_forget = find_and_load_rmu_file("Unlearning/rmu-training-data", "bio-forget-corpus")
        
        # FIX: Added split="train" to get a Dataset instead of DatasetDict
        rewritten = load_dataset("Unlearning/wmdp-lie-o-rewritten", split="train")
        
        retain = find_and_load_rmu_file("Unlearning/rmu-training-data", "bio-retain-corpus")

    return {
        "bio_forget": assert_type(Dataset, bio_forget),
        "rewritten": assert_type(Dataset, rewritten),
        "retain": assert_type(Dataset, retain),
    }


def main():
    SEQ_LEN = 1024
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    transfer_ds_path = Path(OUTPUT_DIR + "/transfer_ds")
    retain_ds_path = Path(OUTPUT_DIR + "/mixed_retain_ds")

    # Load and tokenize datasets
    ds = load_datasets()
    if is_debug():
        print("Tokenize", flush=True)

    ds = tokenize_ds(ds, tokenizer, SEQ_LEN)
    for ds_name, dataset in ds.items():
        if is_debug():
            print(
                f"{ds_name} dataset length: {len(dataset)}, columns:"
                f"{dataset.column_names}"
            )

    retain_ds = ds["retain"]

    # Mix in ultrachat

    # Load ultrachat and prepare it for mixing
    ultrachat = load_dataset("stingning/ultrachat", split="train")
    ultrachat = assert_type(Dataset, ultrachat)
    # Mix in at 1:2 ratio
    ultrachat = ultrachat.select(range(min(len(ultrachat), len(retain_ds) // 2)))

    # Flatten ultrachat conversations to text
    def flatten_ultrachat(example):
        return {"text": "\n".join(example["data"])}

    ultrachat = ultrachat.map(flatten_ultrachat, remove_columns=ultrachat.column_names)
    ultrachat = ultrachat.map(lambda ex: tokenizer(ex["text"], truncation=True, max_length=SEQ_LEN), remove_columns=["text"])

    # Mix with retain set
    retain_ds = concatenate_datasets([retain_ds, ultrachat]).shuffle(seed=42)
    retain_ds.save_to_disk(str(retain_ds_path))

    if is_debug():
        print("Retain set len", len(retain_ds), flush=True)

    # Create dataset dict
    dataset_dict = DatasetDict({
        "bio_forget": ds["bio_forget"],
        "rewritten": ds["rewritten"],
        "retain": ds["retain"],
    })
    dataset_dict.save_to_disk(OUTPUT_DIR + "/aligned_tokenized_datasets")

    # Ensure bio and rewritten have the same length for alignment logic.
    assert len(ds["bio_forget"]) == len(ds["rewritten"]), (
        f"Bio-forget and rewritten datasets must have the same length for alignment. "
        f"Got {len(ds['bio_forget'])} and {len(ds['rewritten'])}."
    )

    transfer_ds = ds["bio_forget"].rename_column("input_ids", "source_input_ids")
    transfer_ds = transfer_ds.add_column(
        "target_input_ids", ds['rewritten']["input_ids"],
        new_fingerprint="transfer"
    )

    def filter(item):
        if len(item["source_input_ids"]) < SEQ_LEN:
            return False
        if len(item["target_input_ids"]) < SEQ_LEN:
            return False
        return True

    transfer_ds = transfer_ds.filter(filter)
    if is_debug():
        print(f"Filtered transfer dataset length: {len(transfer_ds)}")

    print("transfer ds length", len(transfer_ds), "saving to disk", flush=True)

    transfer_ds.save_to_disk(str(transfer_ds_path))
    print("done", flush=True)


if __name__ == "__main__":
    main()