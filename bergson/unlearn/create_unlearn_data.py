import os

from datasets import load_dataset, load_from_disk, Dataset
from pathlib import Path
from transformers import AutoTokenizer

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
        bio_forget = load_dataset(
            "Unlearning/rmu-training-data", data_files="bio-forget-corpus.jsonl"
        )
        rewritten = load_dataset("Unlearning/wmdp-lie-o-rewritten")
        retain = load_dataset(
            "Unlearning/rmu-training-data", data_files="bio-retain-corpus.jsonl"
        )

    return {
        "bio_forget": assert_type(Dataset, bio_forget),
        "rewritten": assert_type(Dataset, rewritten),
        "retain": assert_type(Dataset, retain),
    }


def main():
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

    ds = tokenize_ds(ds, tokenizer)
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