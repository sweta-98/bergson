"""Tokenize a larger benchmark dataset for scaling benchmarks.

Tokenizes examples from EleutherAI/SmolLM2-135M-10B and saves to disk
for use as a pre-tokenized benchmark dataset.
"""

from pathlib import Path

from datasets import load_dataset

from bergson.config import DataConfig, IndexConfig
from bergson.utils.worker_utils import setup_data_pipeline

OUTPUT_PATH = Path(
    "/projects/a6a/public/lucia/SmolLM2-135M-10B-tokenized"
)
NUM_EXAMPLES = 100_000


def main():
    if OUTPUT_PATH.exists():
        print(f"Output path already exists: {OUTPUT_PATH}")
        print("Delete it first if you want to re-tokenize.")
        return

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    print(f"Loading {NUM_EXAMPLES:,} examples from HuggingFace...")
    raw_ds = load_dataset(
        "EleutherAI/SmolLM2-135M-10B",
        split=f"train[:{NUM_EXAMPLES}]",
    )
    print(f"Loaded {len(raw_ds):,} examples")

    # Save raw subset to a temp path so setup_data_pipeline can load
    temp_path = OUTPUT_PATH.parent / "SmolLM2-135M-10B-raw-temp"
    temp_path.mkdir(parents=True, exist_ok=True)
    raw_ds.save_to_disk(str(temp_path))
    print(f"Saved raw subset to {temp_path}")

    index_cfg = IndexConfig(
        run_path=str(OUTPUT_PATH),
        token_batch_size=1024,
        data=DataConfig(
            dataset=str(temp_path),
            split="train",
            truncation=True,
        ),
        autobatchsize=True,
    )
    ds = setup_data_pipeline(index_cfg)
    ds.save_to_disk(str(OUTPUT_PATH))

    total_tokens = sum(len(tokens) for tokens in ds["input_ids"])
    num_long = sum(
        1 for length in ds["length"] if length >= 1024
    )
    long_tokens = sum(
        length for length in ds["length"] if length >= 1024
    )
    print(f"\nTokenized dataset saved to {OUTPUT_PATH}")
    print(f"  Total examples: {len(ds):,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Examples >= 1024 tokens: {num_long:,}")
    print(f"  Tokens from long examples: {long_tokens:,}")

    # Clean up temp
    import shutil

    shutil.rmtree(temp_path)
    print(f"Cleaned up temp directory: {temp_path}")


if __name__ == "__main__":
    main()
