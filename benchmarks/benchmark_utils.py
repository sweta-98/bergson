import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset, load_from_disk

from bergson.config import DataConfig, IndexConfig
from bergson.utils.worker_utils import setup_data_pipeline

MAX_BENCHMARK_LENGTH = 1024


def prepare_benchmark_ds_path():
    benchmark_ds_path = Path("data/EleutherAI/SmolLM2-135M-10B-tokenized")
    if not benchmark_ds_path.exists():
        benchmark_ds_path.mkdir(parents=True, exist_ok=True)

        index_cfg = IndexConfig(
            run_path="data/EleutherAI/SmolLM2-135M-10B",
            token_batch_size=1024,
            data=DataConfig(
                dataset="EleutherAI/SmolLM2-135M-10B",
                split="train",
                truncation=True,
            ),
            autobatchsize=True,
        )
        ds = setup_data_pipeline(index_cfg)
        ds.save_to_disk(benchmark_ds_path)

        # Count number of tokens in the dataset
        total_tokens = sum(len(tokens) for tokens in ds["input_ids"])
        print(f"Total tokens: {total_tokens}")

    return benchmark_ds_path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    params: float


MODEL_SPECS: dict[str, ModelSpec] = {
    "pythia-14m": ModelSpec("pythia-14m", "EleutherAI/pythia-14m", 14_000_000),
    "pythia-70m": ModelSpec("pythia-70m", "EleutherAI/pythia-70m", 70_000_000),
    "pythia-160m": ModelSpec("pythia-160m", "EleutherAI/pythia-160m", 160_000_000),
    "pythia-1b": ModelSpec("pythia-1b", "EleutherAI/pythia-1b", 1_000_000_000),
    "pythia-6.9b": ModelSpec("pythia-6.9b", "EleutherAI/pythia-6.9b", 6_900_000_000),
    "pythia-12b": ModelSpec("pythia-12b", "EleutherAI/pythia-12b", 12_000_000_000),
}


def save_record(path: Path, record, filename: str = "benchmark.json") -> None:
    assert is_dataclass(record) and not isinstance(record, type)

    path.mkdir(parents=True, exist_ok=True)
    with open(path / filename, "w", encoding="utf-8") as fh:
        json.dump(asdict(record), fh, indent=2)


def get_run_path(
    base: Path,
    spec: ModelSpec,
    train_tokens: int,
    eval_tokens: int,
    eval_sequences: int,
    tag: str | None,
    num_gpus: int = 1,
) -> Path:
    """Create a run directory with a standardized naming convention."""
    train_label = format_tokens(train_tokens)
    eval_label = format_tokens(eval_tokens)
    run_tag = tag or timestamp()
    gpu_label = f"{num_gpus}gpu"
    path = (
        base
        / spec.key
        / f"{train_label}-{eval_label}-{eval_sequences}-{gpu_label}-{run_tag}"
    )
    return path


def timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def format_tokens(tokens: int) -> str:
    if tokens >= 1_000_000_000:
        value = tokens / 1_000_000_000
        suffix = "B"
    elif tokens >= 1_000_000:
        value = tokens / 1_000_000
        suffix = "M"
    elif tokens >= 1_000:
        value = tokens / 1_000
        suffix = "K"
    else:
        return str(tokens)
    if value.is_integer():
        return f"{int(value)}{suffix}"
    return f"{value:.2f}{suffix}"


def parse_tokens(value: str) -> int:
    text = value.strip().lower().replace(",", "")
    if text.endswith("tokens"):
        text = text[:-6]
    if not text:
        raise ValueError("empty token spec")

    suffixes = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    unit = 1
    if text[-1] in suffixes:
        unit = suffixes[text[-1]]
        text = text[:-1]
    number = float(text)
    return int(number * unit)


def load_benchmark_dataset(
    path: str | Path = prepare_benchmark_ds_path(),
    min_length: int = MAX_BENCHMARK_LENGTH,
) -> Dataset:
    """
    Load the on-disk tokenized benchmark dataset and filter to sequences >= min_length.

    This ensures all sequences are the same length (max_length) for consistent batching
    and benchmarking.

    Parameters
    ----------
    path : str | Path
        Path to the tokenized dataset on disk.
    min_length : int
        Minimum sequence length to keep. Sequences shorter than this are filtered out.

    Returns
    -------
    Dataset
        Filtered dataset with only sequences >= min_length.
    """
    path = Path(path)

    print(f"Loading tokenized dataset from {path}...")
    ds = load_from_disk(str(path))

    # Count tokens before filtering
    total_tokens_before = sum(len(tokens) for tokens in ds["input_ids"])
    num_examples_before = len(ds)

    print(
        f"Dataset loaded: {num_examples_before:,} examples, {total_tokens_before:,} "
        "tokens"
    )

    # Filter to only sequences >= min_length
    print(f"Filtering sequences to length >= {min_length}...")
    ds = ds.filter(lambda ex: len(ex["input_ids"]) >= min_length)

    # Count tokens after filtering
    total_tokens_after = sum(len(tokens) for tokens in ds["input_ids"])
    num_examples_after = len(ds)

    num_examples_removed = num_examples_before - num_examples_after
    tokens_removed = total_tokens_before - total_tokens_after

    print("\nFiltered dataset:")
    print(f"  Examples: {num_examples_after:,} (removed {num_examples_removed:,})")
    print(f"  Tokens: {total_tokens_after:,} (removed {tokens_removed:,})")
    print(
        f"  Average length: {total_tokens_after / num_examples_after:.1f}"
        " tokens/example"
    )

    return ds
