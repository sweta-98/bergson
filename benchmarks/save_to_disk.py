from pathlib import Path

from bergson.config import DataConfig, IndexConfig
from bergson.utils.worker_utils import setup_data_pipeline

index_cfg = IndexConfig(
    run_path="data/EleutherAI/SmolLM2-135M-10B",
    token_batch_size=1024,
    data=DataConfig(
        dataset="EleutherAI/SmolLM2-135M-10B",
        split="train",
        truncation=True,
    ),
)

ds = setup_data_pipeline(index_cfg)

save_path = Path("data/EleutherAI/SmolLM2-135M-10B-tokenized")
save_path.mkdir(parents=True, exist_ok=True)
ds.save_to_disk(save_path)


# Count number of tokens in the dataset
total_tokens = sum(len(tokens) for tokens in ds["input_ids"])
print(f"Total tokens: {total_tokens}")
