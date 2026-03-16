#!/usr/bin/env python3
"""Compute raw cosine similarities (no preconditioners) between WMDP-bio
query gradients and multiple value datasets: pile-10k and wmdp retain.

Usage::

    python scripts/raw_cosine_sim.py
"""

from pathlib import Path

import numpy as np
from bergson.build import build
from bergson.config import DataConfig, IndexConfig, PreprocessConfig, ScoreConfig
from bergson.score.score import score_dataset
from datasets import load_from_disk

RAW_QUERY = "runs/olmo_wmdp_raw_query"
MODEL = "runs/olmo_wmdp_lora/final_adapter"


def make_index_cfg(run_path: str, data: DataConfig, overwrite: bool = True):
    return IndexConfig(
        run_path=run_path,
        model=MODEL,
        data=data,
        normalizer="none",
        precision="bf16",
        token_batch_size=1024,
        projection_dim=32,
        fsdp=False,
        overwrite=overwrite,
        skip_preconditioners=True,
    )


def prepare_retain_dataset():
    """Extract retain split from wmdp_mixed and save as standalone dataset."""
    out_path = "data/wmdp_retain"
    if Path(out_path).exists():
        print("Retain dataset already exists.")
        return out_path
    ds = load_from_disk("data/wmdp_mixed")
    retain = ds.filter(lambda x: x["source"] == "retain")
    retain.save_to_disk(out_path)
    print(f"Saved {len(retain)} retain examples to {out_path}")
    return out_path


def main():
    preprocess_cfg = PreprocessConfig(
        unit_normalize=True,
        aggregation="mean",
        normalize_aggregated_grad=True,
    )

    # Prepare retain dataset
    retain_path = prepare_retain_dataset()

    datasets = {
        "pile-10k": DataConfig(
            dataset="NeelNanda/pile-10k",
            split="train",
            truncation=True,
        ),
        "wmdp-retain": DataConfig(
            dataset=retain_path,
            split="train",
            truncation=True,
        ),
    }

    # Build query index (reuse if exists)
    query_cfg = make_index_cfg(
        RAW_QUERY,
        DataConfig(
            dataset="cais/wmdp",
            split="test",
            subset="wmdp-bio",
            format_template="bergson/templates/mcqa.yaml",
            truncation=True,
        ),
        overwrite=False,
    )
    if not Path(RAW_QUERY, "gradients.bin").exists():
        print("Building query index...")
        build(query_cfg, preprocess_cfg)
    else:
        print("Query index already exists, skipping.")

    score_cfg = ScoreConfig(
        query_path=RAW_QUERY,
        score="individual",
    )

    results = {}
    for name, data_cfg in datasets.items():
        run_path = f"runs/olmo_raw_{name}"
        value_cfg = make_index_cfg(run_path, data_cfg)

        print(f"\n{'='*60}")
        print(f"Dataset: {name} ({data_cfg.dataset})")
        print(f"{'='*60}")

        print("Building value index...")
        build(value_cfg, preprocess_cfg)

        print("Scoring...")
        score_dataset(value_cfg, score_cfg, preprocess_cfg)

        scores_path = find_scores(run_path)
        if scores_path:
            results[name] = analyze_scores(name, scores_path)
        else:
            print(f"WARNING: No scores.bin found under {run_path}")

    # Compare distributions
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        for name, s in results.items():
            print(f"{name:20s}: mean={s.mean():.6f}, std={s.std():.6f}, "
                  f"median={np.median(s):.6f}")


def find_scores(run_path: str) -> str | None:
    for p in Path(run_path).rglob("scores.bin"):
        return str(p)
    return None


def analyze_scores(name: str, scores_path: str) -> np.ndarray:
    dtype = np.dtype({
        "names": ["score_0", "written_0"],
        "formats": ["float32", "bool"],
        "offsets": [0, 4],
        "itemsize": 8,
    })
    scores = np.memmap(scores_path, dtype=dtype, mode="r")
    s = scores["score_0"].copy()

    print(f"\n--- {name} ---")
    print(f"N={len(s)}, range=[{s.min():.6f}, {s.max():.6f}], "
          f"mean={s.mean():.6f}, std={s.std():.6f}, median={np.median(s):.6f}")
    return s


if __name__ == "__main__":
    main()
