#!/usr/bin/env python3
"""Run a Trackstar ablation from a YAML config and save results to CSV.

Usage::

    python scripts/wikitext_wmdp_trackstar.py ablations/my_experiment.yaml

The YAML file specifies all bergson trackstar arguments. Results are saved to
a CSV with the same stem next to the YAML (e.g. ``ablations/my_experiment.csv``).

The CSV contains the top and bottom N scoring documents with columns:
    rank, direction, index, score, dataset, subset, split, text_first_100_words
"""

import argparse
import csv
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset


def run_trackstar(yaml_path: str) -> str:
    """Run bergson trackstar using a YAML config file, return the run_path."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}

    run_path = _get_cfg_value(cfg, "run_path")
    if not run_path:
        # Derive run_path from the YAML filename
        run_path = f"runs/{Path(yaml_path).stem}"

    cmd = ["bergson", "trackstar", "--config", yaml_path]
    if not cfg.get("run_path"):
        cmd.insert(2, run_path)

    print(f"Running: {' '.join(cmd)}")
    print()

    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)

    proc.wait()
    elapsed = time.monotonic() - t0
    mins, secs = divmod(elapsed, 60)
    print(f"\nTotal time: {int(mins)}m {secs:.1f}s")

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    return run_path


def first_n_words(text: str, n: int = 100) -> str:
    """Return the first n words of text, collapsed to a single line."""
    words = text.split()[:n]
    return " ".join(words)


def load_scores(run_path: str) -> np.ndarray:
    """Load scores from a trackstar run."""
    scores_dir = Path(run_path) / "scores"
    if not scores_dir.exists():
        raise FileNotFoundError(f"No scores found at {scores_dir}")

    # Standard trackstar score dtype
    dtype = np.dtype({
        "names": ["score_0", "written_0"],
        "formats": ["float32", "bool"],
        "offsets": [0, 4],
        "itemsize": 8,
    })
    scores = np.memmap(str(scores_dir / "scores.bin"), dtype=dtype, mode="r")
    return scores["score_0"]


def _get_cfg_value(cfg: dict, dotted_key: str, default=None):
    """Get a value from a YAML config supporting old and new Bergson schemas.

    Supported forms include:
    1. Structured YAML, e.g. ``index.data.dataset`` / ``trackstar.query.dataset``
    2. Nested legacy YAML, e.g. ``data.dataset`` / ``query.dataset``
    3. Flat dotted legacy YAML, e.g. ``cfg["data.dataset"]``
    """
    if dotted_key == "run_path":
        candidates = [
            ("index", "run_path"),
            ("run_path",),
        ]
    elif dotted_key.startswith("data."):
        suffix = tuple(dotted_key.split(".")[1:])
        candidates = [
            ("index", "data", *suffix),
            ("data", *suffix),
        ]
    elif dotted_key.startswith("query."):
        suffix = tuple(dotted_key.split(".")[1:])
        candidates = [
            ("trackstar", "query", *suffix),
            ("query", *suffix),
        ]
    else:
        candidates = [tuple(dotted_key.split("."))]

    for path in candidates:
        obj = cfg
        for part in path:
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                break
        else:
            return obj

    return cfg.get(dotted_key, default)


def save_results_csv(
    yaml_path: str,
    run_path: str,
    n: int = 10,
) -> str:
    """Save top/bottom N results to a CSV next to the YAML config."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    s = load_scores(run_path)

    # Print summary stats
    print(f"\nTotal documents scored: {len(s)}")
    print(f"Score mean: {s.mean():.6f}")
    print(f"Score std:  {s.std():.6f}")
    print(f"Score min:  {s.min():.6f}")
    print(f"Score max:  {s.max():.6f}")

    # Load the value dataset (supports both nested and dotted YAML keys)
    dataset_name = _get_cfg_value(cfg, "data.dataset", "NeelNanda/pile-10k")
    subset = _get_cfg_value(cfg, "data.subset")
    split = _get_cfg_value(cfg, "data.split", "train")

    load_kwargs = {"path": dataset_name, "split": split}
    if subset:
        load_kwargs["name"] = subset
    ds = load_dataset(**load_kwargs)

    # Determine top and bottom indices
    top_idx = np.argsort(s)[-n:][::-1]
    bot_idx = np.argsort(s)[:n]

    # CSV output path: same directory and stem as the YAML
    csv_path = Path(yaml_path).with_suffix(".csv")
    prompt_column = _get_cfg_value(cfg, "data.prompt_column", "text")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "direction",
            "index",
            "score",
            "dataset",
            "subset",
            "split",
            "text_first_100_words",
        ])

        for rank, idx in enumerate(top_idx, 1):
            text = ds[int(idx)][prompt_column]
            writer.writerow([
                rank,
                "top",
                int(idx),
                f"{s[idx]:.8f}",
                dataset_name,
                subset or "",
                split,
                first_n_words(text),
            ])

        for rank, idx in enumerate(bot_idx, 1):
            text = ds[int(idx)][prompt_column]
            writer.writerow([
                rank,
                "bottom",
                int(idx),
                f"{s[idx]:.8f}",
                dataset_name,
                subset or "",
                split,
                first_n_words(text),
            ])

    print(f"\nResults saved to {csv_path}")

    # Also print the results
    print(f"\n{'='*80}")
    print(f"TOP {n} SCORING DOCUMENTS")
    print(f"{'='*80}")
    for rank, idx in enumerate(top_idx, 1):
        text = ds[int(idx)][prompt_column]
        print(f"\n#{rank} | idx={idx} | score={s[idx]:.6f}")
        print(f"  {first_n_words(text)}")

    print(f"\n{'='*80}")
    print(f"BOTTOM {n} SCORING DOCUMENTS")
    print(f"{'='*80}")
    for rank, idx in enumerate(bot_idx, 1):
        text = ds[int(idx)][prompt_column]
        print(f"\n#{rank} | idx={idx} | score={s[idx]:.6f}")
        print(f"  {first_n_words(text)}")

    return str(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a Trackstar ablation from a YAML config."
    )
    parser.add_argument(
        "yaml_config",
        help="Path to YAML config file (e.g. ablations/my_experiment.yaml)",
    )
    parser.add_argument(
        "-n", "--top_n",
        type=int,
        default=10,
        help="Number of top/bottom results to save (default: 10)",
    )
    parser.add_argument(
        "--scores_only",
        action="store_true",
        help="Skip trackstar run, only generate CSV from existing scores.",
    )
    cli_args = parser.parse_args()

    if not cli_args.scores_only:
        run_path = run_trackstar(cli_args.yaml_config)
    else:
        with open(cli_args.yaml_config) as f:
            cfg = yaml.safe_load(f) or {}
        run_path = _get_cfg_value(
            cfg, "run_path", f"runs/{Path(cli_args.yaml_config).stem}"
        )

    save_results_csv(cli_args.yaml_config, run_path, n=cli_args.top_n)
