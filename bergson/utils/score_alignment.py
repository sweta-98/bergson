"""Permute attribution scores to match the order MAGIC's worker sees.

`bergson magic` shuffles the tokenized training dataset with
``run_cfg.seed`` (see ``bergson/magic/cli.py`` ``run_magic``).
``bergson ekfac`` and ``bergson trackstar`` build their indices on the
unshuffled order. To run ``bergson validate --scores ...`` against
EK-FAC or trakstar scores we therefore have to reorder them so that
``scores[i]`` lines up with ``train_ds[i]`` after the shuffle.
"""

import torch
from datasets import Dataset

from ..config import TrainingConfig
from ..magic.cli import attach_doc_ids_if_missing
from .worker_utils import setup_data_pipeline


def shuffle_perm_for_run(run_cfg: TrainingConfig) -> torch.Tensor:
    """Replicate the permutation `bergson magic` applies to the training set.

    Returns a 1-D LongTensor ``perm`` such that
    ``scores_unshuffled[perm]`` is in the order ``bergson validate``
    will index into. Length equals the number of rows in the tokenized
    training dataset (after doc-id attachment, before any batch-size
    padding — padding is appended later by the worker and does not
    affect the prefix used here).
    """
    train_ds, _ = setup_data_pipeline(run_cfg)
    train_ds = attach_doc_ids_if_missing(train_ds)
    shuffled: Dataset = train_ds.shuffle(seed=run_cfg.seed)

    # HF stores the shuffle as an Arrow table with a single "indices" column.
    # Materialize once into a tensor so callers can index scores directly.
    if shuffled._indices is None:
        # No shuffle was applied (shouldn't happen with seeded shuffle, but
        # be defensive): identity permutation.
        return torch.arange(len(train_ds), dtype=torch.long)

    indices = shuffled._indices["indices"].to_pylist()
    return torch.tensor(indices, dtype=torch.long)
