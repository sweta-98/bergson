"""Precompute Hessian factors at multiple training checkpoints.

For each entry in ``approx_unrolling_cfg.checkpoints`` (an absolute path or
HF model ID), run the existing :func:`approximate_hessians` pipeline once.
Each checkpoint's output lands at
``<index_cfg.run_path>/segment_{l}/ckpt_{i}/<method>/``, where ``l`` is the
segment the checkpoint belongs to and ``i`` is its index *within* that
segment. EV correction is forced off because per-checkpoint lambda is
wasted work — the segment-averaging step recomputes it once it knows the
segment eigenbasis.
"""

import shutil
from copy import deepcopy
from pathlib import Path

from bergson.config import (
    ApproxUnrollingConfig,
    HessianConfig,
    IndexConfig,
)
from bergson.hessians.hessian_approximations import approximate_hessians
from bergson.utils.logger import get_logger


def precompute_checkpoint_hessians(
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    approx_unrolling_cfg: ApproxUnrollingConfig,
    *,
    overwrite: bool = False,
) -> None:
    """Run :func:`approximate_hessians` once per checkpoint.

    The pipeline-level divisibility check (``n_ckpts % n_segments == 0``)
    runs in :func:`approx_unrolling_pipeline` before this is called.
    """
    logger = get_logger("precompute_checkpoint_hessians")
    base_run = Path(index_cfg.run_path)
    method = hessian_cfg.method
    n_ckpts = len(approx_unrolling_cfg.checkpoints)
    n_segments = approx_unrolling_cfg.segments
    per_segment = n_ckpts // n_segments

    for c, ckpt in enumerate(approx_unrolling_cfg.checkpoints):
        ckpt = str(ckpt)
        seg = c // per_segment
        idx_in_seg = c % per_segment
        ckpt_dir = base_run / f"segment_{seg}" / f"ckpt_{idx_in_seg}"
        out_path = ckpt_dir / method

        if out_path.exists():
            if not overwrite:
                logger.info(
                    f"[seg {seg} ckpt {idx_in_seg}] skip — exists at {out_path}"
                )
                continue
            shutil.rmtree(out_path)

        logger.info(
            f"[seg {seg} ckpt {idx_in_seg}] computing {method} at model={ckpt!r}"
        )

        ckpt_index_cfg = deepcopy(index_cfg)
        ckpt_index_cfg.run_path = str(ckpt_dir)
        ckpt_index_cfg.model = ckpt

        ckpt_hessian_cfg = deepcopy(hessian_cfg)
        ckpt_hessian_cfg.ev_correction = False

        approximate_hessians(
            ckpt_index_cfg, ckpt_hessian_cfg, do_eigendecomposition=False
        )
