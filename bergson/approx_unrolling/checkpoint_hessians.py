"""Precompute Hessian factors at multiple training checkpoints.

For each entry in ``approx_unrolling_cfg.checkpoints`` (an absolute path or
HF model ID), run the existing :func:`approximate_hessians` pipeline once.
Each checkpoint's output lands at ``<index_cfg.run_path>/ckpt_{c}/<method>/``.
EV correction is forced off because per-checkpoint lambda is wasted work —
the segment-averaging step recomputes it once it knows the segment
eigenbasis.
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
    resume: bool = False,
) -> None:
    """Run :func:`approximate_hessians` once per checkpoint."""
    logger = get_logger("precompute_checkpoint_hessians")
    base_run = Path(index_cfg.run_path)
    method = hessian_cfg.method

    for c, ckpt in enumerate(approx_unrolling_cfg.checkpoints):
        ckpt = str(ckpt)
        out_path = base_run / f"ckpt_{c}" / method

        if out_path.exists():
            if resume:
                logger.info(f"[ckpt {c}] skip — exists at {out_path}")
                continue
            shutil.rmtree(out_path)

        logger.info(f"[ckpt {c}] computing {method} at model={ckpt!r}")

        ckpt_index_cfg = deepcopy(index_cfg)
        ckpt_index_cfg.run_path = str(base_run / f"ckpt_{c}")
        ckpt_index_cfg.model = ckpt

        ckpt_hessian_cfg = deepcopy(hessian_cfg)
        ckpt_hessian_cfg.ev_correction = False

        approximate_hessians(ckpt_index_cfg, ckpt_hessian_cfg)
