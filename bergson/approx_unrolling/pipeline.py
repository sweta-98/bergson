"""Top-level orchestrator for the SOURCE training-data attribution pipeline.

Mirrors :func:`bergson.hessians.pipeline.hessian_pipeline` in style: a flat
sequence of numbered steps, each delegating to existing distributed
primitives.

Distributed model lifecycle
---------------------------
Each step calls :func:`bergson.distributed.launch_distributed_run`, which:

* In multi-GPU mode (``world_size > 1``), spawns a fresh batch of child
  processes via ``mp.get_context("spawn").Process`` and joins them. Children
  die after the step completes, automatically reclaiming GPU memory and the
  loaded model — **no special cleanup needed across checkpoints**.
* In single-GPU mode (``world_size <= 1``), runs the worker inline in the
  parent process. The model loaded by ``setup_model_and_peft`` is local to
  the worker function and becomes eligible for GC once the worker returns.
  We additionally call :func:`gc.collect` between checkpoint passes to
  encourage prompt release. ``CLAUDE.md`` forbids
  ``torch.cuda.empty_cache``; the caching allocator will reuse freed
  blocks for the next checkpoint anyway.

This means the per-checkpoint loop in
:func:`bergson.approx_unrolling.checkpoint_hessians.precompute_checkpoint_hessians`
correctly loads-off and loads-in the next checkpoint on each iteration.

Pipeline steps
--------------
Currently only step 1 is implemented. Subsequent steps are stubbed and
will be filled in incrementally:

1. **Per-checkpoint Hessian precompute.** Run :func:`approximate_hessians`
   at each checkpoint, output to ``<run>/ckpt_{c}/<method>/``. EV correction
   is forced off — segment averaging recomputes lambda once it knows the
   segment eigenbasis.
2. *(TBD)* Per-segment covariance averaging + eigendecomposition (+ optional
   lambda).
3. *(TBD)* Per-checkpoint training-gradient build, then per-segment average.
4. *(TBD)* Build query gradient at the final checkpoint.
5. *(TBD)* Walk query through segments via ``F_S`` / ``F_r`` → ``ψ_ℓ``.
6. *(TBD)* Per-segment scoring, summed and divided by ``N``.
"""

import gc
import shutil
from copy import deepcopy
from pathlib import Path

from ..build import build
from ..config import (
    ApproxUnrollingConfig,
    HessianConfig,
    IndexConfig,
    PreprocessConfig,
)
from ..utils.logger import get_logger
from .checkpoint_hessians import precompute_checkpoint_hessians
from .checkpoint_lambdas import precompute_checkpoint_averaged_lambdas
from .segment_aggregation import (
    aggregate_segment_covariances,
    aggregate_segment_lambdas,
)

# Total number of steps in the full SOURCE pipeline. Used only for the
# user-facing "Step k/N_TOTAL_STEPS:" prefix. Bump as steps land.
_N_TOTAL_STEPS = 6


def approx_unrolling_pipeline(
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    approx_unrolling_cfg: ApproxUnrollingConfig,
    *,
    resume: bool = False,
):
    """Run the SOURCE (approximate unrolling) training-data attribution pipeline.

    Parameters
    ----------
    index_cfg : IndexConfig
        Base index/run config. ``index_cfg.run_path`` is the parent directory
        for **all** SOURCE artifacts (per-checkpoint outputs, per-segment
        outputs, transformed queries, scores).
    hessian_cfg : HessianConfig
        EKFAC method and dtype. ``ev_correction`` is forced off in step 1
        (per-checkpoint) because per-checkpoint lambda is wasted work; the
        eventual segment-averaging step recomputes it once per segment in
        the segment's averaged eigenbasis.
    approx_unrolling_cfg : ApproxUnrollingConfig
        SOURCE-specific config: list of checkpoint revisions, segment count,
        learning-rate schedule, etc.
    resume : bool
        If ``True``, skip steps whose output directories already exist.
    """
    logger = get_logger("approx_unrolling_pipeline")

    n_ckpts = len(approx_unrolling_cfg.checkpoints)
    n_segments = approx_unrolling_cfg.segments
    if n_ckpts == 0:
        raise ValueError("approx_unrolling_cfg.checkpoints is empty.")
    if n_segments < 1:
        raise ValueError(
            f"approx_unrolling_cfg.segments must be ≥ 1, got {n_segments}."
        )
    if n_ckpts % n_segments != 0:
        raise ValueError(
            f"checkpoints ({n_ckpts}) must be divisible by segments "
            f"({n_segments}); got {n_ckpts}/{n_segments} = "
            f"{n_ckpts / n_segments:.3f} per segment."
        )

    logger.info("=" * 70)
    logger.info(f"SOURCE pipeline → {index_cfg.run_path}")
    logger.info(f"  base model        : {index_cfg.model}")
    logger.info(f"  checkpoints (C)   : {len(approx_unrolling_cfg.checkpoints)}")
    logger.info(f"  segments (L)      : {approx_unrolling_cfg.segments}")
    logger.info(f"  hessian method    : {hessian_cfg.method}")
    logger.info(f"  resume            : {resume}")
    logger.info("=" * 70)

    # ── Step 1: Per-checkpoint Hessian precompute ─────────────────────────
    logger.info(
        f"Step 1/{_N_TOTAL_STEPS}: "
        f"Precomputing {hessian_cfg.method} factors at each checkpoint..."
    )
    precompute_checkpoint_hessians(
        index_cfg,
        hessian_cfg,
        approx_unrolling_cfg,
        resume=resume,
    )
    # Encourage GC between expensive steps; matters most in single-GPU mode
    # where the worker ran in-process and may still hold model references.
    gc.collect()

    # ── Step 2: Per-segment covariance aggregation + eigendecomposition ───
    logger.info(
        f"Step 2/{_N_TOTAL_STEPS}: "
        f"Aggregating per-checkpoint covariances into segment averages..."
    )
    aggregate_segment_covariances(
        run_path=index_cfg.run_path,
        method=hessian_cfg.method,
        n_segments=n_segments,
        per_segment=n_ckpts // n_segments,
        distributed=index_cfg.distributed,
        resume=resume,
    )

    # ── Step 3: Per-checkpoint lambda in segment eigenbasis ───────────────
    if hessian_cfg.ev_correction:
        logger.info(
            f"Step 3/{_N_TOTAL_STEPS}: "
            f"Per-checkpoint lambda using segment eigvecs..."
        )
        precompute_checkpoint_averaged_lambdas(
            index_cfg,
            hessian_cfg,
            approx_unrolling_cfg,
            resume=resume,
        )
    else:
        logger.info(f"Step 3/{_N_TOTAL_STEPS}: skipped (ev_correction=False).")

    # ── Step 4: Per-segment lambda aggregation ────────────────────────────
    if hessian_cfg.ev_correction:
        logger.info(
            f"Step 4/{_N_TOTAL_STEPS}: "
            f"Aggregating per-checkpoint lambdas into segment lambdas..."
        )
        aggregate_segment_lambdas(
            run_path=index_cfg.run_path,
            method=hessian_cfg.method,
            n_segments=n_segments,
            per_segment=n_ckpts // n_segments,
            distributed=index_cfg.distributed,
            resume=resume,
        )
    else:
        logger.info(f"Step 4/{_N_TOTAL_STEPS}: skipped (ev_correction=False).")

    # ── Step 5: Mean query gradient at the final checkpoint ───────────────
    logger.info(
        f"Step 5/{_N_TOTAL_STEPS}: "
        f"Building mean query gradient at the final checkpoint..."
    )
    query_path = Path(index_cfg.run_path) / "query"
    if resume and query_path.exists():
        logger.info(f"  skip — exists at {query_path}")
    else:
        if query_path.exists():
            shutil.rmtree(query_path)
        query_cfg = deepcopy(index_cfg)
        query_cfg.model = str(approx_unrolling_cfg.checkpoints[-1])
        query_cfg.data = approx_unrolling_cfg.query
        query_cfg.run_path = str(query_path)
        query_cfg.projection_dim = 0
        query_cfg.skip_hessians = True
        build(query_cfg, PreprocessConfig(aggregation="mean"))

    # ── Step 6: TBD ───────────────────────────────────────────────────────
    # 6. Walk query through segments via F_S / F_r → ψ_ℓ; per-segment score.
    logger.info(
        f"[SOURCE pipeline] steps 1-5 complete. "
        f"Step {_N_TOTAL_STEPS} not yet implemented."
    )
