import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from ..build import build
from ..config import (
    HessianConfig,
    HessianPipelineConfig,
    IndexConfig,
    PreprocessConfig,
    ScoreConfig,
)
from ..distributed import launch_distributed_run
from ..score.score import score_dataset
from ..utils.worker_utils import validate_run_path
from .apply_hessian import (
    EkfacConfig,
    apply_worker,
    build_projections_worker,
)
from .hessian_approximations import approximate_hessians


def _step_complete(path: str, resume: bool) -> bool:
    """Check if a step's output already exists and should be skipped."""
    if not resume:
        return False
    if Path(path).exists():
        print(f"  Skipping (output exists at {path})")
        return True
    return False


@contextmanager
def _timed(label: str, durations: dict[str, float]):
    """Time a pipeline step and print the wall-clock duration on exit."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        durations[label] = elapsed
        print(f"  [{label}] took {elapsed:.1f}s")


def hessian_pipeline(
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    score_cfg: ScoreConfig,
    preprocess_cfg: PreprocessConfig,
    hessian_pipeline_cfg: HessianPipelineConfig,
):
    """Run the full Hessian-preconditioned influence pipeline.

    1. Build mean query gradient.
    2. Fit Hessian factors (kfac, tkfac, shampoo) on the training dataset.
    3.   (legacy)      Apply the inverse Hessian to the mean query gradient.
       OR
       3.5 (compression) Compute R · cov^{-1/2} (precondition+sketch) and
                         project the query gradient with the resulting M.
    4. Score each training example against the transformed query gradient.
    """
    run_path = index_cfg.run_path
    method = hessian_cfg.method
    query_path = f"{run_path}/query"
    hessian_path = f"{run_path}/hessian"
    transformed_query_path = f"{run_path}/{method}_query"
    scores_path = f"{run_path}/scores"
    resume = hessian_pipeline_cfg.resume

    if index_cfg.projection_dim > 0 and hessian_cfg.ev_correction:
        raise ValueError(
            "Compression (projection_dim > 0) is incompatible with "
            "HessianConfig.ev_correction=True. Use a Kronecker-factored "
            "method (kfac, tkfac, shampoo) without ev_correction."
        )

    def _validate(cfg: IndexConfig):
        if resume and cfg.partial_run_path.exists():
            return
        validate_run_path(cfg)

    durations: dict[str, float] = {}

    # ── Step 1: Build mean query gradient ─────────────────────────────────
    print("Step 1/4: Building mean query gradient...")
    if not _step_complete(query_path, resume):
        with _timed("step1_build_query", durations):
            query_cfg = deepcopy(index_cfg)
            query_cfg.run_path = query_path
            query_cfg.data = hessian_pipeline_cfg.query
            query_cfg.projection_dim = 0
            query_cfg.skip_hessians = True
            _validate(query_cfg)

            query_preprocess_cfg = PreprocessConfig(aggregation="mean")
            build(query_cfg, query_preprocess_cfg)

    # ── Step 2: Fit Hessian factors on training data ──────────────────────
    print(f"Step 2/4: Fitting {method} factors on training data...")
    if not _step_complete(hessian_path, resume):
        with _timed("step2_fit_hessian", durations):
            hessian_index_cfg = deepcopy(index_cfg)
            hessian_index_cfg.run_path = hessian_path
            _validate(hessian_index_cfg)

        approximate_hessians(hessian_index_cfg, hessian_cfg)

    hessian_method_path = f"{hessian_path}/{method}"
    projections_path = f"{hessian_method_path}/projection_left_sharded"
    ekfac_cfg = EkfacConfig(
        hessian_method_path=hessian_method_path,
        gradient_path=query_path,
        run_path=transformed_query_path,
        ev_correction=hessian_cfg.ev_correction,
        lambda_damp_factor=hessian_pipeline_cfg.lambda_damp_factor,
        projection_dim=index_cfg.projection_dim,
        projection_type=index_cfg.projection_type,
    )

    if index_cfg.projection_dim > 0:
        # ── Step 3.5 (compression): build M = R · cov^{-1/2} before applying ──
        print("Step 3.5/4: Computing R · cov^{-1/2} (precondition+sketch)...")
        if not _step_complete(projections_path, resume):
            with _timed("step3.5_build_projections", durations):
                launch_distributed_run(
                    "build_projections",
                    build_projections_worker,
                    [ekfac_cfg],
                    index_cfg.distributed,
                )
    else:
        # ── Step 3 (legacy): Apply inverse Hessian via rotate-divide-rotate ──
        print(
            f"Step 3/4: Applying {method} inverse Hessian to mean query " "gradient..."
        )

    if not _step_complete(transformed_query_path, resume):
        launch_distributed_run(
            "apply_hessian",
            apply_worker,
            [ekfac_cfg],
            index_cfg.distributed,
        )

    # ── Step 4: Score training examples ───────────────────────────────────
    print("Step 4/4: Scoring training data against transformed query...")
    if not _step_complete(scores_path, resume):
        score_index_cfg = deepcopy(index_cfg)
        score_index_cfg.run_path = scores_path
        score_index_cfg.skip_hessians = True
        if index_cfg.projection_dim > 0:
            score_index_cfg.kfac_projection_path = hessian_method_path
        score_cfg.query_path = transformed_query_path
        score_cfg.higher_is_better = True
        _validate(score_index_cfg)

        score_dataset(score_index_cfg, score_cfg, preprocess_cfg)

    print(f"Done! Scores saved to: {scores_path}")
    if durations:
        total = sum(durations.values())
        print(f"Step timings (s): {durations} | total {total:.1f}s")
