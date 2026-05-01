"""Precompute Hessian factors at multiple training checkpoints.

This is the first step of the SOURCE pipeline. For each checkpoint listed in
``source_cfg.checkpoints``, run the existing
:func:`bergson.hessians.hessian_approximations.approximate_hessians` pipeline
and stash its output under ``<index_cfg.run_path>/ckpt_{c}/<method>/``.

Each entry of ``source_cfg.checkpoints`` is interpreted as a Hugging Face
revision on the base repo ``index_cfg.model``. The base model itself is left
untouched; we just swap ``index_cfg.revision`` per pass and rely on
``approximate_hessians`` (and its inner ``setup_model_and_peft``) to load the
right snapshot.

EV correction (the EKFAC ``lambda`` pass) is forced **off** here. The reason:
when SOURCE later partitions checkpoints into segments and averages the raw
covariances per segment, the resulting eigenbasis differs from any individual
checkpoint's eigenbasis, so per-checkpoint ``lambda`` factors would have to be
discarded. We recompute ``lambda`` once per segment in a later step.
TODO: Check this carefully
"""

import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import Serializable

from bergson.config import (
    ApproxUnrollingConfig,
    HessianConfig,
    IndexConfig,
)
from bergson.hessians.hessian_approximations import approximate_hessians


@dataclass
class CheckpointHessianResult:
    """One per checkpoint, returned for downstream pipeline use."""

    checkpoint: str
    """The revision string passed in via ``source_cfg.checkpoints``."""

    run_path: Path
    """Per-checkpoint run path: ``<base_run>/ckpt_{c}``."""

    hessian_path: Path
    """Per-checkpoint hessian-output path: ``<base_run>/ckpt_{c}/<method>``."""

    skipped: bool
    """True iff resume-mode found existing output and we did not recompute."""


def precompute_checkpoint_hessians(
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    source_cfg: ApproxUnrollingConfig,
    *,
    resume: bool = True,
) -> list[CheckpointHessianResult]:
    """Run :func:`approximate_hessians` once per checkpoint.

    Parameters
    ----------
    index_cfg : IndexConfig
        Base config. ``index_cfg.model`` is treated as the base repo. The
        ``run_path`` is treated as the **parent** directory under which
        per-checkpoint outputs live.
    hessian_cfg : HessianConfig
        Hessian config. ``ev_correction`` is overridden to ``False`` for each
        per-checkpoint call regardless of what the caller passed in.
    source_cfg : ApproxUnrollingConfig
        Provides the list of checkpoint revision strings.
    resume : bool
        If ``True`` (default), skip a checkpoint whose
        ``ckpt_{c}/<method>`` directory already exists.

    Returns
    -------
    list[CheckpointHessianResult]
        One result per checkpoint, in the same order as
        ``source_cfg.checkpoints``.
    """
    if not source_cfg.checkpoints:
        raise ValueError("source_cfg.checkpoints is empty; nothing to precompute.")

    base_run_path = Path(index_cfg.run_path)
    method = hessian_cfg.method

    # ev_correction is wasted work here; segment averaging owns lambda.
    if hessian_cfg.ev_correction:
        print(
            "precompute_checkpoint_hessians: forcing hessian_cfg.ev_correction=False "
            "for per-checkpoint passes (lambda is recomputed per segment)."
        )

    # Print the equivalent CLI invocations so a user could replay any single
    # checkpoint by hand if it crashes mid-loop.
    _print_replay_commands(index_cfg, hessian_cfg, source_cfg)

    rank = index_cfg.distributed.rank

    results: list[CheckpointHessianResult] = []
    n = len(source_cfg.checkpoints)
    for c, ckpt in enumerate(source_cfg.checkpoints):
        ckpt = str(ckpt)
        ckpt_run_path = base_run_path / f"ckpt_{c}"
        out_path = ckpt_run_path / method

        if resume and out_path.exists():
            if rank == 0:
                print(
                    f"[ckpt {c + 1}/{n}] skip ({ckpt!r}) — output exists at {out_path}"
                )
            results.append(
                CheckpointHessianResult(
                    checkpoint=ckpt,
                    run_path=ckpt_run_path,
                    hessian_path=out_path,
                    skipped=True,
                )
            )
            continue

        if rank == 0:
            print(
                f"[ckpt {c + 1}/{n}] computing {method} at revision={ckpt!r} "
                f"→ {out_path}"
            )
            # Clean any half-written attempt from a previous crash.
            partial = ckpt_run_path / f"{method}.part"
            if partial.exists():
                print(f"  removing stale partial dir: {partial}")
                shutil.rmtree(partial)
            if out_path.exists():
                # resume=False fall-through: caller wants a clean rebuild.
                print(f"  removing existing dir: {out_path}")
                shutil.rmtree(out_path)
            ckpt_run_path.mkdir(parents=True, exist_ok=True)

        ckpt_index_cfg = deepcopy(index_cfg)
        ckpt_index_cfg.run_path = str(ckpt_run_path)
        ckpt_index_cfg.revision = ckpt

        ckpt_hessian_cfg = deepcopy(hessian_cfg)
        ckpt_hessian_cfg.ev_correction = False

        approximate_hessians(ckpt_index_cfg, ckpt_hessian_cfg)

        results.append(
            CheckpointHessianResult(
                checkpoint=ckpt,
                run_path=ckpt_run_path,
                hessian_path=out_path,
                skipped=False,
            )
        )

    if rank == 0:
        n_done = sum(not r.skipped for r in results)
        n_skip = sum(r.skipped for r in results)
        print(
            f"precompute_checkpoint_hessians: done. "
            f"{n_done} computed, {n_skip} skipped, {n} total."
        )
    return results


def _print_replay_commands(
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    source_cfg: ApproxUnrollingConfig,
) -> None:
    """Print one ``bergson hessian …`` line per checkpoint for reproducibility.

    CLAUDE.md asks any subprocess-launching script to print the equivalent CLI
    so a user can rerun a single step. We don't actually launch subprocesses
    here, but the per-checkpoint passes are the kind of thing users will want
    to debug individually, so we still emit the commands.
    """
    if index_cfg.distributed.rank != 0:
        return

    base_run_path = Path(index_cfg.run_path)
    for c, ckpt in enumerate(source_cfg.checkpoints):
        ckpt_run_path = base_run_path / f"ckpt_{c}"
        cmd = (
            f"bergson hessian "
            f"--index_cfg.run_path {ckpt_run_path} "
            f"--index_cfg.model {index_cfg.model} "
            f"--index_cfg.revision {ckpt} "
            f"--hessian_cfg.method {hessian_cfg.method} "
            f"--hessian_cfg.ev_correction false"
        )
        print(f"# replay [ckpt {c}]: {cmd}")


@dataclass
class _CheckpointHessianCliConfig(Serializable):
    """Standalone CLI for testing this step in isolation."""

    index_cfg: IndexConfig
    hessian_cfg: HessianConfig
    source_cfg: ApproxUnrollingConfig
    resume: bool = True


if __name__ == "__main__":
    # Allow running the precompute step on its own:
    #   python -m bergson.approx_unrolling.checkpoint_hessians \
    #       --index_cfg.run_path runs/source_smoke \
    #       --index_cfg.model EleutherAI/pythia-14m \
    #       --source_cfg.checkpoints step1000 step2000 step3000 \
    #       --hessian_cfg.method kfac
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(_CheckpointHessianCliConfig, dest="cfg")
    cli_cfg: _CheckpointHessianCliConfig = parser.parse_args().cfg

    precompute_checkpoint_hessians(
        cli_cfg.index_cfg,
        cli_cfg.hessian_cfg,
        cli_cfg.source_cfg,
        resume=cli_cfg.resume,
    )
