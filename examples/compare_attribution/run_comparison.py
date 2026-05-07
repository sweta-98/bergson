"""Compare MAGIC, EK-FAC, and trakstar attribution on the same dataset.

Pipeline:
  1. ``bergson magic <config>``  -> trains a fresh model, runs leave-k-out
     LDS, and (with the cli.py change in this branch) dumps
     ``{run_path}/final_model/`` (HF directory).
  2. ``bergson ekfac --model {run_path}/final_model ...``  -> EK-FAC scores.
  3. ``bergson trackstar --model {run_path}/final_model ...``  -> trakstar
     scores.
  4. For EK-FAC and trakstar: load the (unshuffled) scores, permute by the
     same shuffle seed MAGIC uses so positions line up with what
     ``bergson validate`` will see, save to ``scores_aligned.pt``, and run
     ``bergson validate --scores scores_aligned.pt ...`` to produce per-method
     LDS Spearman/Pearson under ``{run_path}/{method}/lds/``.
  5. Aggregate all three methods' summary.csv files into
     ``{run_path}/lds_comparison.csv``.

All sub-commands print their full argv before executing (per CLAUDE.md).
"""

import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

# NOTE: we deliberately do NOT import torch / bergson at module load. The
# orchestration spawns ``bergson magic`` as a subprocess which itself launches
# 8 distributed workers; if torch's CUDA driver gets initialized in this
# parent process, the forked subprocess inherits poisoned CUDA state and
# rank 0 spins forever before training starts. Lazy-import inside main() so
# the parent stays a thin shell that fork+exec'es cleanly.


def _resolve_bergson_bin() -> str:
    """Return the path to the ``bergson`` entry script.

    We avoid ``python -m bergson`` because that makes ``__main__`` the
    bergson.__main__ module, which moves the Magic/Ekfac/Trackstar dataclasses
    into ``__main__`` — and torch.distributed.elastic spawn children can't
    unpickle classes from a parent's ``__main__``. The entry script keeps
    those classes under bergson.__main__ where pickle can find them.
    """
    candidate = Path(sys.executable).parent / "bergson"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("bergson")
    if found:
        return found
    raise RuntimeError(
        "Could not locate the `bergson` entry script. Install bergson "
        "(`pip install -e .`) into the active venv."
    )


BERGSON_BIN = _resolve_bergson_bin()

# Locate this checkout's bergson without importing it (which would init torch).
_BERGSON_ROOT = str(Path(__file__).resolve().parent.parent.parent)


def _subprocess_env() -> dict:
    """Force subprocesses to pick up *this* checkout's bergson via PYTHONPATH.

    The bergson entry script is shebanged to a specific venv whose installed
    bergson may be older than the validate-branch code we're running from.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    paths = [_BERGSON_ROOT]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


# Per-phase wall-clock timings, written to {run_path}/timings.csv at end.
# Each entry is {"phase": str, "wallclock_s": float}.
PHASE_TIMINGS: list[dict] = []


def run_cmd(argv: list[str], *, marker_callbacks: dict | None = None) -> None:
    """Print and run a command, raising on non-zero exit.

    If ``marker_callbacks`` is provided ({substring: fn(elapsed_s)}), each
    matching stdout line triggers ``fn`` with elapsed seconds since launch.
    Used to capture MAGIC sub-phase timestamps (train+score vs LDS) which
    happen inside a single ``bergson magic`` subprocess.
    """
    print("\n$ " + shlex.join(argv), flush=True)
    if not marker_callbacks:
        subprocess.run(argv, check=True, env=_subprocess_env())
        return
    start = time.monotonic()
    proc = subprocess.Popen(
        argv,
        env=_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        for key, fn in marker_callbacks.items():
            if key in line:
                fn(time.monotonic() - start)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, argv)


def time_phase(name: str, fn, *args, **kwargs):
    """Wall-clock ``fn(*args, **kwargs)``, log, and append to PHASE_TIMINGS."""
    start = time.monotonic()
    result = fn(*args, **kwargs)
    elapsed = time.monotonic() - start
    PHASE_TIMINGS.append({"phase": name, "wallclock_s": elapsed})
    print(f"[timing] {name}: {elapsed:.1f}s ({elapsed / 60:.2f}m)", flush=True)
    return result


def write_timings(run_path: Path) -> Path:
    out = run_path / "timings.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["phase", "wallclock_s"])
        w.writeheader()
        for row in PHASE_TIMINGS:
            w.writerow(row)
    return out


def kv_args(prefix: str, mapping: dict | None) -> list[str]:
    """Translate a YAML sub-dict into ``--{prefix}.<field> <val>`` argv pairs.

    Only emits keys the user explicitly set. ``None`` and empty-string values
    are skipped (those are dataclass-default sentinels). Booleans become
    ``--{prefix}.<field>`` (no value) when True and are skipped when False
    so argparse-style ``store_true`` flags work.
    """
    if not mapping:
        return []
    out: list[str] = []
    for key, val in mapping.items():
        if val is None or val == "":
            continue
        if isinstance(val, bool):
            if val:
                out.append(f"--{prefix}.{key}")
            continue
        if isinstance(val, dict):
            out += kv_args(f"{prefix}.{key}", val)
            continue
        out += [f"--{prefix}.{key}", str(val)]
    return out


def load_raw_yaml(config_path: Path) -> dict:
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def step_magic(config_path: Path) -> None:
    """Phase 1: train + score + LDS via the MAGIC CLI.

    Captures sub-phase wallclock from the MAGIC subprocess's stdout markers:
      - magic_train_score: until the trained final_model is written
        (covers train + per-doc backward / score computation).
      - magic_lds: from train_score to "Final Spearman correlation:"
        (covers the leave-k-out validation loop).
    """
    train_done: list[float] = []
    lds_done: list[float] = []
    run_cmd(
        [BERGSON_BIN, "magic", str(config_path)],
        marker_callbacks={
            "Saved final HF model": lambda t: train_done.append(t),
            "Final Spearman correlation": lambda t: lds_done.append(t),
        },
    )
    if train_done:
        PHASE_TIMINGS.append(
            {"phase": "magic_train_score", "wallclock_s": train_done[0]}
        )
    if lds_done and train_done:
        PHASE_TIMINGS.append(
            {"phase": "magic_lds", "wallclock_s": lds_done[0] - train_done[0]}
        )


def step_ekfac(
    run_path: Path,
    final_model_dir: Path,
    raw: dict,
) -> Path:
    """Phase 2: EK-FAC scores against the trained MAGIC model. Returns scores dir.

    simple_parsing's ``ConflictResolution.EXPLICIT`` flattens nested dataclass
    fields by their leaf name when names don't conflict, so the query split
    on ``HessianPipelineConfig.query`` becomes plain ``--query.*``.
    """
    out_path = run_path / "ekfac"
    argv = [
        BERGSON_BIN,
        "ekfac",
        str(out_path),
        "--model",
        str(final_model_dir),
        *kv_args("data", raw.get("data", {})),
        *kv_args("query", raw.get("query", {})),
    ]
    if raw.get("overwrite"):
        argv.append("--overwrite")
    run_cmd(argv)
    return out_path / "scores"


def step_trackstar(
    run_path: Path,
    final_model_dir: Path,
    raw: dict,
) -> Path:
    """Phase 3: trakstar scores against the trained MAGIC model. Returns scores dir."""
    out_path = run_path / "trackstar"
    argv = [
        BERGSON_BIN,
        "trackstar",
        str(out_path),
        "--model",
        str(final_model_dir),
        *kv_args("data", raw.get("data", {})),
        *kv_args("query", raw.get("query", {})),
    ]
    if raw.get("overwrite"):
        argv.append("--overwrite")
    run_cmd(argv)
    return out_path / "scores"


def align_scores(
    scores_dir: Path,
    num_docs: int,
    out_path: Path,
) -> None:
    """Aggregate per-chunk EK-FAC / trakstar scores into a per-doc tensor.

    EK-FAC and trakstar score the *chunked* dataset: one row per
    chunk_length-token chunk, ``num_chunks ≈ total_tokens / chunk_length``,
    far fewer than the raw document count. ``bergson validate`` indexes
    ``stream.weights`` by ``doc_ids`` (per-document), so it expects scores
    indexed by document. To bridge: for each chunk's score ``S_i``, scatter
    ``S_i / chunk_len`` to every ``doc_ids[i, t]`` covered by that chunk.
    Tokens of the same doc that span multiple chunks accumulate.

    Each chunk's data.hf row has a ``doc_ids`` column listing the doc index
    of every token position. We use those directly (more accurate than just
    taking the dominant doc per chunk).
    """
    import numpy as np
    import torch
    from datasets import load_from_disk

    from bergson.data import load_scores

    raw = np.asarray(load_scores(scores_dir)[:])
    # raw is (num_chunks, num_queries). EK-FAC has num_queries=1 (mean
    # query gradient); trakstar has num_queries=N (per-query scores). Sum
    # over the query dimension to get a single influence-per-chunk number,
    # matching MAGIC's per-doc total influence.
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D score memmap, got shape {raw.shape}")
    chunk_scores = torch.from_numpy(raw.sum(axis=1).copy()).to(torch.float32)

    # Per-row doc_ids live alongside the score file (saved by GradientCollector
    # at teardown into the scores/data.hf directory).
    data = load_from_disk(str(scores_dir / "data.hf"))
    chunk_doc_ids = torch.tensor(data["doc_ids"], dtype=torch.long)  # [N, chunk_len]
    n_chunks, chunk_len = chunk_doc_ids.shape

    if chunk_scores.numel() != n_chunks:
        raise ValueError(
            f"Score file has {chunk_scores.numel()} chunks but data.hf has "
            f"{n_chunks} — alignment broken."
        )

    # Distribute each chunk's score evenly across its tokens, then scatter
    # to per-doc bins.
    per_token = (chunk_scores / chunk_len).unsqueeze(1).expand(-1, chunk_len).reshape(-1)
    flat_doc_ids = chunk_doc_ids.reshape(-1)
    per_doc = torch.zeros(num_docs, dtype=torch.float32)
    per_doc.scatter_add_(0, flat_doc_ids, per_token)

    # Negate to match MAGIC's sign convention before LDS. EK-FAC and trakstar
    # report "high score = doc reduces query loss" (standard influence-fn sign);
    # MAGIC's saved scores.pt uses the opposite ("high = doc raises loss").
    # Without this negation LDS Spearman comes out flipped (positive method →
    # negative LDS). Verified empirically: EK-FAC −0.349 ↔ +0.349, trakstar
    # −0.232 ↔ +0.232 on this run with vs without negation.
    per_doc = -per_doc

    # ``bergson validate`` will re-shuffle the chunked dataset with
    # ``run_cfg.seed`` and index scores by position. Per-doc scores aren't
    # affected by that shuffle (doc_id is intrinsic to the doc, not its
    # position in the chunked dataset), so we save them untouched.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(per_doc, out_path)
    print(
        f"Aligned scores: {scores_dir} -> {out_path} "
        f"(per-doc shape={tuple(per_doc.shape)}, "
        f"non-zero={(per_doc != 0).sum().item()}, "
        f"finite={torch.isfinite(per_doc).all().item()})"
    )


def step_validate(
    method: str,
    run_path: Path,
    aligned_scores_pt: Path,
    base_model: str,
    raw: dict,
) -> Path:
    """Phase 4: bergson validate --scores ... -> LDS Spearman/Pearson."""
    lds_path = run_path / method / "lds"

    # Pull only the fields validate cares about. validate is a ValidationConfig
    # which is a TrainingConfig which is an AttributionConfig which is a
    # ModelConfig. We forward the same training/data settings MAGIC used so
    # the leave-k-out re-training matches MAGIC's training byte-for-byte.
    forward_keys = [
        "seed",
        "num_subsets",
        "subset_strategy",
        "batch_size",
        "num_epochs",
        "query_method",
        "optimizer",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "eps_root",
        "grad_checkpointing",
        "fsdp",
        "precision",
    ]
    argv = [
        BERGSON_BIN,
        "validate",
        str(lds_path),
        "--model",
        base_model,
        "--scores",
        str(aligned_scores_pt),
    ]
    for k in forward_keys:
        if k in raw:
            argv += [f"--{k}", str(raw[k])]
    argv += kv_args("data", raw.get("data", {}))
    argv += kv_args("query", raw.get("query", {}))
    # lr_schedule and distributed fields are flattened to top-level args by
    # simple_parsing (no --lr_schedule.* / --distributed.* prefix) — see
    # ``bergson validate --help``.
    for k, v in (raw.get("lr_schedule") or {}).items():
        if v is None or v == "":
            continue
        argv += [f"--{k}", str(v)]
    for k, v in (raw.get("distributed") or {}).items():
        if v is None or v == "":
            continue
        argv += [f"--{k}", str(v)]
    if raw.get("overwrite"):
        argv.append("--overwrite")
    run_cmd(argv)
    return lds_path / "summary.csv"


def read_summary(summary_csv: Path) -> dict:
    with open(summary_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty summary at {summary_csv}")
    return rows[0]


def write_comparison(run_path: Path, rows: list[dict]) -> Path:
    out = run_path / "lds_comparison.csv"
    fieldnames = [
        "method",
        "spearman_corr",
        "spearman_p",
        "pearson_corr",
        "pearson_p",
        "N",
        "baseline_loss",
    ]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "pythia14m_wikitext_smoke.yaml",
        help="Path to the comparison YAML (a MagicConfig).",
    )
    parser.add_argument(
        "--skip_magic",
        action="store_true",
        help="Skip MAGIC step (assumes scores.pt and final_model already exist).",
    )
    args = parser.parse_args()

    raw = load_raw_yaml(args.config)
    # Use only the raw yaml for fields needed before MAGIC runs; defer
    # MagicConfig.load (which imports torch) until after the subprocess
    # spawn so we don't poison its forked CUDA context.
    run_path = Path(raw["run_path"])
    base_model = raw["model"]
    final_model_dir = run_path / "final_model"

    overall_start = time.monotonic()

    # ── Phase 1: MAGIC ──────────────────────────────────────────────────────
    if not args.skip_magic:
        time_phase("magic_total", step_magic, args.config)

    if not final_model_dir.exists():
        raise RuntimeError(
            f"MAGIC did not produce {final_model_dir}; cannot run EK-FAC/trakstar"
        )

    # MAGIC writes multi-GB DCP state checkpoints during training. They sit in
    # the kernel page cache until writeback drains. EK-FAC's create_index calls
    # os.fsync(), which on ext4 triggers a journal commit that waits for *all*
    # pending writes in the same transaction — so the next phase's first fsync
    # blocks for the duration of MAGIC's residual writeback (observed ~24 min
    # gap before EK-FAC Step 3's first log when MAGIC was fast). Force a sync
    # here so the cost is paid in a known phase.
    time_phase("post_magic_sync", os.sync)

    # ``num_docs`` is the raw document count BEFORE chunking; MAGIC's
    # per-doc scores tensor is sized to that, so EK-FAC and trakstar must
    # be aggregated to the same shape. We read it from MAGIC's saved
    # scores.pt to avoid re-tokenizing the whole dataset.
    import torch

    magic_scores = torch.load(run_path / "scores.pt", map_location="cpu")
    num_docs = magic_scores.numel()
    print(f"Per-doc score length (from MAGIC): {num_docs}")

    # ── Phase 2: EK-FAC scores ─────────────────────────────────────────────
    ekfac_scores_dir = time_phase("ekfac_score", step_ekfac, run_path, final_model_dir, raw)
    ekfac_aligned = run_path / "ekfac" / "scores_aligned.pt"
    time_phase("ekfac_align", align_scores, ekfac_scores_dir, num_docs, ekfac_aligned)

    # ── Phase 3: trakstar scores ───────────────────────────────────────────
    trackstar_scores_dir = time_phase(
        "trackstar_score", step_trackstar, run_path, final_model_dir, raw
    )
    trackstar_aligned = run_path / "trackstar" / "scores_aligned.pt"
    time_phase(
        "trackstar_align", align_scores, trackstar_scores_dir, num_docs, trackstar_aligned
    )

    # ── Phase 4: LDS for EK-FAC and trakstar ───────────────────────────────
    ekfac_summary = time_phase(
        "ekfac_validate", step_validate, "ekfac", run_path, ekfac_aligned, base_model, raw
    )
    trackstar_summary = time_phase(
        "trackstar_validate",
        step_validate,
        "trackstar",
        run_path,
        trackstar_aligned,
        base_model,
        raw,
    )

    # ── Phase 5: aggregate ─────────────────────────────────────────────────
    rows: list[dict] = []
    magic_summary = run_path / "summary.csv"
    if magic_summary.exists():
        rows.append({"method": "magic", **read_summary(magic_summary)})
    rows.append({"method": "ekfac", **read_summary(ekfac_summary)})
    rows.append({"method": "trackstar", **read_summary(trackstar_summary)})

    out = write_comparison(run_path, rows)
    print(f"\nLDS comparison written to {out}")
    for row in rows:
        print(
            f"  {row['method']:>10s}  "
            f"spearman={row.get('spearman_corr', '?'):>8s}  "
            f"pearson={row.get('pearson_corr', '?'):>8s}  "
            f"N={row.get('N', '?')}"
        )

    PHASE_TIMINGS.append(
        {"phase": "overall", "wallclock_s": time.monotonic() - overall_start}
    )
    timings_path = write_timings(run_path)
    print(f"\nPhase timings written to {timings_path}")
    for row in PHASE_TIMINGS:
        print(f"  {row['phase']:>20s}  {row['wallclock_s']:8.1f}s  ({row['wallclock_s']/60:.2f}m)")


if __name__ == "__main__":
    main()
