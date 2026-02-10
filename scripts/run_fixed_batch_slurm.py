"""Submit bergson benchmarks with fixed batch size via Slurm.

Experimental: runs bergson with token_batch_size=4096 to match
dattri's batch_size=4 (4 examples x 1024 tokens). This lets us
see if the auto-tuned batch size (2048) is suboptimal.

Each job runs on an exclusive node to eliminate contention.
Results are saved to /projects/a6a/public/lucia/proj_bench_bergson_fixedbatch/.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

DATASET = "data/EleutherAI/SmolLM2-135M-10B-tokenized"
RUN_ROOT = Path("/projects/a6a/public/lucia/proj_bench_bergson_fixedbatch")
MAX_LENGTH = 1024
PROJ_DIM = 16
TOKEN_BATCH_SIZE = 4096  # Match dattri's batch_size=4
WORK_DIR = Path("/home/a6a/lucia.a6a/bergson")
PYTHON = "/home/a6a/lucia.a6a/miniforge3/bin/python"
TIME_LIMIT = "01:00:00"
PARTITION = "workq"

MODELS_WITH_PROJ = [
    "pythia-14m",
    "pythia-70m",
    "pythia-160m",
    "pythia-1b",
]
MODELS_WITHOUT_PROJ = [
    "pythia-14m",
    "pythia-70m",
    "pythia-160m",
]
TOKEN_SCALES = ["10K", "100K", "1M"]


@dataclass
class Job:
    projection: str  # "with" or "without"
    model: str
    tokens: str

    @property
    def label(self) -> str:
        proj = "proj" if self.projection == "with" else "noproj"
        return f"bergson_fb_{proj}_{self.model}_{self.tokens}"

    def bench_cmd(self) -> str:
        root = RUN_ROOT / f"bergson_{'proj' if self.projection == 'with' else 'noproj'}"
        proj_dim = PROJ_DIM if self.projection == "with" else 0
        return (
            f"{PYTHON} -m benchmarks.benchmark_bergson"
            f" {self.model} {self.tokens} {root}"
            f" --dataset {DATASET}"
            f" --projection_dim {proj_dim}"
            f" --max_length {MAX_LENGTH}"
            f" --skip_build true"
            f" --token_batch_size {TOKEN_BATCH_SIZE}"
        )


def build_jobs() -> list[Job]:
    jobs: list[Job] = []
    for projection in ("with", "without"):
        models = MODELS_WITH_PROJ if projection == "with" else MODELS_WITHOUT_PROJ
        for model in models:
            for tokens in TOKEN_SCALES:
                jobs.append(Job(projection, model, tokens))
    return jobs


def submit_job(job: Job) -> str | None:
    """Submit a single job via sbatch, return job ID."""
    log_dir = WORK_DIR / "runs" / "fixed_batch_slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    script = dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name={job.label}
        #SBATCH --output={log_dir / (job.label + ".out")}
        #SBATCH --error={log_dir / (job.label + ".err")}
        #SBATCH --partition={PARTITION}
        #SBATCH --nodes=1
        #SBATCH --exclusive
        #SBATCH --gres=gpu:1
        #SBATCH --time={TIME_LIMIT}

        cd {WORK_DIR}
        export CUDA_VISIBLE_DEVICES=0
        echo "Running on $(hostname)"
        echo "$ {job.bench_cmd()}"
        {job.bench_cmd()}
    """
    )

    result = subprocess.run(
        ["sbatch"],
        input=script,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"FAILED to submit {job.label}: {result.stderr}")
        return None

    # Parse "Submitted batch job 12345"
    job_id = result.stdout.strip().split()[-1]
    return job_id


def main() -> None:
    jobs = build_jobs()
    print(f"Submitting {len(jobs)} bergson fixed-batch jobs")
    print(f"token_batch_size={TOKEN_BATCH_SIZE}")
    print(f"Results dir: {RUN_ROOT}")
    print()

    submitted: list[tuple[Job, str]] = []
    failed: list[Job] = []

    for job in jobs:
        job_id = submit_job(job)
        if job_id:
            submitted.append((job, job_id))
            print(f"  [{job_id}] {job.label}")
        else:
            failed.append(job)

    print(f"\n{len(submitted)}/{len(jobs)} submitted")
    if failed:
        print(f"{len(failed)} failed:")
        for j in failed:
            print(f"  {j.label}")
        sys.exit(1)

    print("\nMonitor with:" "\n  squeue -u $USER | grep fb")


if __name__ == "__main__":
    main()
