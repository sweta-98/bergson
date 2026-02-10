"""Submit all projection comparison benchmarks via Slurm.

Each job runs on an exclusive node to eliminate contention.
Results are saved to /projects/a6a/public/lucia/proj_bench/.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

DATASET = "data/EleutherAI/SmolLM2-135M-10B-tokenized"
RUN_ROOT = Path("/projects/a6a/public/lucia/proj_bench")
MAX_LENGTH = 1024
PROJ_DIM = 16
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
    method: str  # "bergson" or "dattri"
    projection: str  # "with" or "without"
    model: str
    tokens: str

    @property
    def label(self) -> str:
        proj = "proj" if self.projection == "with" else "noproj"
        return f"{self.method}_{proj}_{self.model}_{self.tokens}"

    def bench_cmd(self) -> str:
        if self.method == "bergson":
            root = (
                RUN_ROOT
                / f"bergson_{'proj' if self.projection == 'with' else 'noproj'}"
            )
            proj_dim = PROJ_DIM if self.projection == "with" else 0
            return (
                f"{PYTHON} -m benchmarks.benchmark_bergson"
                f" {self.model} {self.tokens} {root}"
                f" --dataset {DATASET}"
                f" --projection_dim {proj_dim}"
                f" --max_length {MAX_LENGTH}"
                f" --skip_build true"
            )
        else:
            root = (
                RUN_ROOT / f"dattri_{'proj' if self.projection == 'with' else 'noproj'}"
            )
            cmd = (
                f"{PYTHON} -m benchmarks.benchmark_dattri"
                f" --model {self.model}"
                f" --train_tokens {self.tokens}"
                f" --run_root {root}"
                f" --dataset {DATASET}"
                f" --max_length {MAX_LENGTH}"
            )
            if self.projection == "with":
                cmd += f" --projection_dim {PROJ_DIM}"
            return cmd


def build_jobs() -> list[Job]:
    jobs: list[Job] = []
    for method in ("bergson", "dattri"):
        for projection in ("with", "without"):
            models = (
                MODELS_WITH_PROJ
                if method == "bergson" and projection == "with"
                else MODELS_WITHOUT_PROJ
            )
            for model in models:
                for tokens in TOKEN_SCALES:
                    jobs.append(Job(method, projection, model, tokens))
    return jobs


def submit_job(job: Job) -> str | None:
    """Submit a single job via sbatch, return job ID."""
    log_dir = WORK_DIR / "runs" / "proj_slurm_logs"
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
        export PYTHONDONTWRITEBYTECODE=1
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
    print(f"Submitting {len(jobs)} jobs to Slurm")
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

    print(
        "\nMonitor with:"
        f"\n  squeue -u $USER --name={','.join(j.label for j, _ in submitted[:5])}..."
        "\n  or: squeue -u $USER | grep proj"
    )


if __name__ == "__main__":
    main()
