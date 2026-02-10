"""Submit dattri benchmarks with bergson-equivalent batch sizes via Slurm.

Experimental: runs dattri with batch sizes matching bergson's auto-tuned
token_batch_size // seq_len, to see if larger batches improve dattri too.

Results saved to /projects/a6a/public/lucia/proj_bench_dattri_largebatch/.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

DATASET = "data/EleutherAI/SmolLM2-135M-10B-tokenized"
RUN_ROOT = Path("/projects/a6a/public/lucia/proj_bench_dattri_largebatch")
MAX_LENGTH = 1024
PROJ_DIM = 16
WORK_DIR = Path("/home/a6a/lucia.a6a/bergson")
PYTHON = "/home/a6a/lucia.a6a/miniforge3/bin/python"
TIME_LIMIT = "01:00:00"
PARTITION = "workq"

# Batch sizes derived from bergson auto-tuned token_batch_size // 1024
BATCH_SIZES = {
    ("pythia-14m", "with"): 128,
    ("pythia-70m", "with"): 128,
    ("pythia-160m", "with"): 128,
    ("pythia-14m", "without"): 128,
    ("pythia-70m", "without"): 128,
    ("pythia-160m", "without"): 64,
}

MODELS = ["pythia-14m", "pythia-70m", "pythia-160m"]
TOKEN_SCALES = ["10K", "100K", "1M"]


@dataclass
class Job:
    projection: str
    model: str
    tokens: str

    @property
    def batch_size(self) -> int:
        return BATCH_SIZES[(self.model, self.projection)]

    @property
    def label(self) -> str:
        proj = "proj" if self.projection == "with" else "noproj"
        return f"dattri_lb_{proj}_{self.model}_{self.tokens}"

    def bench_cmd(self) -> str:
        proj = "proj" if self.projection == "with" else "noproj"
        root = RUN_ROOT / f"dattri_{proj}"
        cmd = (
            f"{PYTHON} -m benchmarks.benchmark_dattri"
            f" --model {self.model}"
            f" --train_tokens {self.tokens}"
            f" --run_root {root}"
            f" --dataset {DATASET}"
            f" --max_length {MAX_LENGTH}"
            f" --batch_size {self.batch_size}"
        )
        if self.projection == "with":
            cmd += f" --projection_dim {PROJ_DIM}"
        return cmd


def build_jobs() -> list[Job]:
    jobs: list[Job] = []
    for projection in ("with", "without"):
        for model in MODELS:
            for tokens in TOKEN_SCALES:
                jobs.append(Job(projection, model, tokens))
    return jobs


def submit_job(job: Job) -> str | None:
    """Submit a single job via sbatch, return job ID."""
    log_dir = WORK_DIR / "runs" / "dattri_lb_slurm_logs"
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

    job_id = result.stdout.strip().split()[-1]
    return job_id


def main() -> None:
    jobs = build_jobs()
    print(f"Submitting {len(jobs)} dattri large-batch jobs")
    print(f"Results dir: {RUN_ROOT}")
    print()

    submitted: list[tuple[Job, str]] = []
    failed: list[Job] = []

    for job in jobs:
        print(
            f"  {job.label} (batch_size={job.batch_size})",
            end=" ",
        )
        job_id = submit_job(job)
        if job_id:
            submitted.append((job, job_id))
            print(f"[{job_id}]")
        else:
            failed.append(job)

    print(f"\n{len(submitted)}/{len(jobs)} submitted")
    if failed:
        print(f"{len(failed)} failed:")
        for j in failed:
            print(f"  {j.label}")
        sys.exit(1)

    print("\nMonitor with:" "\n  squeue -u $USER | grep lb")


if __name__ == "__main__":
    main()
