"""Submit 1-GPU CLI benchmarks via Slurm.

Each job runs on an exclusive node with a single GPU.
Results are saved to /projects/a6a/public/lucia/cli_bench_1gpu/.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

RUN_ROOT = Path(
    "/projects/a6a/public/lucia/cli_bench_1gpu"
)
WORK_DIR = Path("/home/a6a/lucia.a6a/bergson")
PYTHON = "/home/a6a/lucia.a6a/miniforge3/bin/python"
PARTITION = "workq"

MODELS = [
    "pythia-14m",
    "pythia-70m",
    "pythia-160m",
    "pythia-1b",
    "pythia-6.9b",
    "pythia-12b",
]
TOKEN_SCALES = ["10K", "100K", "1M", "10M", "100M"]

# Time limits per model size (larger models need more time)
TIME_LIMITS: dict[str, str] = {
    "pythia-14m": "02:00:00",
    "pythia-70m": "02:00:00",
    "pythia-160m": "02:00:00",
    "pythia-1b": "04:00:00",
    "pythia-6.9b": "08:00:00",
    "pythia-12b": "08:00:00",
}


@dataclass
class Job:
    model: str
    tokens: str

    @property
    def label(self) -> str:
        return f"cli_1gpu_{self.model}_{self.tokens}"

    def bench_cmd(self) -> str:
        return (
            f"{PYTHON} -m benchmarks.benchmark_bergson_cli"
            f" {self.model} {self.tokens} {RUN_ROOT}"
            f" --num_gpus 1"
        )


def build_jobs() -> list[Job]:
    jobs: list[Job] = []
    for model in MODELS:
        for tokens in TOKEN_SCALES:
            jobs.append(Job(model, tokens))
    return jobs


def submit_job(job: Job) -> str | None:
    """Submit a single job via sbatch, return job ID."""
    log_dir = WORK_DIR / "runs" / "cli_1gpu_slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    time_limit = TIME_LIMITS[job.model]

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
        #SBATCH --time={time_limit}

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
    print(f"Submitting {len(jobs)} 1-GPU CLI benchmark jobs")
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
        "\n  squeue -u $USER | grep cli_1gpu"
    )


if __name__ == "__main__":
    main()
