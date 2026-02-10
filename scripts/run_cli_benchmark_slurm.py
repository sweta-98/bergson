"""Submit CLI benchmarks via Slurm.

Each job runs on an exclusive node.
Results are saved to /projects/a6a/public/lucia/cli_bench_{num_gpus}gpu/.

Usage:
    python scripts/run_cli_benchmark_slurm.py --num_gpus 1
    python scripts/run_cli_benchmark_slurm.py --num_gpus 4
    python scripts/run_cli_benchmark_slurm.py --num_gpus 1 --tokens 10M 100M
    python scripts/run_cli_benchmark_slurm.py --num_gpus 1 --skip_existing false
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

RUN_ROOT_BASE = Path("/projects/a6a/public/lucia")
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
ALL_TOKEN_SCALES = ["10K", "100K", "1M", "10M", "100M"]

# Time limits per model size
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
    num_gpus: int
    run_root: Path
    skip_existing: bool

    @property
    def label(self) -> str:
        return (
            f"cli_{self.num_gpus}gpu"
            f"_{self.model}_{self.tokens}"
        )

    def bench_cmd(self) -> str:
        cmd = (
            f"{PYTHON} -m benchmarks.benchmark_bergson_cli"
            f" {self.model} {self.tokens} {self.run_root}"
            f" --num_gpus {self.num_gpus}"
        )
        if not self.skip_existing:
            cmd += " --skip_existing false"
        return cmd


def submit_job(job: Job) -> str | None:
    """Submit a single job via sbatch, return job ID."""
    log_dir = (
        WORK_DIR / "runs" / f"cli_{job.num_gpus}gpu_slurm_logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    time_limit = TIME_LIMITS[job.model]

    # Set CUDA_VISIBLE_DEVICES for < 4 GPUs
    if job.num_gpus < 4:
        devices = ",".join(str(i) for i in range(job.num_gpus))
        cuda_line = f"export CUDA_VISIBLE_DEVICES={devices}"
    else:
        cuda_line = "# Using all available GPUs"

    script = dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name={job.label}
        #SBATCH --output={log_dir / (job.label + ".out")}
        #SBATCH --error={log_dir / (job.label + ".err")}
        #SBATCH --partition={PARTITION}
        #SBATCH --nodes=1
        #SBATCH --exclusive
        #SBATCH --gres=gpu:{job.num_gpus}
        #SBATCH --time={time_limit}

        cd {WORK_DIR}
        {cuda_line}
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

    job_id = result.stdout.strip().split()[-1]
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit CLI benchmark Slurm jobs",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        required=True,
        help="Number of GPUs per job",
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        default=ALL_TOKEN_SCALES,
        help="Token scales to run (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Models to run (default: all)",
    )
    parser.add_argument(
        "--skip_existing",
        type=str,
        default="true",
        help="Skip existing successful runs (default: true)",
    )
    args = parser.parse_args()

    skip = args.skip_existing.lower() == "true"
    run_root = RUN_ROOT_BASE / f"cli_bench_{args.num_gpus}gpu"

    jobs: list[Job] = []
    for model in args.models:
        for tokens in args.tokens:
            jobs.append(
                Job(model, tokens, args.num_gpus, run_root, skip)
            )

    print(
        f"Submitting {len(jobs)} {args.num_gpus}-GPU "
        f"CLI benchmark jobs"
    )
    print(f"Results dir: {run_root}")
    print(f"Token scales: {args.tokens}")
    print(f"Skip existing: {skip}")
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

    tag = f"cli_{args.num_gpus}gpu"
    print(f"\nMonitor with:\n  squeue -u $USER | grep {tag}")


if __name__ == "__main__":
    main()
