"""Run all projection comparison benchmarks on fresh hardware.

Uses LPT scheduling to balance work across available GPUs.
Each GPU runs its assigned benchmarks sequentially.
"""

from __future__ import annotations

import heapq
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

DATASET = "data/EleutherAI/SmolLM2-135M-10B-tokenized"
RUN_ROOT = Path("runs/proj_fresh")
MAX_LENGTH = 1024
PROJ_DIM = 16

# Models and scales to benchmark
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

# Estimated runtimes (seconds) for scheduling.
# Keys are (method, projection, model, tokens).
ESTIMATES: dict[tuple[str, str, str, str], float] = {
    ("bergson", "without", "pythia-160m", "1M"): 940,
    ("bergson", "without", "pythia-70m", "1M"): 259,
    ("bergson", "without", "pythia-160m", "100K"): 116,
    ("dattri", "with", "pythia-160m", "1M"): 95,
    ("dattri", "without", "pythia-160m", "1M"): 93,
    ("bergson", "with", "pythia-1b", "1M"): 81,
    ("bergson", "without", "pythia-14m", "1M"): 32,
    ("bergson", "with", "pythia-160m", "1M"): 31,
    ("dattri", "with", "pythia-70m", "1M"): 29,
    ("bergson", "without", "pythia-70m", "100K"): 28,
    ("dattri", "without", "pythia-70m", "1M"): 27,
    ("bergson", "with", "pythia-70m", "1M"): 19,
    ("bergson", "with", "pythia-14m", "1M"): 17,
    ("dattri", "with", "pythia-14m", "1M"): 14,
    ("dattri", "without", "pythia-14m", "1M"): 13,
}


@dataclass
class Job:
    method: str  # "bergson" or "dattri"
    projection: str  # "with" or "without"
    model: str
    tokens: str
    estimated_seconds: float

    def cmd(self) -> list[str]:
        if self.method == "bergson":
            root = (
                RUN_ROOT
                / f"bergson_{'proj' if self.projection == 'with' else 'noproj'}"
            )
            proj_dim = PROJ_DIM if self.projection == "with" else 0
            return [
                sys.executable,
                "-m",
                "benchmarks.benchmark_bergson",
                self.model,
                self.tokens,
                str(root),
                "--dataset",
                DATASET,
                "--projection_dim",
                str(proj_dim),
                "--max_length",
                str(MAX_LENGTH),
            ]
        else:
            root = (
                RUN_ROOT / f"dattri_{'proj' if self.projection == 'with' else 'noproj'}"
            )
            cmd = [
                sys.executable,
                "-m",
                "benchmarks.benchmark_dattri",
                "--model",
                self.model,
                "--train_tokens",
                self.tokens,
                "--run_root",
                str(root),
                "--dataset",
                DATASET,
                "--max_length",
                str(MAX_LENGTH),
            ]
            if self.projection == "with":
                cmd.extend(["--projection_dim", str(PROJ_DIM)])
            return cmd

    @property
    def label(self) -> str:
        return f"{self.method} {self.projection} proj " f"{self.model} {self.tokens}"


def build_jobs() -> list[Job]:
    """Build the full list of benchmark jobs."""
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
                    key = (method, projection, model, tokens)
                    est = ESTIMATES.get(key, 2.0)
                    jobs.append(Job(method, projection, model, tokens, est))
    return jobs


def schedule_lpt(jobs: list[Job], num_gpus: int) -> list[list[Job]]:
    """LPT scheduling: assign longest jobs first to least-loaded GPU."""
    sorted_jobs = sorted(jobs, key=lambda j: j.estimated_seconds, reverse=True)
    # Min-heap of (total_time, gpu_index)
    heap: list[tuple[float, int]] = [(0.0, i) for i in range(num_gpus)]
    assignments: list[list[Job]] = [[] for _ in range(num_gpus)]
    for job in sorted_jobs:
        total, gpu_idx = heapq.heappop(heap)
        assignments[gpu_idx].append(job)
        heapq.heappush(heap, (total + job.estimated_seconds, gpu_idx))
    return assignments


def run_gpu_queue(
    gpu_idx: int,
    jobs: list[Job],
    results: dict[int, list[tuple[Job, int]]],
) -> None:
    """Run a queue of jobs on a single GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    gpu_results: list[tuple[Job, int]] = []
    for job in jobs:
        cmd = job.cmd()
        print(f"[GPU {gpu_idx}] {job.label}")
        print(f"  $ {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(
                f"[GPU {gpu_idx}] FAILED: {job.label}"
                f"\n  stderr: {result.stderr[-200:]}"
            )
        else:
            print(f"[GPU {gpu_idx}] done: {job.label}")
        gpu_results.append((job, result.returncode))
    results[gpu_idx] = gpu_results


def main() -> None:
    num_gpus = 4
    jobs = build_jobs()
    assignments = schedule_lpt(jobs, num_gpus)

    print(f"Total jobs: {len(jobs)}")
    for i, gpu_jobs in enumerate(assignments):
        est = sum(j.estimated_seconds for j in gpu_jobs)
        print(f"  GPU {i}: {len(gpu_jobs)} jobs, " f"~{est:.0f}s estimated")
    print()

    results: dict[int, list[tuple[Job, int]]] = {}
    threads: list[threading.Thread] = []
    for gpu_idx, gpu_jobs in enumerate(assignments):
        t = threading.Thread(
            target=run_gpu_queue,
            args=(gpu_idx, gpu_jobs, results),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    total = 0
    failed = 0
    for gpu_idx in sorted(results):
        for job, rc in results[gpu_idx]:
            total += 1
            status = "OK" if rc == 0 else "FAIL"
            if rc != 0:
                failed += 1
            print(f"  [{status}] GPU {gpu_idx}: {job.label}")
    print(f"\n{total - failed}/{total} succeeded")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
