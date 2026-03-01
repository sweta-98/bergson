"""Re-run factor benchmarks at 10M tokens for pythia-14m.

Runs all method/factor_type combinations that previously
succeeded. Prints each CLI command before executing.
"""

import subprocess
import sys

RUN_ROOT = "runs/factor-test-10M"
MODEL = "pythia-14m"
TOKENS = "10M"

# (method, factor_type) combinations that succeeded before
RUNS = [
    ("bergson", "normalizer"),
    ("bergson", "preconditioner"),
    ("bergson", "kfac"),
    ("bergson", "ekfac"),
    ("kronfluence", "diagonal"),
    ("kronfluence", "kfac"),
    ("kronfluence", "ekfac"),
    ("dattri", "arnoldi"),
]


def main():
    for method, factor_type in RUNS:
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.benchmark_factors",
            MODEL,
            TOKENS,
            RUN_ROOT,
            method,
            factor_type,
        ]
        print(f"\n{'='*60}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'='*60}\n")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"FAILED: {method}/{factor_type}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
