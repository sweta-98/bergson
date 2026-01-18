# Bergson Benchmarks

This directory contains benchmarking scripts for measuring Bergson's performance across different models and configurations.

## Benchmark Scripts

### Core Benchmarks

- **`benchmark_bergson.py`** - Programmatic benchmarks for Bergson
  - `run` - In-memory benchmark using `InMemoryCollector` (fast, single GPU)
  - `run-disk` - Disk-based benchmark using real `build()`, `reduce()`, `score_dataset()` (single GPU)

- **`benchmark_bergson_cli.py`** - CLI-based benchmark using subprocess
  - Tests the actual CLI commands (`bergson build`, `bergson reduce`, `bergson score`)
  - Supports multi-GPU via `--num-gpus`

### Comparison Benchmarks

- **`benchmark_dattri.py`** - Dattri influence function benchmark
- **`kronfluence_benchmark.py`** - Kronfluence influence function benchmark

### Utilities

- **`benchmark_utils.py`** - Shared utilities for all benchmarks
  - Model specifications
  - Token parsing
  - Path generation
  - Timestamp utilities
  - `load_benchmark_dataset()` - Load on-disk tokenized dataset with filtering

- **`save_to_disk.py`** - Utility for preprocessing and saving tokenized datasets to disk

### Analysis

- **`plot_cli_benchmark.py`** - Plot benchmark results
  - Automatically separates plots by num_gpus and hardware
  - Generates `cli_benchmark_1gpu.png`, `cli_benchmark_8gpu.png`, etc.
  - Each PNG only contains results from the same GPU/hardware configuration
- **`run_full_benchmark.py`** - Orchestrate full benchmark suite

## Usage Examples

### Loading the Benchmark Dataset

All benchmarks should use the pre-tokenized on-disk dataset for consistency:

```python
from benchmarks.benchmark_utils import load_benchmark_dataset

# Load and filter to sequences >= 1024 tokens
ds = load_benchmark_dataset()
```

Or test it directly:
```bash
python -m benchmarks.test_load_dataset
```

This will:
- Load the tokenized dataset from `data/EleutherAI/SmolLM2-135M-10B-tokenized`
- Filter out sequences shorter than 1024 tokens (for even batching)
- Print statistics about total tokens available

### In-Memory Benchmark (fastest)
```bash
python -m benchmarks.benchmark_bergson run pythia-14m 1M 100K
```

### Disk-Based Benchmark (tests real code paths)
```bash
python -m benchmarks.benchmark_bergson run-disk pythia-14m 1M 100K
```

### CLI Benchmark (multi-GPU support)

Single GPU (default):
```bash
python -m benchmarks.benchmark_bergson_cli pythia-70m 10M
```

Multi-GPU (8 GPUs):
```bash
python -m benchmarks.benchmark_bergson_cli pythia-70m 10M --num_gpus 8
```

### Running Full Benchmark Suites

**Small models (1 GPU):**
```bash
./benchmarks/run_small_models_cli_benchmark.sh
```

**Small models (8 GPUs):**
```bash
./benchmarks/run_small_models_8gpu.sh
```

**Large models (1 GPU):**
```bash
./benchmarks/run_large_models_cli_benchmark.sh
```

**Large models (8 GPUs):**
```bash
./benchmarks/run_large_models_8gpu.sh
```

### Generating Plots

The plotting script automatically separates results by GPU count and hardware:

```bash
python -m benchmarks.plot_cli_benchmark
```

This will:
- Load all benchmark results from `runs/bergson_cli_benchmark/`
- Group by (num_gpus, hardware) combination
- Generate separate plots for each configuration:
  - `figures/cli_benchmark_1gpu.png` - Single GPU results
  - `figures/cli_benchmark_8gpu.png` - 8 GPU results
  - `runs/benchmarks/cli_benchmark_1gpu.csv` - Single GPU data
  - `runs/benchmarks/cli_benchmark_8gpu.csv` - 8 GPU data

Each plot only contains results from the same GPU/hardware configuration, making comparisons fair and meaningful.

## Benchmark Comparison

| Benchmark | Method | Multi-GPU | Disk I/O | Use Case |
|-----------|--------|-----------|----------|----------|
| `run` | In-memory collector | No (FSDP only) | None | Quick memory scaling tests |
| `run-disk` | Real build/reduce/score | No | Yes | Test production code paths |
| CLI (1 GPU) | Subprocess CLI commands | No | Yes | Single GPU baseline |
| CLI (8 GPU) | Subprocess CLI commands | Yes | Yes | Full multi-GPU distributed |

## Benchmark Records

All benchmarks now include:
- **num_gpus**: Number of GPUs used for the run
- **hardware**: Hardware information (node name + GPU type/count)

This allows proper comparison between single-GPU and multi-GPU runs.

## Adding New Benchmarks

1. Add your benchmark script to this directory
2. Import from `benchmarks.benchmark_utils` for shared functionality
3. Follow the existing pattern for saving results (JSON records)
4. Update this README with your benchmark's purpose and usage
