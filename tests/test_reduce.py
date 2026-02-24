import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from bergson import (
    CollectorComputer,
    DataConfig,
    IndexConfig,
    InMemoryCollector,
    PreprocessConfig,
    ReduceConfig,
)
from bergson.data import allocate_batches, load_gradient_dataset
from bergson.reduce import reduce
from bergson.utils.worker_utils import (
    create_processor,
    setup_data_pipeline,
    setup_model_and_peft,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_reduce_cli(tmp_path: Path):
    result = subprocess.run(
        [
            "python",
            "-m",
            "bergson",
            "reduce",
            "test_reduce_e2e",
            "--model",
            "EleutherAI/pythia-14m",
            "--dataset",
            "NeelNanda/pile-10k",
            "--split",
            "train[:100]",
            "--truncation",
            "--method",
            "mean",
            "--unit_normalize",
            "--skip_preconditioners",
            "--token_batch_size",
            "1024",
        ],
        cwd=tmp_path,
        capture_output=True,
        # Get strings instead of bytes
        text=True,
    )

    assert (
        "error" not in result.stderr.lower()
    ), f"Error found in stderr: {result.stderr}"

    # Load the gradient index
    index_cfg = IndexConfig(run_path=str(tmp_path / "test_reduce_e2e"))
    ds = load_gradient_dataset(Path(index_cfg.run_path), structured=False)
    assert len(ds) == 1

    grads = torch.tensor(ds["gradients"][:])
    assert not torch.isnan(grads).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_programmatic_reduce(tmp_path: Path):
    index_cfg = IndexConfig(
        run_path=str(tmp_path / "reduction"),
        data=DataConfig(truncation=True, split="train[:100]"),
        model="EleutherAI/pythia-14m",
        skip_preconditioners=True,
        token_batch_size=1024,
    )
    reduce_cfg = ReduceConfig()
    preprocess_cfg = PreprocessConfig()

    reduce(index_cfg, reduce_cfg, preprocess_cfg)

    # Assert 1-row reduction exists at the tmp_path
    ds = load_gradient_dataset(Path(index_cfg.run_path), structured=False)
    assert len(ds) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_in_memory_reduce(tmp_path: Path):
    index_cfg = IndexConfig(
        run_path=str(tmp_path / "reduction"),
        data=DataConfig(truncation=True, split="train[:100]"),
        model="EleutherAI/pythia-14m",
        skip_preconditioners=True,
        token_batch_size=1024,
    )
    reduce_cfg = ReduceConfig()
    preprocess_cfg = PreprocessConfig()

    ds = setup_data_pipeline(index_cfg)
    model, target_modules = setup_model_and_peft(index_cfg)
    processor = create_processor(model, ds, index_cfg, target_modules)
    batches = allocate_batches(ds["length"], index_cfg.token_batch_size)

    collector = InMemoryCollector(
        model=model.base_model,  # type: ignore
        cfg=index_cfg,
        processor=processor,
        target_modules=target_modules,
        data=ds,
        scorer=None,
        reduce_cfg=reduce_cfg,
        preprocess_cfg=preprocess_cfg,
        attention_cfgs={},
    )

    computer = CollectorComputer(
        model=model,  # type: ignore
        data=ds,
        collector=collector,
        batches=batches,
        cfg=index_cfg,
    )
    computer.run_with_collector_hooks(desc="New worker - Collecting gradients")

    shutil.move(index_cfg.partial_run_path, index_cfg.run_path)

    results = collector.gradients

    assert all(len(results[name]) == 1 for name in results.keys())
