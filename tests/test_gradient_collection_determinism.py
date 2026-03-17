"""Test that OLMo gradient collection with random projection is deterministic.

Uses InMemoryCollector to collect gradients in memory (no disk I/O),
then compares cosine similarity and L2 distance between two runs.
"""

import os

import numpy as np
import pytest
import torch

from bergson.collector.collector import CollectorComputer
from bergson.collector.in_memory_collector import InMemoryCollector
from bergson.config import DataConfig, IndexConfig, PreprocessConfig
from bergson.data import load_gradients
from bergson.gradients import GradientProcessor

MODEL = "allenai/OLMo-2-1124-7B-Instruct"
NUM_EXAMPLES = 20


def _make_data():
    """Load a small slice of pile-10k, tokenized."""
    from datasets import load_dataset

    ds = load_dataset("NeelNanda/pile-10k", split=f"train[:{NUM_EXAMPLES}]")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ds = ds.map(
        lambda x: tokenizer(
            x["text"], truncation=True, max_length=128, return_attention_mask=False
        ),
        remove_columns=ds.column_names,
    )
    return ds


def _collect_grads(model, data, precision="bf16", autocast=False, projection_dim=8, tmp_dir=None):
    """Run InMemoryCollector and return flat gradient tensor."""
    import tempfile

    run_path = tmp_dir or tempfile.mkdtemp()
    # IndexConfig expects run_path.part to exist
    os.makedirs(run_path + ".part", exist_ok=True)
    cfg = IndexConfig(
        run_path=run_path,
        precision=precision,
        projection_dim=projection_dim,
        skip_preconditioners=True,
        skip_index=True,
        overwrite=True,
    )
    if autocast:
        cfg.autocast = True

    preprocess = PreprocessConfig(
        unit_normalize=True,
        aggregation="none",
        normalize_aggregated_grad=False,
    )

    processor = GradientProcessor(projection_dim=projection_dim)
    collector = InMemoryCollector(
        model=model,
        data=data,
        cfg=cfg,
        preprocess_cfg=preprocess,
        processor=processor,
    )

    computer = CollectorComputer(
        model=model,
        data=data,
        collector=collector,
        cfg=cfg,
    )
    computer.run_with_collector_hooks(desc="test collection")

    # Concatenate per-module gradients into a flat array
    all_grads = []
    for name in sorted(collector.gradients.keys()):
        all_grads.append(collector.gradients[name].float().cpu())
    return torch.cat(all_grads, dim=1).numpy()


def _per_example_cosine(a, b):
    norms_a = np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-10)
    norms_b = np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-10)
    return np.sum((a / norms_a) * (b / norms_b), axis=1)


def _per_example_l2(a, b):
    return np.linalg.norm(a - b, axis=1)


def _per_example_relative_l2(a, b):
    l2 = np.linalg.norm(a - b, axis=1)
    norms = (np.linalg.norm(a, axis=1) + np.linalg.norm(b, axis=1)) / 2
    return l2 / norms.clip(min=1e-10)


def _report(grads_a, grads_b, label):
    cosine = _per_example_cosine(grads_a, grads_b)
    l2 = _per_example_l2(grads_a, grads_b)
    rel_l2 = _per_example_relative_l2(grads_a, grads_b)
    exact = np.mean(grads_a == grads_b)
    print(f"\n{label} ({grads_a.shape[0]} examples, dim={grads_a.shape[1]}):")
    print(f"  Exact match:  {exact:.1%}")
    print(f"  Cosine sim:   min={cosine.min():.8f}  mean={cosine.mean():.8f}")
    print(f"  L2 dist:      max={l2.max():.6e}  mean={l2.mean():.6e}")
    print(f"  Relative L2:  max={rel_l2.max():.6e}  mean={rel_l2.mean():.6e}")
    return cosine, l2, rel_l2


@pytest.fixture(scope="module")
def olmo_model():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )


@pytest.fixture(scope="module")
def olmo_model_fp32():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, device_map="cuda"
    )


@pytest.fixture(scope="module")
def data():
    return _make_data()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bf16_deterministic(olmo_model, data):
    """Two bf16 collections produce approximately equal gradients."""
    a = _collect_grads(olmo_model, data, "bf16")
    b = _collect_grads(olmo_model, data, "bf16")

    assert a.shape == b.shape
    cosine, _, rel_l2 = _report(a, b, "BF16")

    assert cosine.min() > 0.99
    assert rel_l2.max() < 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fp32_deterministic(olmo_model_fp32, data):
    """Two fp32 collections produce nearly identical gradients."""
    a = _collect_grads(olmo_model_fp32, data, "fp32")
    b = _collect_grads(olmo_model_fp32, data, "fp32")

    assert a.shape == b.shape
    cosine, _, rel_l2 = _report(a, b, "FP32")

    assert cosine.min() > 0.9999
    assert rel_l2.max() < 0.001


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bf16_autocast_deterministic(olmo_model, data):
    """Two bf16 autocast collections produce approximately equal gradients."""
    a = _collect_grads(olmo_model, data, "bf16", autocast=True)
    b = _collect_grads(olmo_model, data, "bf16", autocast=True)

    assert a.shape == b.shape
    cosine, _, rel_l2 = _report(a, b, "BF16 autocast")

    assert cosine.min() > 0.99
    assert rel_l2.max() < 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_determinism_stats(olmo_model, olmo_model_fp32, data):
    """Collect N=10 pairs for each precision and report mean/std."""
    n = 10
    for label, model, precision, ac in [
        ("BF16", olmo_model, "bf16", False),
        ("BF16 autocast", olmo_model, "bf16", True),
        ("FP32", olmo_model_fp32, "fp32", False),
    ]:
        cosines, l2s, rel_l2s, exacts = [], [], [], []
        for _ in range(n):
            a = _collect_grads(model, data, precision, autocast=ac)
            b = _collect_grads(model, data, precision, autocast=ac)
            cosines.append(_per_example_cosine(a, b).mean())
            l2s.append(_per_example_l2(a, b).mean())
            rel_l2s.append(_per_example_relative_l2(a, b).mean())
            exacts.append(np.mean(a == b))

        print(f"\n{label} (N={n} pairs):")
        for name, vals in [
            ("cosine", cosines),
            ("l2", l2s),
            ("rel_l2", rel_l2s),
            ("exact_match", exacts),
        ]:
            v = np.array(vals)
            print(f"  {name:15s}  mean={v.mean():.8f}  std={v.std():.8f}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bf16_model_reload_deterministic(data):
    """Two collections with independently loaded models.

    Isolates whether model reload (not disk I/O) causes the nondeterminism
    seen in build-vs-build.
    """
    from transformers import AutoModelForCausalLM

    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    a = _collect_grads(model_a, data, "bf16")
    del model_a
    torch.cuda.empty_cache()

    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    b = _collect_grads(model_b, data, "bf16")
    del model_b
    torch.cuda.empty_cache()

    assert a.shape == b.shape
    cosine, _, rel_l2 = _report(a, b, "BF16 model reload")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nonzero(olmo_model, data):
    """Collected gradients are not all zeros."""
    grads = _collect_grads(olmo_model, data, "bf16")
    assert grads.shape[0] > 0
    assert not np.all(grads == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batching_deterministic(olmo_model, data):
    """Compare default batching (size 1) vs token-length batching.

    Isolates whether allocate_batches changes results.
    """
    from bergson.build import allocate_batches

    a = _collect_grads(olmo_model, data, "bf16")

    # Collect with token-length batching
    cfg = IndexConfig(
        run_path="/tmp/unused_batch",
        precision="bf16",
        projection_dim=8,
        skip_preconditioners=True,
        skip_index=True,
        overwrite=True,
        token_batch_size=512,
    )
    os.makedirs(cfg.run_path + ".part", exist_ok=True)
    preprocess = PreprocessConfig(
        unit_normalize=True,
        aggregation="none",
        normalize_aggregated_grad=False,
    )
    processor = GradientProcessor(projection_dim=8)

    lengths = [len(ids) for ids in data["input_ids"]]
    batches = allocate_batches(lengths, cfg.token_batch_size)

    collector = InMemoryCollector(
        model=olmo_model,
        data=data,
        cfg=cfg,
        preprocess_cfg=preprocess,
        processor=processor,
    )
    computer = CollectorComputer(
        model=olmo_model,
        data=data,
        collector=collector,
        batches=batches,
        cfg=cfg,
    )
    computer.run_with_collector_hooks(desc="batched collection")

    all_grads = []
    for name in sorted(collector.gradients.keys()):
        all_grads.append(collector.gradients[name].float().cpu())
    b = torch.cat(all_grads, dim=1).numpy()

    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    cosine, _, _ = _report(a, b, "sdpa: bs=1 vs token-length batching")
    # sdpa padding affects gradients — this documents the magnitude
    assert cosine.min() > 0.99


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_attn_batching_deterministic(data):
    """Test if flash_attention_2 gives batch-invariant gradients.

    Flash attention handles variable-length sequences natively without
    padding affecting the computation.
    """
    from transformers import AutoModelForCausalLM
    from bergson.build import allocate_batches

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="flash_attention_2",
    )

    # Batch size 1
    a = _collect_grads(model, data, "bf16")

    # Token-length batching
    cfg = IndexConfig(
        run_path="/tmp/unused_flash",
        precision="bf16",
        projection_dim=8,
        skip_preconditioners=True,
        skip_index=True,
        overwrite=True,
        token_batch_size=512,
    )
    os.makedirs(cfg.run_path + ".part", exist_ok=True)
    preprocess = PreprocessConfig(
        unit_normalize=True,
        aggregation="none",
        normalize_aggregated_grad=False,
    )
    processor = GradientProcessor(projection_dim=8)

    lengths = [len(ids) for ids in data["input_ids"]]
    batches = allocate_batches(lengths, cfg.token_batch_size)

    collector = InMemoryCollector(
        model=model,
        data=data,
        cfg=cfg,
        preprocess_cfg=preprocess,
        processor=processor,
    )
    computer = CollectorComputer(
        model=model,
        data=data,
        collector=collector,
        batches=batches,
        cfg=cfg,
    )
    computer.run_with_collector_hooks(desc="flash_attn batched")

    all_grads = []
    for name in sorted(collector.gradients.keys()):
        all_grads.append(collector.gradients[name].float().cpu())
    b = torch.cat(all_grads, dim=1).numpy()

    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    cosine, _, _ = _report(a, b, "flash_attention_2: bs=1 vs token-length batching")

    del model
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_disk_roundtrip_bf16(olmo_model, data, tmp_path):
    """Compare in-memory gradients vs disk-roundtripped (write+read) gradients.

    Collects gradients in-memory, writes them to a numpy memmap in bfloat16,
    reads them back, and checks for lossless roundtrip.
    """
    from ml_dtypes import bfloat16 as np_bf16

    mem_grads = _collect_grads(olmo_model, data, "bf16")

    # Write to memmap in bfloat16 (same as bergson's disk path)
    disk_path = tmp_path / "grads.bin"
    bf16_grads = mem_grads.astype(np_bf16)
    mmap = np.memmap(disk_path, dtype=np_bf16, mode="w+", shape=bf16_grads.shape)
    mmap[:] = bf16_grads
    mmap.flush()

    # Read back
    loaded = np.memmap(disk_path, dtype=np_bf16, mode="r", shape=bf16_grads.shape)
    loaded_f32 = np.array(loaded).astype(np.float32)

    cosine, _, _ = _report(mem_grads, loaded_f32, "BF16 memory vs disk roundtrip")

    exact = np.mean(mem_grads == loaded_f32)
    print(f"  Exact match after roundtrip: {exact:.1%}")

    if exact < 1.0:
        diff_mask = mem_grads != loaded_f32
        abs_diffs = np.abs(mem_grads[diff_mask] - loaded_f32[diff_mask])
        print(f"  {diff_mask.sum()} elements differ, max diff={abs_diffs.max():.6e}")

    assert cosine.min() > 0.9999, f"Cosine sim too low: {cosine.min():.8f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_build_vs_build_bf16(tmp_path):
    """Two bergson builds from scratch produce consistent gradients.

    This tests the full pipeline including model loading, gradient
    collection, and disk write/read.
    """
    from bergson.build import build as bergson_build

    def _run_build(run_path):
        cfg = IndexConfig(
            run_path=run_path,
            model=MODEL,
            data=DataConfig(
                dataset="NeelNanda/pile-10k",
                split=f"train[:{NUM_EXAMPLES}]",
                truncation=True,
            ),
            precision="bf16",
            projection_dim=8,
            normalizer="none",
            token_batch_size=512,
            skip_preconditioners=True,
            overwrite=True,
        )
        preprocess = PreprocessConfig(
            unit_normalize=True,
            aggregation="none",
            normalize_aggregated_grad=False,
        )
        bergson_build(cfg, preprocess)
        return np.array(load_gradients(run_path, structured=False)).astype(np.float32)

    a = _run_build(str(tmp_path / "run_a"))
    b = _run_build(str(tmp_path / "run_b"))

    assert a.shape == b.shape
    cosine, _, rel_l2 = _report(a, b, "BF16 build vs build")

    # May not be bitwise identical due to model reload + CUDA state,
    # but should be very close
    assert cosine.min() > 0.99, f"Cosine sim too low: {cosine.min():.6f}"
