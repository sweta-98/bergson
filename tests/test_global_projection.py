"""Tests for the LESS-style global TRAK projection mode."""

from pathlib import Path

import pytest
import torch

from bergson import GradientProcessor, collect_gradients
from bergson.config import IndexConfig
from bergson.data import load_gradients


def test_global_projector_processor_field_default():
    p = GradientProcessor(projection_dim=16)
    assert p.projection_target == "per_module"


def test_global_projector_processor_field_set():
    p = GradientProcessor(projection_dim=8192, projection_target="global")
    assert p.projection_target == "global"


def test_global_shapes_collapse_to_single_key():
    """In global mode, shapes() returns one synthetic 'gradients' entry."""
    from bergson.collector.collector import HookCollectorBase

    # Construct a minimal stand-in for HookCollectorBase that exercises shapes()
    # without running a full forward/backward. We only need processor + a fake
    # target_info dict.
    class _Stub(HookCollectorBase):
        def __init__(self, processor, target_info):
            self.processor = processor
            self.target_info = target_info
            self.attention_cfgs = {}

        def setup(self):
            pass

        def teardown(self):
            pass

        def discover_targets(self, *_a, **_kw):
            return {}

        def backward_hook(self, module, g):
            pass

        def process_batch(self, indices, **kwargs):
            pass

    target_info = {
        "model.layers.0.q_proj": (None, torch.Size((4, 4)), False),
        "model.layers.0.k_proj": (None, torch.Size((4, 4)), False),
    }
    proc = GradientProcessor(projection_dim=64, projection_target="global")
    stub = _Stub(proc, target_info)
    shapes = stub.shapes()
    assert set(shapes.keys()) == {"gradients"}
    assert tuple(shapes["gradients"]) == (64,)


def test_global_shapes_requires_projection_dim():
    from bergson.collector.collector import HookCollectorBase

    class _Stub(HookCollectorBase):
        def __init__(self, processor, target_info):
            self.processor = processor
            self.target_info = target_info
            self.attention_cfgs = {}

        def setup(self):
            pass

        def teardown(self):
            pass

        def discover_targets(self, *_a, **_kw):
            return {}

        def backward_hook(self, module, g):
            pass

        def process_batch(self, indices, **kwargs):
            pass

    proc = GradientProcessor(projection_dim=None, projection_target="global")
    stub = _Stub(proc, target_info={})
    with pytest.raises(AssertionError):
        stub.shapes()


def test_per_module_shapes_unchanged():
    """Default per_module mode returns one shape per target module."""
    from bergson.collector.collector import HookCollectorBase

    class _Stub(HookCollectorBase):
        def __init__(self, processor, target_info):
            self.processor = processor
            self.target_info = target_info
            self.attention_cfgs = {}

        def setup(self):
            pass

        def teardown(self):
            pass

        def discover_targets(self, *_a, **_kw):
            return {}

        def backward_hook(self, module, g):
            pass

        def process_batch(self, indices, **kwargs):
            pass

    target_info = {
        "model.layers.0.q_proj": (None, torch.Size((4, 4)), False),
        "model.layers.0.k_proj": (None, torch.Size((4, 4)), False),
    }
    proc = GradientProcessor(projection_dim=8, projection_target="per_module")
    stub = _Stub(proc, target_info)
    shapes = stub.shapes()
    assert set(shapes.keys()) == set(target_info.keys())
    for s in shapes.values():
        assert tuple(s) == (8, 8)


# ---------------------------------------------------------------------------
# End-to-end build with global TRAK projection. Skipped when ``trak`` is not
# installed (BasicProjector lives in trak too).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_global_projector_e2e(tmp_path: Path, model, dataset):
    # CudaProjector requires proj_dim to be a multiple of 512.
    proj_dim = 512
    cfg = IndexConfig(
        run_path=str(tmp_path),
        skip_hessians=True,
        token_batch_size=64,
        projection_dim=proj_dim,
        projection_target="global",
    )
    processor = GradientProcessor(
        projection_dim=proj_dim,
        projection_target="global",
    )

    collect_gradients(model=model.cuda(), data=dataset, processor=processor, cfg=cfg)

    grads = load_gradients(cfg.partial_run_path)
    assert grads.dtype.names == ("gradients",)
    assert grads["gradients"].shape == (len(dataset), proj_dim)
