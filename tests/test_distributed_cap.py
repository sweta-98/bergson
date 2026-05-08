"""Tests for capping world_size to dataset size in build/score paths.

When a dataset has fewer docs than the configured world_size, the build
path used to pad the dataset and truncate the resulting index. The new
behavior reduces world_size to ``len(dataset)`` instead — no padding,
no truncation, no duplicate gradient computation across ranks.
"""

from bergson.config import DistributedConfig
from bergson.distributed import cap_world_size_to_dataset


def test_cap_to_dataset_size_when_smaller():
    """Tiny dataset → capped to single node, nproc_per_node = dataset size."""
    cfg = DistributedConfig(nnode=4, nproc_per_node=4)
    assert cfg.world_size == 16
    capped = cap_world_size_to_dataset(cfg, dataset_size=1)
    assert capped.world_size == 1
    assert capped.nnode == 1
    assert capped.nproc_per_node == 1


def test_cap_to_intermediate_dataset_size():
    """Dataset between 1 and original world_size → single node, fitting nproc."""
    cfg = DistributedConfig(nnode=4, nproc_per_node=4)
    capped = cap_world_size_to_dataset(cfg, dataset_size=3)
    assert capped.world_size == 3
    assert capped.nnode == 1
    assert capped.nproc_per_node == 3


def test_unchanged_when_dataset_at_least_world_size():
    """Dataset with >= world_size docs → cfg returned unchanged."""
    cfg = DistributedConfig(nnode=4, nproc_per_node=4)
    capped = cap_world_size_to_dataset(cfg, dataset_size=100)
    assert capped.world_size == 16
    assert capped.nnode == 4
    assert capped.nproc_per_node == 4


def test_caps_to_one_node_when_dataset_exceeds_nproc_per_node():
    """Dataset between nproc_per_node and world_size → 1 node, full nproc_per_node."""
    cfg = DistributedConfig(nnode=4, nproc_per_node=4)
    # 5 docs: too many for 1 GPU, too few for the full 16-GPU world.
    capped = cap_world_size_to_dataset(cfg, dataset_size=5)
    assert capped.world_size == 4
    assert capped.nnode == 1
    assert capped.nproc_per_node == 4


def test_unchanged_when_dataset_exactly_world_size():
    """Boundary: dataset == world_size leaves cfg unchanged (cap on strict-less)."""
    cfg = DistributedConfig(nnode=4, nproc_per_node=4)
    capped = cap_world_size_to_dataset(cfg, dataset_size=16)
    assert capped.world_size == 16
    assert capped.nnode == 4
    assert capped.nproc_per_node == 4


def test_does_not_mutate_input():
    """Capping returns a new cfg; the original is untouched."""
    cfg = DistributedConfig(nnode=4, nproc_per_node=4)
    cap_world_size_to_dataset(cfg, dataset_size=1)
    assert cfg.nnode == 4
    assert cfg.nproc_per_node == 4
    assert cfg.world_size == 16


def test_zero_dataset_size_caps_to_one_worker():
    """Dataset with zero docs caps to a single worker (downstream raises)."""
    cfg = DistributedConfig(nnode=4, nproc_per_node=4)
    capped = cap_world_size_to_dataset(cfg, dataset_size=0)
    assert capped.world_size == 1
    assert capped.nnode == 1
    assert capped.nproc_per_node == 1


def test_single_node_unchanged_when_already_small():
    """Single-node config with dataset >= existing world_size is unchanged."""
    cfg = DistributedConfig(nnode=1, nproc_per_node=2)
    capped = cap_world_size_to_dataset(cfg, dataset_size=10)
    assert capped.world_size == 2
    assert capped.nnode == 1
    assert capped.nproc_per_node == 2
