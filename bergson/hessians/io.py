"""Discovery helpers for saved Hessian approximations.

Every ``bergson hessian`` invocation writes its output under
``<run_path>/<method>/`` along with a ``hessian_config.yaml`` that records the
``HessianConfig`` used to produce it. The helpers here let callers identify the
method behind a saved directory without having to know which method-specific
loader to call.

The on-disk format itself differs by method (sharded safetensors for the K-FAC
family vs. ``GradientProcessor.save`` artifacts for autocorrelation); these
helpers do not unify the formats — they just expose the metadata needed to
dispatch to the right loader. See ``bergson.hessians.apply_hessian`` for K-FAC
inverse-Hessian application and ``bergson.process_grads.get_trackstar_hessian``
for autocorrelation Hessian loading.
"""

from __future__ import annotations

from pathlib import Path

from bergson.config import HessianConfig

HESSIAN_CONFIG_FILENAME = "hessian_config.yaml"


def load_hessian_config(path: str | Path) -> HessianConfig:
    """Read ``hessian_config.yaml`` from a saved Hessian directory."""
    cfg_path = Path(path) / HESSIAN_CONFIG_FILENAME
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No {HESSIAN_CONFIG_FILENAME} found at {cfg_path}. "
            "This directory was not produced by `bergson hessian` or "
            "`bergson build` (with skip_hessians=False), or is from a "
            "version of bergson that pre-dates Hessian config serialization."
        )
    return HessianConfig.load_yaml(str(cfg_path))


def hessian_method(path: str | Path) -> str:
    """Return the method ('kfac', 'tkfac', 'shampoo', 'autocorrelation') used
    to compute the Hessian saved at ``path``."""
    return load_hessian_config(path).method
