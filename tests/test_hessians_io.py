from pathlib import Path

import pytest

from bergson.config import HessianConfig
from bergson.hessians.io import (
    HESSIAN_CONFIG_FILENAME,
    hessian_method,
    load_hessian_config,
)


def _save(tmp_path: Path, cfg: HessianConfig) -> Path:
    cfg.save_yaml(tmp_path / HESSIAN_CONFIG_FILENAME)
    return tmp_path


@pytest.mark.parametrize(
    "method,ev_correction",
    [
        ("autocorrelation", False),
        ("kfac", True),
        ("tkfac", False),
        ("shampoo", False),
    ],
)
def test_load_hessian_config_round_trip(
    tmp_path: Path, method: str, ev_correction: bool
):
    saved = _save(tmp_path, HessianConfig(method=method, ev_correction=ev_correction))
    loaded = load_hessian_config(saved)
    assert loaded.method == method
    assert loaded.ev_correction is ev_correction
    assert hessian_method(saved) == method


def test_load_hessian_config_accepts_str_path(tmp_path: Path):
    saved = _save(tmp_path, HessianConfig(method="kfac"))
    assert hessian_method(str(saved)) == "kfac"


def test_load_hessian_config_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as exc_info:
        load_hessian_config(tmp_path)
    assert HESSIAN_CONFIG_FILENAME in str(exc_info.value)
