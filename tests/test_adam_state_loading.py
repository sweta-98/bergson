import tempfile

import pytest
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from bergson.gradients import AdafactorNormalizer, AdamNormalizer
from bergson.utils.load_from_optimizer import (
    OptimizerStateFormat,
    get_optimizer_state_format,
    get_unfactored_second_moment,
    load_from_optimizer,
)
from bergson.utils.worker_utils import extract_peft_target_modules


def _create_model():
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    return AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)


def _create_fake_optimizer_state(model, lr=1e-3):
    """Create a fake optimizer state dict matching the model's parameters."""
    state = {}
    param_groups = [{"lr": lr, "params": []}]

    for idx, (name, param) in enumerate(model.named_parameters()):
        param_groups[0]["params"].append(idx)
        state[idx] = {
            "step": torch.tensor(100),
            "exp_avg": torch.zeros_like(param),
            "exp_avg_sq": torch.rand_like(param) * 0.01,
        }

    return {"state": state, "param_groups": param_groups}


def _train_checkpoint(optim_name: str) -> tuple:
    """Train a tiny model for a few steps and return (checkpoint_path, model)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    dummy_data = Dataset.from_dict({"text": ["hello world"] * 20, "label": [0, 1] * 10})
    dummy_data = dummy_data.map(
        lambda x: tokenizer(
            x["text"], padding="max_length", truncation=True, max_length=32
        ),
        batched=True,
    )

    tmpdir = tempfile.mkdtemp()
    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=3,
        save_steps=3,
        per_device_train_batch_size=4,
        optim=optim_name,
        learning_rate=1e-4,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dummy_data)
    trainer.train()

    import os

    ckpt = [d for d in os.listdir(tmpdir) if d.startswith("checkpoint")][0]
    return os.path.join(tmpdir, ckpt), model


# ---------------------------------------------------------------------------
# Unit tests with fake state
# ---------------------------------------------------------------------------


def test_load_from_optimizer_file(tmp_path):
    """Load normalizers from a bare optimizer.pt file."""
    model = _create_model()
    opt_state = _create_fake_optimizer_state(model)

    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    normalizers = load_from_optimizer(model, str(opt_path))

    assert len(normalizers) > 0
    for name, norm in normalizers.items():
        assert isinstance(norm, AdamNormalizer)
        assert norm.weight_avg_sq.ndim == 2


def test_load_from_checkpoint_dir(tmp_path):
    """Load normalizers from a checkpoint directory containing optimizer.pt."""
    model = _create_model()
    opt_state = _create_fake_optimizer_state(model)

    checkpoint_dir = tmp_path / "checkpoint-100"
    checkpoint_dir.mkdir()
    torch.save(opt_state, checkpoint_dir / "optimizer.pt")

    normalizers = load_from_optimizer(model, str(checkpoint_dir))
    assert len(normalizers) > 0


def test_target_modules_filter(tmp_path):
    """Only layers in target_modules are loaded."""
    model = _create_model()
    opt_state = _create_fake_optimizer_state(model)

    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    all_linear = {
        name for name, module in model.named_modules() if isinstance(module, nn.Linear)
    }
    subset = set(list(all_linear)[:2])

    normalizers = load_from_optimizer(model, str(opt_path), target_modules=subset)
    assert set(normalizers.keys()) == subset


def test_missing_optimizer_file(tmp_path):
    """Error when directory has no optimizer.pt."""
    model = _create_model()

    with pytest.raises(FileNotFoundError):
        load_from_optimizer(model, str(tmp_path))


# ---------------------------------------------------------------------------
# load_optimizer (local + Hub) tests
# ---------------------------------------------------------------------------


def test_load_optimizer_local_file(tmp_path):
    from bergson.utils.load_from_optimizer import load_optimizer

    model = _create_model()
    state = _create_fake_optimizer_state(model)
    opt_path = tmp_path / "optimizer.pt"
    torch.save(state, opt_path)

    loaded = load_optimizer(str(opt_path))
    assert "state" in loaded and "param_groups" in loaded


def test_load_optimizer_local_dir(tmp_path):
    from bergson.utils.load_from_optimizer import load_optimizer

    model = _create_model()
    state = _create_fake_optimizer_state(model)
    torch.save(state, tmp_path / "optimizer.pt")

    loaded = load_optimizer(str(tmp_path))
    assert "state" in loaded


def test_load_optimizer_hub_dispatch(tmp_path, monkeypatch):
    """hf:// URIs should dispatch to hf_hub_download with parsed args."""
    from bergson.utils import load_from_optimizer as mod

    model = _create_model()
    state = _create_fake_optimizer_state(model)
    cached = tmp_path / "optimizer.pt"
    torch.save(state, cached)

    calls = []

    def fake_download(repo_id, filename, revision=None, repo_type=None, **_):
        calls.append((repo_id, filename, revision, repo_type))
        return str(cached)

    monkeypatch.setattr(mod, "hf_hub_download", fake_download)

    cases = [
        ("hf://org/repo", ("org/repo", "optimizer.pt", None, "model")),
        ("hf://org/repo@rev", ("org/repo", "optimizer.pt", "rev", "model")),
        (
            "hf://org/repo/checkpoint-1",
            ("org/repo", "checkpoint-1/optimizer.pt", None, "model"),
        ),
        ("hf://org/repo/custom.pt", ("org/repo", "custom.pt", None, "model")),
        (
            "hf://org/repo@v2/sub/dir/optimizer.pth",
            ("org/repo", "sub/dir/optimizer.pth", "v2", "model"),
        ),
        (
            "hf://datasets/org/repo/optimizer.pt",
            ("org/repo", "optimizer.pt", None, "dataset"),
        ),
    ]
    for spec, expected in cases:
        calls.clear()
        mod.load_optimizer(spec)
        assert calls == [expected], f"{spec} -> {calls}"


def test_load_optimizer_invalid_spec():
    from bergson.utils.load_from_optimizer import load_optimizer

    with pytest.raises(FileNotFoundError):
        load_optimizer("not/a/local/path")


# ---------------------------------------------------------------------------
# get_optimizer_state_format / get_unfactored_second_moment unit tests
# ---------------------------------------------------------------------------


def test_get_optimizer_state_format_adam():
    state = {"exp_avg_sq": torch.zeros(2, 3), "step": torch.tensor(1)}
    assert get_optimizer_state_format(state) == OptimizerStateFormat.UNFACTORED


def test_get_optimizer_state_format_adafactor():
    state = {"exp_avg_sq_row": torch.zeros(2), "exp_avg_sq_col": torch.zeros(3)}
    assert get_optimizer_state_format(state) == OptimizerStateFormat.FACTORED


def test_get_optimizer_state_format_bnb_8bit_adam():
    state = {"__bnb_optimizer_quant_state__": {"state2": torch.zeros(8)}}
    assert get_optimizer_state_format(state) == OptimizerStateFormat.UNFACTORED


def test_get_optimizer_state_format_empty_or_unknown_returns_none():
    # Empty (param registered but never stepped) and unknown formats both
    # return None so the main loop can skip without crashing.
    assert get_optimizer_state_format({}) is None
    assert get_optimizer_state_format({"step": torch.tensor(1)}) is None
    assert get_optimizer_state_format({"square_avg": torch.zeros(2)}) is None


def test_get_optimizer_state_format_non_dict_returns_none():
    # The isinstance guard means None / non-dicts return None instead of
    # raising "argument of type 'NoneType' is not iterable" on the `in` check.
    assert get_optimizer_state_format(None) is None
    assert get_optimizer_state_format("not a dict") is None


def test_get_unfactored_second_moment_adam_and_bnb():
    sq = torch.rand(2, 3)
    assert torch.equal(get_unfactored_second_moment({"exp_avg_sq": sq}), sq)

    bnb_sq = torch.rand(8)
    state = {"__bnb_optimizer_quant_state__": {"state2": bnb_sq}}
    assert torch.equal(get_unfactored_second_moment(state), bnb_sq)


# ---------------------------------------------------------------------------
# Bad / empty target_modules paths
# ---------------------------------------------------------------------------


def test_target_modules_no_overlap_raises(tmp_path):
    """If target_modules names don't match any param, no normalizers loaded."""
    model = _create_model()
    opt_state = _create_fake_optimizer_state(model)
    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    with pytest.raises(AssertionError, match="No optimizer second moments"):
        load_from_optimizer(
            model, str(opt_path), target_modules={"definitely.not.a.real.module"}
        )


def test_target_modules_empty_set_raises(tmp_path):
    """An empty target_modules set rejects everything → assertion."""
    model = _create_model()
    opt_state = _create_fake_optimizer_state(model)
    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    with pytest.raises(AssertionError, match="No optimizer second moments"):
        load_from_optimizer(model, str(opt_path), target_modules=set())


# ---------------------------------------------------------------------------
# Unrecognized / mixed-format states
# ---------------------------------------------------------------------------


def test_unrecognized_state_skipped_others_loaded(tmp_path):
    """A state entry with no recognised keys is skipped; the rest still load."""
    model = _create_model()
    opt_state = _create_fake_optimizer_state(model)

    # Replace the first param's state with an unknown-format dict.
    first_idx = next(iter(opt_state["state"]))
    opt_state["state"][first_idx] = {"step": torch.tensor(1)}

    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    normalizers = load_from_optimizer(model, str(opt_path))

    # Some normalizers loaded — the unrecognised one was simply skipped.
    assert len(normalizers) > 0


# ---------------------------------------------------------------------------
# include_bias path
# ---------------------------------------------------------------------------


def test_include_bias_loads_bias_normalizer(tmp_path):
    """Bias second moments are attached when include_bias=True."""
    # Build a minimal model with a bias and craft optimizer state for both
    # weight and bias of the same Linear.
    model = nn.Sequential(nn.Linear(3, 4, bias=True))
    state: dict = {}
    param_groups = [{"lr": 1e-3, "params": []}]
    for idx, (_name, param) in enumerate(model.named_parameters()):
        param_groups[0]["params"].append(idx)
        state[idx] = {
            "step": torch.tensor(1),
            "exp_avg": torch.zeros_like(param),
            "exp_avg_sq": torch.rand_like(param) * 0.01,
        }
    opt_state = {"state": state, "param_groups": param_groups}
    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    normalizers = load_from_optimizer(model, str(opt_path), include_bias=True)  # type: ignore[arg-type]
    assert len(normalizers) == 1
    norm = next(iter(normalizers.values()))
    assert isinstance(norm, AdamNormalizer)
    assert norm.bias_avg_sq is not None
    assert norm.bias_avg_sq.shape == (4,)


def test_include_bias_false_leaves_bias_unset(tmp_path):
    model = nn.Sequential(nn.Linear(3, 4, bias=True))
    state: dict = {}
    param_groups = [{"lr": 1e-3, "params": []}]
    for idx, (_name, param) in enumerate(model.named_parameters()):
        param_groups[0]["params"].append(idx)
        state[idx] = {
            "step": torch.tensor(1),
            "exp_avg": torch.zeros_like(param),
            "exp_avg_sq": torch.rand_like(param) * 0.01,
        }
    opt_state = {"state": state, "param_groups": param_groups}
    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    normalizers = load_from_optimizer(model, str(opt_path), include_bias=False)  # type: ignore[arg-type]
    assert len(normalizers) == 1
    norm = next(iter(normalizers.values()))
    assert isinstance(norm, AdamNormalizer)
    assert norm.bias_avg_sq is None


# ---------------------------------------------------------------------------
# PEFT path: adapter-suffixed target_modules must match
# ---------------------------------------------------------------------------


def _create_peft_model() -> PeftModel:
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    base = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    model = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["qkv_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    assert isinstance(model, PeftModel)
    return model


def _fake_optimizer_state_for_peft(peft_model):
    """Build optimizer state keyed by index, matching get_peft_model_state_dict
    order (which is what load_from_optimizer uses for PEFT models)."""
    state: dict = {}
    param_groups = [{"lr": 1e-3, "params": []}]
    psd = get_peft_model_state_dict(peft_model)
    for idx, (_name, param) in enumerate(psd.items()):
        param_groups[0]["params"].append(idx)
        state[idx] = {
            "step": torch.tensor(1),
            "exp_avg": torch.zeros_like(param),
            "exp_avg_sq": torch.rand_like(param) * 0.01,
        }
    return {"state": state, "param_groups": param_groups}


def test_load_from_peft_model_with_adapter_suffix(tmp_path):
    """Regression: PEFT module names from extract_peft_target_modules include
    the adapter suffix (``.default``); load_from_optimizer must produce
    matching keys, otherwise the target_modules filter rejects everything."""
    model = _create_peft_model()
    opt_state = _fake_optimizer_state_for_peft(model)
    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    target_modules = extract_peft_target_modules(model)
    assert any(name.endswith(".default") for name in target_modules)

    normalizers = load_from_optimizer(
        model, str(opt_path), target_modules=target_modules
    )

    # Every adapter-suffixed module should have a normalizer.
    assert set(normalizers.keys()) == target_modules
    for norm in normalizers.values():
        assert isinstance(norm, AdamNormalizer)
        assert norm.weight_avg_sq.ndim == 2


def test_load_from_peft_model_without_target_modules(tmp_path):
    """target_modules=None on a PEFT model still loads every LoRA weight."""
    model = _create_peft_model()
    opt_state = _fake_optimizer_state_for_peft(model)
    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    normalizers = load_from_optimizer(model, str(opt_path))
    # Every LoRA weight produces one normalizer; names carry the adapter
    # suffix because adapter_suffix is appended unconditionally for PEFT.
    assert len(normalizers) > 0
    for name in normalizers:
        assert name.endswith(".default")


def test_load_from_peft_strip_adapter_target_modules_misses(tmp_path):
    """If a caller passes target_modules WITHOUT the adapter suffix (the bug
    we just fixed), nothing matches and the assertion fires."""
    model = _create_peft_model()
    opt_state = _fake_optimizer_state_for_peft(model)
    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    target_modules = extract_peft_target_modules(model)
    stripped = {name.removesuffix(".default") for name in target_modules}

    with pytest.raises(AssertionError, match="No optimizer second moments"):
        load_from_optimizer(model, str(opt_path), target_modules=stripped)


# ---------------------------------------------------------------------------
# Integration tests with real training checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_adam_checkpoint():
    """Load from a real AdamW training checkpoint and verify values match."""
    ckpt_path, model = _train_checkpoint("adamw_torch")
    opt_state = torch.load(
        f"{ckpt_path}/optimizer.pt", map_location="cpu", weights_only=False
    )

    normalizers = load_from_optimizer(model, ckpt_path)

    assert len(normalizers) > 0
    for norm in normalizers.values():
        assert isinstance(norm, AdamNormalizer)

    # Verify loaded values match the raw checkpoint
    params = list(model.named_parameters())
    for idx, (name, _param) in enumerate(params):
        if not name.endswith(".weight"):
            continue
        module_name = name.removesuffix(".weight")
        if module_name not in normalizers:
            continue

        raw = opt_state["state"][idx]["exp_avg_sq"]
        norm = normalizers[module_name]
        assert isinstance(norm, AdamNormalizer)
        torch.testing.assert_close(norm.weight_avg_sq.cpu(), raw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_adafactor_checkpoint():
    """Load from a real Adafactor training checkpoint and verify values match."""
    ckpt_path, model = _train_checkpoint("adafactor")
    opt_state = torch.load(
        f"{ckpt_path}/optimizer.pt", map_location="cpu", weights_only=False
    )

    normalizers = load_from_optimizer(model, ckpt_path)

    assert len(normalizers) > 0
    for norm in normalizers.values():
        assert isinstance(norm, AdafactorNormalizer)

    # Verify loaded values match the raw checkpoint
    params = list(model.named_parameters())
    for idx, (name, _param) in enumerate(params):
        if not name.endswith(".weight"):
            continue
        module_name = name.removesuffix(".weight")
        if module_name not in normalizers:
            continue

        raw_row = opt_state["state"][idx]["exp_avg_sq_row"]
        raw_col = opt_state["state"][idx]["exp_avg_sq_col"]
        norm = normalizers[module_name]
        assert isinstance(norm, AdafactorNormalizer)
        torch.testing.assert_close(norm.row.cpu(), raw_row)
        torch.testing.assert_close(norm.col.cpu(), raw_col)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_8bit_adam_checkpoint():
    """Load from a real 8-bit Adam (bitsandbytes) training checkpoint."""
    ckpt_path, model = _train_checkpoint("adamw_bnb_8bit")
    opt_state = torch.load(
        f"{ckpt_path}/optimizer.pt", map_location="cpu", weights_only=False
    )

    normalizers = load_from_optimizer(model, ckpt_path)

    assert len(normalizers) > 0
    for norm in normalizers.values():
        assert isinstance(norm, AdamNormalizer)

    # Verify loaded values match the raw checkpoint
    params = list(model.named_parameters())
    for idx, (name, _param) in enumerate(params):
        if not name.endswith(".weight"):
            continue
        module_name = name.removesuffix(".weight")
        if module_name not in normalizers:
            continue

        raw = opt_state["state"][idx]["__bnb_optimizer_quant_state__"]["state2"]
        norm = normalizers[module_name]
        assert isinstance(norm, AdamNormalizer)
        torch.testing.assert_close(norm.weight_avg_sq.cpu(), raw)
