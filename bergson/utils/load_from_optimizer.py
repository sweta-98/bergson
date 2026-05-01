from enum import Enum
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict
from transformers import PreTrainedModel

from bergson.gradients import AdafactorNormalizer, AdamNormalizer, Normalizer


class OptimizerStateFormat(Enum):
    """Optimizer state format for a single module - Adafactor-style factored
    optimizer states (e.g. 2D+ modules in Adafactor optimizers), or Adam-style
    unfactored states."""

    UNFACTORED = 1
    FACTORED = 2


def get_optimizer_state_format(param_state) -> OptimizerStateFormat | None:
    if not isinstance(param_state, dict):
        return None

    if "exp_avg_sq" in param_state:
        return OptimizerStateFormat.UNFACTORED

    if "exp_avg_sq_row" in param_state:
        return OptimizerStateFormat.FACTORED

    bnb_state = param_state.get("__bnb_optimizer_quant_state__")
    if isinstance(bnb_state, dict) and "state2" in bnb_state:
        # 8-bit Adam
        return OptimizerStateFormat.UNFACTORED

    return None


def get_unfactored_second_moment(state: dict) -> torch.Tensor:
    """Return the second moment tensor for an unfactored optimizer state.

    Adam and 8-bit Adam always use unfactored tensors.
    Adafactor has multiple factored moment tensors for 2D+ parameters,
    and unfactored tensors for 1D parameters.
    """
    if "exp_avg_sq" in state:
        return state["exp_avg_sq"]
    return state["__bnb_optimizer_quant_state__"]["state2"]


def get_normalizers(
    optimizer_state,
    target_param_index_to_name,
    target_modules,
    adapter_suffix,
    include_bias,
    device,
):
    normalizers: dict[str, Normalizer] = {}
    for param_idx, state in optimizer_state["state"].items():
        param_idx = int(param_idx)
        if param_idx not in target_param_index_to_name:
            continue

        param_name = target_param_index_to_name[param_idx]
        if not param_name.endswith(".weight"):
            continue

        layer_name = param_name.removesuffix(".weight")
        module_name = layer_name.removeprefix("base_model.") + adapter_suffix

        if target_modules is not None and module_name not in target_modules:
            continue

        optimizer_format = get_optimizer_state_format(state)

        if optimizer_format is None:
            print("Unrecognized format, skipping normalizer for param_idx", param_idx)
            continue

        bias_exp_avg_sq = _get_bias_second_moment(
            layer_name, target_param_index_to_name, optimizer_state, include_bias
        )
        bias_on_device = (
            bias_exp_avg_sq.to(device) if bias_exp_avg_sq is not None else None
        )

        if optimizer_format == OptimizerStateFormat.UNFACTORED:
            exp_avg_sq = get_unfactored_second_moment(state)
            if exp_avg_sq.ndim != 2:
                continue
            normalizers[module_name] = AdamNormalizer(
                weight_avg_sq=exp_avg_sq.to(device),
                bias_avg_sq=bias_on_device,
            )
        elif optimizer_format == OptimizerStateFormat.FACTORED:
            row = state["exp_avg_sq_row"]
            col = state.get("exp_avg_sq_col")
            if row.ndim != 1 or col is None:
                continue
            normalizers[module_name] = AdafactorNormalizer(
                row=row.to(device),
                col=col.to(device),
                bias_avg_sq=bias_on_device,
            )

    return normalizers


def load_from_optimizer(
    model: PreTrainedModel | PeftModel,
    optimizer_state_path: str,
    include_bias: bool = False,
    target_modules: set[str] | None = None,
) -> dict[str, Normalizer]:
    """Load optimizer second moments from a checkpoint and create normalizer
    instances for each target linear layer.

    Auto-detects the optimizer format:

    - Adam/AdamW: ``exp_avg_sq`` -> AdamNormalizer
    - Adafactor: ``exp_avg_sq_row``/``exp_avg_sq_col`` -> AdafactorNormalizer
    - 8-bit Adam (BitsAndBytes): ``state2`` -> AdamNormalizer

    Args:
        model: The model whose parameter names are used to map optimizer
            state indices to layer names.
        optimizer_state_path: Path to either a checkpoint directory containing
            ``optimizer.pt`` or directly to an optimizer state file.
        include_bias: Whether to include bias second moments.
        target_modules: Optional set of module names to include. If ``None``,
            all linear layers are included.

    Returns:
        Dictionary mapping layer names to normalizer instances.
    """
    optimizer_path = Path(optimizer_state_path)
    if optimizer_path.is_dir():
        optimizer_path = optimizer_path / "optimizer.pt"

    optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)

    # The optimizer state is keyed by position in the trainable parameter list.
    # For PEFT checkpoints, only include PEFT params.
    adapter_suffix = ""
    if isinstance(model, PeftModel):
        st = get_peft_model_state_dict(model)
        params_for_index = list(st.items())
        # peft serializes LoRA keys without the active adapter name (e.g.
        # ``...lora_A.weight``), but extract_peft_target_modules and the
        # actual submodule paths include it (``...lora_A.default``). Append
        # the adapter name so module_name lookups match target_modules.
        adapters = list(model.peft_config.keys())
        if len(adapters) == 1:
            adapter_suffix = "." + adapters[0]
    else:
        params_for_index = list(model.named_parameters())

    target_param_index_to_name: dict[int, str] = {}
    for idx, (name, _param) in enumerate(params_for_index):
        target_param_index_to_name[idx] = name

    device = next(model.parameters()).device

    normalizers = get_normalizers(
        optimizer_state,
        target_param_index_to_name,
        target_modules,
        adapter_suffix,
        include_bias,
        device,
    )
    assert normalizers, (
        f"No optimizer second moments found in '{optimizer_state_path}'. "
        "Ensure the checkpoint was saved from an Adam-family or Adafactor optimizer."
    )

    types = {type(n).__name__ for n in normalizers.values()}
    print(
        f"Loaded {len(normalizers)} normalizers ({', '.join(types)}) "
        f"from '{optimizer_state_path}'"
    )
    return normalizers


def _get_bias_second_moment(
    layer_name: str,
    param_index_to_name: dict[int, str],
    optimizer_state: dict,
    include_bias: bool,
) -> torch.Tensor | None:
    """Look up bias exp_avg_sq for a layer, if present and requested."""
    if not include_bias:
        return None

    bias_name = layer_name + ".bias"
    for idx, name in param_index_to_name.items():
        if name == bias_name:
            bias_state = optimizer_state["state"].get(idx)
            optimizer_format = get_optimizer_state_format(bias_state)
            if optimizer_format == OptimizerStateFormat.UNFACTORED:
                return get_unfactored_second_moment(bias_state)
            return None

    return None
