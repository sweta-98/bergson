import gc

import torch
from peft import PeftModel
from transformers import PreTrainedModel


def test_fwd_bwd(
    model: PreTrainedModel | PeftModel,
    token_batch_size: int,
) -> None:
    """One worst-case fwd+bwd at ``token_batch_size``.

    Raises if the configured batch size will not fit.
    """
    device = next(model.parameters()).device
    if device.type != "cuda":
        return

    max_seq_len = (
        getattr(model.config, "max_position_embeddings", 0) or token_batch_size
    )
    seq_len = min(token_batch_size, max_seq_len)
    num_seqs = max(1, token_batch_size // seq_len)

    gc.collect()
    torch.cuda.empty_cache()

    try:
        input_ids = torch.randint(
            0, 10, (num_seqs, seq_len), device=device, dtype=torch.long
        )
        labels = torch.randint(
            0, 10, (num_seqs, seq_len), device=device, dtype=torch.long
        )
        logits = model(input_ids).logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
        )
        loss.backward()
    except torch.cuda.OutOfMemoryError as e:
        raise torch.cuda.OutOfMemoryError(
            f"Pre-flight VRAM check failed for token_batch_size={token_batch_size} "
            f"({num_seqs} x {seq_len}) on {device}. Lower --token_batch_size and "
            f"retry. Underlying error: {e}"
        ) from e
    finally:
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
