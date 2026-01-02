import torch
from typing import Any

from bergson.unlearn.token_alignment import AlignmentStrategy


class TransferDataCollator:
    def __init__(self, alignment_strategy, pad_token_id, max_seq_len):
        self.alignment_strategy = alignment_strategy
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def _pad_tensor(self, tensor, pad_value):
        curr_len = len(tensor)
        if curr_len >= self.max_seq_len:
            return tensor[: self.max_seq_len]
        padding = torch.full(
            (self.max_seq_len - curr_len,), pad_value, 
            dtype=tensor.dtype, device=tensor.device
        )
        return torch.cat([tensor, padding])

    def __call__(self, batch):
        """Returns source and target input IDs interleaved into a single batch of input IDs"""
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        alignment_map_list = []

        for item in batch:
            source_tokens = item["source_input_ids"]
            target_tokens = item["target_input_ids"]
            
            # Align (CPU intensive, done here)
            alignment = self.alignment_strategy.align_tokens(source_tokens, target_tokens)
            
            source_t = torch.tensor(source_tokens, dtype=torch.long)
            target_t = torch.tensor(target_tokens, dtype=torch.long)
            map_t = torch.tensor(alignment, dtype=torch.long)
            
            # Masks
            source_m = torch.tensor(item.get("source_attention_mask", [1]*len(source_t)))
            target_m = torch.tensor(item.get("target_attention_mask", [1]*len(target_t)))

            # --- 2. Pad Individually ---
            p_source_id = self._pad_tensor(source_t, self.pad_token_id)
            p_target_id = self._pad_tensor(target_t, self.pad_token_id)
            
            p_source_m = self._pad_tensor(source_m, 0)
            p_target_m = self._pad_tensor(target_m, 0)
            
            # Pad Map with -1
            p_map = self._pad_tensor(map_t, -1)
            # Clamp map indices to ensure they are within max_seq_len
            p_map = torch.clamp(p_map, max=self.max_seq_len - 1)
            
            # Create Dummy Map for the target row (needed to keep tensor rectangular)
            p_dummy_map = torch.full_like(p_map, -1)

            # --- 3. Prepare Labels ---
            # source -> -100
            l_source = torch.full_like(p_source_id, -100)
            # target -> ID (masked pad)
            l_target = p_target_id.clone()
            l_target[l_target == self.pad_token_id] = -100

            # --- 4. Interleave into Lists ---
            # Order: [source_i, target_i]
            input_ids_list.extend([p_source_id, p_target_id])
            attention_mask_list.extend([p_source_m, p_target_m])
            labels_list.extend([l_source, l_target])
            
            # Alignment map follows input_ids structure
            alignment_map_list.extend([p_map, p_dummy_map])

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
            "alignment_map": torch.stack(alignment_map_list),
        }

class VanillaDataCollator:
    """Data collator for retain phase: standard language modeling."""

    def __init__(self, pad_token_id, max_seq_len):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def _pad_tensor(self, tensor, pad_value):
        """Pad or truncate tensor to fixed max_seq_len."""
        curr_len = len(tensor)
        if curr_len >= self.max_seq_len:
            return tensor[: self.max_seq_len]

        padding = torch.full(
            (self.max_seq_len - curr_len,),
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, padding])

    def __call__(self, features):
        input_ids = []
        attention_mask = []

        def ensure_tensor(val):
            if isinstance(val, torch.Tensor):
                return val.clone().detach()
            return torch.tensor(val, dtype=torch.long)

        for f in features:
            input_ids.append(ensure_tensor(f["input_ids"]))
            mask = f.get("attention_mask", [1] * len(f["input_ids"]))
            attention_mask.append(ensure_tensor(mask))

        # Pad to strict fixed length
        input_ids_batch = torch.stack(
            [self._pad_tensor(t, self.pad_token_id) for t in input_ids]
        )
        attention_mask_batch = torch.stack(
            [self._pad_tensor(t, 0) for t in attention_mask]
        )

        labels_batch = input_ids_batch.clone()
        # Mask padding in labels
        labels_batch[labels_batch == self.pad_token_id] = -100

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
        }


class PhaseAwareCollator:
    def __init__(self, transfer_collator, retain_collator, pairs_per_batch: int):
        """pairs_per_batch: purely for observability purposes."""
        
        self.transfer_collator = transfer_collator
        self.retain_collator = retain_collator
        self.pairs_per_batch = pairs_per_batch

    def __call__(self, batch: list[dict[str, Any]]):
        is_transfer_phase = "source_input_ids" in batch[0]

        if is_transfer_phase:
            return self.transfer_collator(batch)
        else:
            return self.retain_collator(batch)


def get_ds_transfer_collator(pairs_per_batch, seq_len: int, tokenizer, alignment_strategy: AlignmentStrategy):
    transfer_collator = TransferDataCollator(
        alignment_strategy, tokenizer.pad_token_id, seq_len
    )
    retain_collator = VanillaDataCollator(tokenizer.pad_token_id, seq_len)
    return PhaseAwareCollator(transfer_collator, retain_collator, pairs_per_batch=pairs_per_batch)