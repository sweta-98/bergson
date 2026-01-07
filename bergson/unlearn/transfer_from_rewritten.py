"""
Transfer activations from bio-forget to wmdp-lie-o-rewritten, then train on retain.

This script alternates between:
1. Transfer epoch: Transfer activations from bio-forget to wmdp-lie-o-rewritten
2. Retain epoch: Train on bio-retain set

Repeats for n=2 epochs total.
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

import os
from pathlib import Path
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from simple_parsing import field, ArgumentParser, ConflictResolution
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.optim import AdamW

from bergson.unlearn.collator import get_ds_transfer_collator
from bergson.unlearn.data import AlternatingDataset
from bergson.unlearn.hook import ActivationCapture
from bergson.unlearn.token_alignment import SnapAlignmentStrategy
from bergson.unlearn.utils import EvalCallback
from bergson.utils.utils import assert_type
# from bergson.unlearn.muon import MuonAdamW


# # Dataset paths - these match the paths used in rmu_data.py
# # Datasets are saved to rmu/ subdirectory relative to project root
# location = "isambard"
# # location = "mnt"
# if location == "mnt":
#     BIO_FORGET_PATH = "/mnt/ssd-1/lucia/bergson/rmu/bio-forget"
#     WMDP_REWRITTEN_PATH = "/mnt/ssd-1/lucia/bergson/rmu/wmdp-lie-o-rewritten"
#     BIO_RETAIN_PATH = "/mnt/ssd-1/lucia/bergson/rmu/bio-retain"

#     OUTPUT_DIR = "/mnt/ssd-1/lucia/bergson/runs/bio_transfer"
#     EVAL_INCLUDE_PATH = "/mnt/ssd-1/lucia/bergson/lm-eval-tasks"
# else:
#     BIO_FORGET_PATH = "/projects/a5k/public/lucia/rmu/bio-forget"
#     WMDP_REWRITTEN_PATH = "/projects/a5k/public/lucia/rmu/wmdp-lie-o-rewritten"
#     BIO_RETAIN_PATH = "/projects/a5k/public/lucia/rmu/bio-retain"

#     OUTPUT_DIR = "/projects/a5k/public/lucia/runs/bio_transfer_test"
#     EVAL_INCLUDE_PATH = "/home/a5k/lucia.a5k/bergson/bergson/unlearn/lm_eval_tasks"


location = "mnt"
# location = "mnt"
if location == "mnt":
    # BIO_FORGET_PATH = "/mnt/ssd-1/lucia/bergson/rmu/bio-forget"
    # WMDP_REWRITTEN_PATH = "/mnt/ssd-1/lucia/bergson/rmu/wmdp-lie-o-rewritten"
    # BIO_RETAIN_PATH = "/mnt/ssd-1/lucia/bergson/rmu/bio-retain"
    BIO_RETAIN_PATH = "/home/lucia/bio_retain"
    WMDP_REWRITTEN_PATH = "/home/lucia/wmdp-lie-o-rewritten"
    BIO_FORGET_PATH = "/home/lucia/bio-forget"

    # OUTPUT_DIR = "/mnt/ssd-1/lucia/bergson/runs/bio_transfer"
    OUTPUT_DIR = "/home/lucia/bio_tmp"
else:
    BIO_FORGET_PATH = "/projects/a5k/public/lucia/rmu/bio-forget"
    WMDP_REWRITTEN_PATH = "/projects/a5k/public/lucia/rmu/wmdp-lie-o-rewritten"
    BIO_RETAIN_PATH = "/projects/a5k/public/lucia/rmu/bio-retain"

    # DO NOT CHANGE EVER
    OUTPUT_DIR = "/projects/a5k/public/lucia/runs/bio_transfer_test"


TARGET_MODULES = [
    "gpt_neox.layers.1.mlp.dense_4h_to_h",
    "gpt_neox.layers.2",
    "gpt_neox.layers.4",
    "gpt_neox.layers.8.mlp.dense_4h_to_h",
    "gpt_neox.layers.12",
    "gpt_neox.layers.16.mlp.dense_4h_to_h",
    "gpt_neox.layers.20",
    "gpt_neox.layers.22",
    "gpt_neox.layers.24.mlp.dense_4h_to_h",
    "gpt_neox.layers.26",
    "gpt_neox.layers.28",
    "gpt_neox.layers.30",
    "gpt_neox.layers.31.mlp.dense_4h_to_h",
    "embed_out",
]


def is_debug():
    return os.environ.get("RANK", "0") == "0"

    # return os.environ.get("DEBUG", "0") == "1"


# def get_optimizer(model, optim_type: str, lr: float):
#     # Pass all model parameters to the wrapper; it handles the splitting
#     if optim_type == "muon":
#         return MuonAdamW(
#             model.parameters(),
#             # Use the Moonshot Muon implementation that 
#             # enables equal lrs
#             muon_lr=lr,
#             adam_lr=lr,
#         )
#     elif optim_type == "adamw":
#         return AdamW(model.parameters(), lr=lr)
#     else:
#         raise ValueError(f"Invalid optimizer type: {optim_type}")


class AlternatingDistillationTrainer(Trainer):
    """
    Trainer that alternates between transfer and retain phases.
    """

    def __init__(self, target_modules, lambda_mse: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_modules = target_modules
        self.lambda_mse = lambda_mse
        self.loss_fn = nn.MSELoss()
        self.hooks = ActivationCapture(self.model, self.target_modules)
        self.hooks.register()

        self.log_mse_accum = 0.0
        self.log_ce_accum = 0.0
        self.log_source_ce_accum = 0.0
        self.log_target_ce_accum = 0.0
        self.log_steps_count = 0

        self.is_transfer_phase = None
        self._last_epoch = -1

        # Ensure alignment_map is not removed by the Trainer
        self._set_signature_columns_if_needed()
        assert self._signature_columns is not None
        if "alignment_map" not in self._signature_columns:
            self._signature_columns.append("alignment_map")

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        self.hooks.clear()

        self.is_transfer_phase = (inputs["alignment_map"] != -1).any()

        if self.is_transfer_phase:
            return self._compute_transfer_loss(model, inputs, return_outputs)
        else:
            return self._compute_retain_loss(model, inputs, return_outputs)

    def _compute_transfer_loss(self, model, inputs, return_outputs=False):
        if is_debug():
            print("Compute transfer loss")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )

        if is_debug():
            print("Outputs obtained")
            print(outputs.logits.shape)
            print(inputs["labels"].shape)
            print(inputs["alignment_map"].shape)

        source_logits = outputs.logits[0::2]
        source_labels = inputs["labels"][0::2]
        target_logits = outputs.logits[1::2]
        target_labels = inputs["labels"][1::2]

        source_loss = F.cross_entropy(
            source_logits[..., :-1, :].flatten(0, 1),
            source_labels[..., 1:].flatten(),
            ignore_index=-100,
        ).detach().float().item()

        target_loss = F.cross_entropy(
            target_logits[..., :-1, :].flatten(0, 1),
            target_labels[..., 1:].flatten(),
            ignore_index=-100,
        ).detach().float().item()

        # --- STEP 1: Slice the Map ---
        # The collator returns [Source, Target, Source, Target...].
        # The alignment map for Source rows is at even indices [0::2].
        # The odd indices are just dummy -1s, so we ignore them.
        alignment_map = inputs["alignment_map"][0::2]

        mse_loss_total = torch.tensor(0.0, device=model.device, dtype=torch.bfloat16)

        for name in self.target_modules:
            act = self.hooks.activations[name]  # Shape: [2N, SeqLen, Hidden]
            # print("act shape", act.shape)

            # Source is at even indices
            source_act = act[0::2]
            # Target is at odd indices
            target_act = act[1::2]
            # source_act = source_acts[name] # Shape: [N, SeqLen, Hidden]
            # target_act = target_acts[name] # Shape: [N, SeqLen, Hidden]

            # Mask valid positions (ignore padding in the map)
            valid_mask = alignment_map != -1

            # Create safe indices for gathering (replace -1 with 0)
            safe_map = alignment_map.clone()
            safe_map[~valid_mask] = 0

            # Expand map to match hidden dimension: [N, SeqLen] -> [N, SeqLen, Hidden]
            hidden_dim = target_act.shape[-1]
            gather_indices = safe_map.unsqueeze(-1).expand(-1, -1, hidden_dim)

            # Gather: Pull the hidden states from target that align with source
            aligned_target_act = torch.gather(target_act, 1, gather_indices)

            # --- Compute & Normalize Loss ---

            # Calculate MSE per element [N, SeqLen, Hidden]
            # We detach aligned_target_act because we only want to update the
            # source representation
            raw_mse_loss = F.mse_loss(
                source_act, aligned_target_act.detach(), reduction="none"
            )
            # print("raw mse loss", raw_mse_loss[0, :10])

            # 1. Average over the Hidden Dimension first -> [N, SeqLen]
            # This keeps the loss magnitude interpretable (e.g., 0.05 instead of 200.0)
            mse_per_token = raw_mse_loss.mean(dim=-1)

            # 2. Zero out loss for invalid/padded tokens
            masked_loss = mse_per_token * valid_mask
            # print("masked loss", masked_loss[0, :10])
            # 3. Average over the number of valid tokens only
            if valid_mask.sum() > 0:
                mse_loss_total += masked_loss.sum() / valid_mask.sum()

        # Average across all layers we are targeting
        mse_loss_term = mse_loss_total / len(self.target_modules)

        # Scale the loss according to lambda
        total_loss = self.lambda_mse * mse_loss_term + (1 - self.lambda_mse) * target_loss

        if model.training:
            self.log_ce_accum += (target_loss + source_loss)
            self.log_source_ce_accum += source_loss
            self.log_target_ce_accum += target_loss
            self.log_mse_accum += mse_loss_term.detach().float().item()
            self.log_steps_count += 1

        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_retain_loss(self, model, inputs, return_outputs=False):
        if is_debug():
            print("Compute retain loss")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        ce_loss = outputs.loss

        if model.training:
            self.log_ce_accum += ce_loss.detach().float().item()
            self.log_steps_count += 1

        return (ce_loss, outputs) if return_outputs else ce_loss

    def log(self, logs, start_time=None):
        if self.log_steps_count > 0:
            logs["ce_loss"] = self.log_ce_accum / self.log_steps_count
            logs["source_ce_loss"] = self.log_source_ce_accum / self.log_steps_count
            logs["target_ce_loss"] = self.log_target_ce_accum / self.log_steps_count
            if self.is_transfer_phase:
                logs["mse_loss"] = self.log_mse_accum / self.log_steps_count
            logs["phase"] = "transfer" if self.is_transfer_phase else "retain"  # type: ignore

            self.log_ce_accum = 0.0
            self.log_source_ce_accum = 0.0
            self.log_target_ce_accum = 0.0
            self.log_mse_accum = 0.0
            self.log_steps_count = 0
        super().log(logs)


def main(args):
    if not torch.cuda.is_available():
        import sys
        print("Error: CUDA is not available. Aborting to prevent CPU hang.", file=sys.stderr)
        sys.exit(1)

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"Initializing Rank {rank} of {world_size}")

    # Set the CUDA device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"Rank {rank}: Using GPU {local_rank}", flush=True)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # Because gradient checkpointing will be enabled
        use_cache=False,
    )
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    transfer_ds_path = Path(OUTPUT_DIR + "/transfer_ds")
    retain_ds_path = Path(OUTPUT_DIR + "/mixed_retain_ds")

    assert transfer_ds_path.exists(), "Transfer dataset does not exist, run create_unlearn_data.py"
    assert retain_ds_path.exists(), "Retain dataset does not exist, run create_unlearn_data.py"

    transfer_ds = Dataset.load_from_disk(
        str(transfer_ds_path), keep_in_memory=False
    )
    retain_ds = Dataset.load_from_disk(str(retain_ds_path), keep_in_memory=False)

    effective_batch_size = args.micro_batch_size * args.grad_accumulation * world_size
    examples_per_phase = args.steps_per_phase * effective_batch_size
    total_training_steps = args.steps_per_phase * args.num_phases

    # Create alternating dataset that provides examples_per_phase
    # items from each dataset in an alternating fashion.
    train_dataset = AlternatingDataset(
        transfer_ds,
        retain_ds,
        rank,
        world_size,
        examples_per_phase=examples_per_phase,
    )
    phase_collator = get_ds_transfer_collator(
        pairs_per_batch=args.micro_batch_size,
        seq_len=args.seq_len,
        tokenizer=tokenizer,
        alignment_strategy=SnapAlignmentStrategy(),
    )

    kwargs = {}
    if args.optimizer_type == "adamw":
        kwargs["optim"] = "adamw_bnb_8bit"
        
    training_args = TrainingArguments(
        run_name=args.wandb_run_name,
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        num_train_epochs=args.num_phases,
        learning_rate=args.learning_rate,
        logging_steps=10,
        bf16=True,
        save_strategy="no",
        local_rank=local_rank,
        report_to="wandb",
        remove_unused_columns=False,
        max_steps=total_training_steps,
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,
        overwrite_output_dir=True,
        accelerator_config={
            "dispatch_batches": False,
        },
        **kwargs
    )

    callbacks_list = [
        EvalCallback(
            tokenizer=tokenizer,
            run_every_steps=args.steps_per_phase * 8,
            ref_model=model,
            include_path=EVAL_INCLUDE_PATH,
            tasks=["wmdp_bio_robust", "wmdp_bio_cloze_verified", "mmlu"],
        ),
    ]

    trainer_kwargs = {}
    # if args.optimizer_type == "muon":
    #     trainer_kwargs["optimizers"] = (
    #         get_optimizer(model, args.optimizer_type, lr=args.learning_rate), 
    #         None
    #     )

    trainer = AlternatingDistillationTrainer(
        target_modules=TARGET_MODULES,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=phase_collator,
        callbacks=callbacks_list,
        lambda_mse=args.lambda_mse,
        **trainer_kwargs
    )

    if is_debug():
        print("Training...", flush=True)

    trainer.train()

    trainer.save_model(os.path.join(OUTPUT_DIR, "aligned_model_final"))


@dataclass
class TransferFromRewrittenConfig:
    wandb_run_name: str = "bio-transfer"
    num_phases: int = 50
    steps_per_phase: int = 2
    micro_batch_size: int = 16
    grad_accumulation: int = 1
    learning_rate: float = 5e-5
    lambda_mse: float = 0.5
    optimizer_type: Literal["adamw", "muon"] = "adamw"
    seq_len: int = 1024
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"


if __name__ == "__main__":
    parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
    parser.add_arguments(TransferFromRewrittenConfig, dest="prog")
    prog: TransferFromRewrittenConfig = parser.parse_args().prog
    main(prog)