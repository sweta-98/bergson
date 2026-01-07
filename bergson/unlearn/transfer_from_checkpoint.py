"""
forget activations from bio-forget to wmdp-lie-o-rewritten, then train on retain.

This script alternates between:
1. forget epoch: forget activations from bio-forget to wmdp-lie-o-rewritten
2. Retain epoch: Train on bio-retain set

Repeats for n=2 epochs total.
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers import BitsAndBytesConfig
from torch.optim import AdamW

from bergson.utils.utils import assert_type
from bergson.unlearn.collator import VanillaDataCollator
from bergson.unlearn.data import AlternatingDataset
from bergson.unlearn.hook import ActivationCapture
from bergson.unlearn.utils import EvalCallback
# from bergson.unlearn.muon import MuonAdamW


# Dataset paths - these match the paths used in rmu_data.py
# Datasets are saved to rmu/ subdirectory relative to project root
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
    EVAL_INCLUDE_PATH = "/home/lucia/bergson/bergson/unlearn/lm_eval_tasks"
    # EVAL_INCLUDE_PATH = "/mnt/ssd-1/lucia/bergson/lm-eval-tasks"
else:
    BIO_FORGET_PATH = "/projects/a5k/public/lucia/rmu/bio-forget"
    WMDP_REWRITTEN_PATH = "/projects/a5k/public/lucia/rmu/wmdp-lie-o-rewritten"
    BIO_RETAIN_PATH = "/projects/a5k/public/lucia/rmu/bio-retain"

    # DO NOT CHANGE EVER
    OUTPUT_DIR = "/projects/a5k/public/lucia/runs/bio_transfer_test"
    EVAL_INCLUDE_PATH = "/home/a5k/lucia.a5k/bergson/bergson/unlearn/lm_eval_tasks"

STUDENT_MODEL_NAME = "EleutherAI/deep-ignorance-unfiltered"

TEACHER_MODEL_NAME = "EleutherAI/deep-ignorance-pretraining-stage-unfiltered"
TEACHER_CHECKPOINT = "global_step38144"

SEQ_LEN = 1024
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

OPTIMIZER_TYPE = "adamw"
# OPTIMIZER_TYPE = "muon"

PAIRS_PER_BATCH = 2
GRAD_ACCUMULATION = 4
LEARNING_RATE = 1e-4

NUM_PHASES = 50
EXAMPLES_PER_PHASE = 256 #8192
# STEPS_PER_PHASE = 50

# Run evaluation every N steps
EVAL_STEPS = 50

LAMBDA_MSE = 0.1


def is_debug():
    return os.environ.get("RANK", "0") == "0"


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


class AlternatingCheckpointTransferTrainer(Trainer):
    """
    Trainer that alternates between forget and retain phases.
    """

    def __init__(self, teacher_model, target_modules, lambda_mse: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
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

        self.is_forget_phase = None

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        self.hooks.clear()

        assert hasattr(self, "state") and hasattr(self.state, "epoch")
        # Check dataset phase directly (forget phase is when phase_idx is even)
        assert hasattr(self.train_dataset, "is_first_phase"), "Train dataset must implement is_first_phase()"
        self.is_forget_phase = self.train_dataset.is_first_phase()

        if self.is_forget_phase:
            return self._compute_forget_loss(model, inputs, return_outputs)
        else:
            return self._compute_retain_loss(model, inputs, return_outputs)

    def _compute_forget_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["input_ids"],
        )

        # Compute acts with early model checkpoint
        teacher_hooks = ActivationCapture(self.teacher_model, self.target_modules)
        teacher_hooks.register()
        self.teacher_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["input_ids"],
        )
        # if is_debug():
            # print("VRAM after teacher fwd", torch.cuda.memory.memory_allocated() / 1024 ** 3, "GB")

        # Compute MSE with early checkpoint activations
        mse_loss_total = torch.tensor(data=0.0, device=model.device, dtype=torch.bfloat16)
        for name in self.target_modules:
            source_act = self.hooks.activations[name]
            target_act = teacher_hooks.activations[name].to(dtype=source_act.dtype)  # Shape: [N, SeqLen, Hidden]

            raw_mse_loss = F.mse_loss(
                source_act, target_act.detach(), reduction="none"
            )

            valid_mask = inputs["input_ids"] != -1

            # Average over the Hidden Dimension first -> [N, SeqLen]
            # This keeps the loss magnitude interpretable (e.g., 0.05 instead of 200.0)
            mse_per_token = raw_mse_loss.mean(dim=-1)

            # Zero out loss for invalid/padded tokens
            masked_loss = mse_per_token * valid_mask

            # Average over the number of valid tokens
            if valid_mask.sum() > 0:
                mse_loss_total += masked_loss.sum() / valid_mask.sum()

        self.hooks.clear()
        teacher_hooks.clear()

        mse_loss_term = mse_loss_total / len(self.target_modules)
        total_loss = self.lambda_mse * mse_loss_term
        
        if model.training:
            self.log_mse_accum += mse_loss_term.detach().float().item()
            self.log_steps_count += 1

        # if is_debug():
            # print("loss dtype", total_loss, total_loss.dtype, flush=True)
        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_retain_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["input_ids"],
        )
        ce_loss = outputs.loss

        if model.training:
            self.log_ce_accum += ce_loss.detach().float().item()
            self.log_steps_count += 1

        return (ce_loss, outputs) if return_outputs else ce_loss

    def log(self, logs, start_time=None):
        if self.log_steps_count > 0:
            logs["ce_loss"] = self.log_ce_accum / self.log_steps_count
            if self.is_forget_phase:
                logs["mse_loss"] = self.log_mse_accum / self.log_steps_count
            logs["phase"] = "forget" if self.is_forget_phase else "retain"  # type: ignore

            self.log_ce_accum = 0.0
            self.log_mse_accum = 0.0
            self.log_steps_count = 0

        super().log(logs)


def load_datasets():
    """Load all required datasets."""
    # Try to resolve paths - check if relative or absolute
    project_root = Path(__file__).parent.parent.parent

    def resolve_path(path):
        if os.path.isabs(path):
            return path
        full_path = project_root / path
        if full_path.exists():
            return str(full_path)
        return path

    bio_forget_path = resolve_path(BIO_FORGET_PATH)
    rewritten_path = resolve_path(WMDP_REWRITTEN_PATH)
    retain_path = resolve_path(BIO_RETAIN_PATH)

    print(f"Loading bio-forget from: {bio_forget_path}")
    print(f"Loading wmdp-lie-o-rewritten from: {rewritten_path}")
    print(f"Loading bio-retain from: {retain_path}")

    try:
        bio_forget = Dataset.load_from_disk(bio_forget_path)
        rewritten = Dataset.load_from_disk(rewritten_path)
        retain = Dataset.load_from_disk(retain_path)
    except Exception as e:
        print(f"Error loading from disk: {e}")
        print("Trying to load from HuggingFace Hub...")
        bio_forget = load_dataset(
            "Unlearning/rmu-training-data", data_files="bio-forget-corpus.jsonl"
        )
        rewritten = load_dataset("Unlearning/wmdp-lie-o-rewritten")
        retain = load_dataset(
            "Unlearning/rmu-training-data", data_files="bio-retain-corpus.jsonl"
        )

    return {
        "bio_forget": assert_type(Dataset, bio_forget),
        "rewritten": assert_type(Dataset, rewritten),
        "retain": assert_type(Dataset, retain),
    }


def tokenize_ds(datasets, tokenizer):
    """Tokenize datasets if not already tokenized."""

    def is_tokenized(example):
        return "input_ids" in example

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=SEQ_LEN,
        )

    for key in datasets:
        sample = datasets[key][0]
        if not is_tokenized(sample):
            datasets[key] = datasets[key].map(
                tokenize_function,
                batched=False,
                remove_columns=["text"],
            )

    return datasets


def process_retain_dataset(example, max_seq_len):
    """Process retain dataset: truncate and add attention mask."""
    input_ids = example.get("input_ids", [])
    
    # Convert to list if needed
    if not isinstance(input_ids, list):
        input_ids = input_ids.tolist()
    
    # Truncate
    input_ids = input_ids[:max_seq_len]
    
    # Create attention mask
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def main(args):
    if not torch.cuda.is_available():
        import sys

        print(
            "Error: CUDA is not available. Aborting to prevent CPU hang.",
            file=sys.stderr,
        )
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
        STUDENT_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # Because gradient checkpointing will be enabled
        use_cache=False,
    )
    model.gradient_checkpointing_enable()
    if rank == 0:
        print("Loaded model", flush=True)


    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use transfer_from_rewritten.py to create datasets
    transfer_ds_path = Path(OUTPUT_DIR + "/transfer_ds")
    retain_ds_path = Path(OUTPUT_DIR + "/mixed_retain_ds")

    transfer_ds = Dataset.load_from_disk(str(transfer_ds_path), keep_in_memory=False)
    forget_ds = transfer_ds.remove_columns(["target_input_ids"]).rename_column(
        "source_input_ids", "input_ids"
    )
    del transfer_ds

    retain_ds = Dataset.load_from_disk(str(retain_ds_path), keep_in_memory=False)
    
    # Process datasets: truncate and add attention masks (replicating generator logic)
    # Note: Only attention mask processing needed here since we don't use transfer logic
    if is_debug():
        print("Processing forget dataset: truncating and adding attention masks", flush=True)
    forget_ds = forget_ds.map(
        lambda ex: process_retain_dataset(ex, SEQ_LEN),
        batched=False,
    )
    
    if is_debug():
        print("Processing retain dataset: truncating and adding attention masks", flush=True)
    retain_ds = retain_ds.map(
        lambda ex: process_retain_dataset(ex, SEQ_LEN),
        batched=False,
    )

    # Create alternating dataset that provides EXAMPLES_PER_PHASE
    # items per "epoch".
    train_dataset = AlternatingDataset(
        forget_ds,
        retain_ds,
        rank,
        world_size,
        examples_per_phase=EXAMPLES_PER_PHASE,
    )
    if rank == 0:
        print("Created alternating dataset", flush=True)

    # Create data collator to handle padding
    collator = VanillaDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=SEQ_LEN,
    )

    total_examples = EXAMPLES_PER_PHASE * NUM_PHASES
    effective_batch_size = PAIRS_PER_BATCH * GRAD_ACCUMULATION * world_size
    max_steps = math.floor(total_examples / effective_batch_size)
    steps_per_phase = math.floor(max_steps / NUM_PHASES)
    
    if is_debug():
        print(f"Total training steps: {max_steps}")

    kwargs = {}
    if OPTIMIZER_TYPE == "adamw":
        kwargs["optim"] = "adamw_bnb_8bit"

    training_args = TrainingArguments(
        run_name=args.wandb_run_name,
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PAIRS_PER_BATCH,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=NUM_PHASES,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        bf16=True,
        save_strategy="no",
        local_rank=local_rank,
        report_to="wandb",
        remove_unused_columns=False,
        max_steps=max_steps,
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,
        overwrite_output_dir=True,
        **kwargs,
    )

    callbacks_list = [
        EvalCallback(
            tokenizer=tokenizer,
            run_every_steps=steps_per_phase,
            ref_model=model,
            include_path=EVAL_INCLUDE_PATH,
            tasks=["wmdp_bio_robust", "wmdp_bio_cloze_verified", "mmlu"],
        ),
    ]

    trainer_kwargs = {}
    # if OPTIMIZER_TYPE == "muon":
    #     trainer_kwargs["optimizers"] = (
    #         get_optimizer(model, OPTIMIZER_TYPE, lr=LEARNING_RATE),
    #         None,
    #     )

    if is_debug():
            print("VRAM before teacher load", torch.cuda.memory.memory_allocated() / 1024 ** 3, "GB")

        
    quantization_cfg = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_NAME,
        quantization_config=quantization_cfg,
        trust_remote_code=True,
        revision=TEACHER_CHECKPOINT,
    )
    teacher_model.eval()

    if is_debug():
        print("VRAM after teacher load", torch.cuda.memory.memory_allocated() / 1024 ** 3, "GB")

    trainer = AlternatingCheckpointTransferTrainer(
        teacher_model=teacher_model,
        target_modules=TARGET_MODULES,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=callbacks_list,
        lambda_mse=LAMBDA_MSE,
        **trainer_kwargs,
    )

    if is_debug():
        print("Starting training...", flush=True)

    trainer.train()

    if hasattr(trainer, "hooks"):
        trainer.hooks.remove()

    trainer.save_model(os.path.join(OUTPUT_DIR, "ckpt_transferred_model_final"))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="WandB run name for logging",
        default="bio-forget",
    )
    args = parser.parse_args()
    main(args)
