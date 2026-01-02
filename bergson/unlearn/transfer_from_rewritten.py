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

import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.optim import AdamW

from bergson.unlearn.collator import get_ds_transfer_collator
from bergson.unlearn.data import AlternatingDataset, PhaseUpdateCallback
from bergson.unlearn.hook import ActivationCapture
from bergson.unlearn.token_alignment import SnapAlignmentStrategy
from bergson.unlearn.utils import EvalCallback
from bergson.utils.utils import assert_type
from bergson.unlearn.muon import MuonAdamW


# Dataset paths - these match the paths used in rmu_data.py
# Datasets are saved to rmu/ subdirectory relative to project root
location = "isambard"
# location = "mnt"
if location == "mnt":
    BIO_FORGET_PATH = "/mnt/ssd-1/lucia/bergson/rmu/bio-forget"
    WMDP_REWRITTEN_PATH = "/mnt/ssd-1/lucia/bergson/rmu/wmdp-lie-o-rewritten"
    BIO_RETAIN_PATH = "/mnt/ssd-1/lucia/bergson/rmu/bio-retain"

    OUTPUT_DIR = "/mnt/ssd-1/lucia/bergson/runs/bio_transfer"
    EVAL_INCLUDE_PATH = "/mnt/ssd-1/lucia/bergson/lm-eval-tasks"
else:
    BIO_FORGET_PATH = "/projects/a5k/public/lucia/rmu/bio-forget"
    WMDP_REWRITTEN_PATH = "/projects/a5k/public/lucia/rmu/wmdp-lie-o-rewritten"
    BIO_RETAIN_PATH = "/projects/a5k/public/lucia/rmu/bio-retain"

    OUTPUT_DIR = "/projects/a5k/public/lucia/runs/bio_transfer_test"
    EVAL_INCLUDE_PATH = "/home/a5k/lucia.a5k/bergson/bergson/unlearn/lm_eval_tasks"

STUDENT_MODEL_NAME = "EleutherAI/deep-ignorance-unfiltered"

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

PAIRS_PER_BATCH = 16
GRAD_ACCUMULATION = 1
LEARNING_RATE = 1e-4

NUM_PHASES = 50
# STEPS_PER_PHASE = 50
EXAMPLES_PER_PHASE = 8192

# Run evaluation every N steps
EVAL_STEPS = 50

LAMBDA_MSE = 0.5

# Token alignment strategy
ALIGNMENT_STRATEGY = SnapAlignmentStrategy()


def is_debug():
    return os.environ.get("RANK", "0") == "0"

    # return os.environ.get("DEBUG", "0") == "1"


def get_optimizer(model, optim_type: str, lr: float):
    # Pass all model parameters to the wrapper; it handles the splitting
    if optim_type == "muon":
        return MuonAdamW(
            model.parameters(),
            # Use the Moonshot Muon implementation that 
            # enables equal lrs
            muon_lr=lr,
            adam_lr=lr,
        )
    elif optim_type == "adamw":
        return AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Invalid optimizer type: {optim_type}")


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

        assert hasattr(self, "state") and hasattr(self.state, "epoch")
        self.is_transfer_phase = "alignment_map" in inputs

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
        total_loss = self.lambda_mse * mse_loss_term  
        # No CE loss for now
        # #+ (1 - self.lambda_mse) * target_loss

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
        STUDENT_MODEL_NAME,
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

    if rank == 0 and not retain_ds_path.exists():
        exit()
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Load and tokenize datasets
        ds = load_datasets()
        if is_debug():
            print("Tokenize", flush=True)

        ds = tokenize_ds(ds, tokenizer)
        for ds_name, dataset in ds.items():
            if is_debug():
                print(
                    f"{ds_name} dataset length: {len(dataset)}, columns:"
                    f"{dataset.column_names}"
                )

        retain_ds = ds["retain"]

        # Mix in ultrachat
        from datasets import concatenate_datasets

        # Load ultrachat and prepare it for mixing
        ultrachat = load_dataset("stingning/ultrachat", split="train")
        ultrachat = assert_type(Dataset, ultrachat)
        # Mix in at 1:2 ratio
        ultrachat = ultrachat.select(range(min(len(ultrachat), len(retain_ds) // 2)))

        # Flatten ultrachat conversations to text
        def flatten_ultrachat(example):
            return {"text": "\n".join(example["data"])}

        ultrachat = ultrachat.map(flatten_ultrachat, remove_columns=ultrachat.column_names)
        ultrachat = ultrachat.map(lambda ex: tokenizer(ex["text"], truncation=True, max_length=SEQ_LEN), remove_columns=["text"])

        # Mix with retain set
        retain_ds = concatenate_datasets([retain_ds, ultrachat]).shuffle(seed=42)
        retain_ds.save_to_disk(str(retain_ds_path))

        if is_debug():
            print("Retain set len", len(retain_ds), flush=True)

        # Create dataset dict
        dataset_dict = DatasetDict({
            "bio_forget": ds["bio_forget"],
            "rewritten": ds["rewritten"],
            "retain": ds["retain"],
        })
        # Save to disk
        dataset_dict.save_to_disk(OUTPUT_DIR + "/aligned_tokenized_datasets")

        # Ensure bio and rewritten have the same length for alignment logic.
        assert len(ds["bio_forget"]) == len(ds["rewritten"]), (
            f"Bio-forget and rewritten datasets must have the same length for alignment. "
            f"Got {len(ds['bio_forget'])} and {len(ds['rewritten'])}."
        )

        transfer_ds = ds["bio_forget"].rename_column("input_ids", "source_input_ids")
        transfer_ds = transfer_ds.add_column(
            "target_input_ids", ds['rewritten']["input_ids"],
            new_fingerprint="transfer"
        )

        def filter(item):
            if len(item["source_input_ids"]) < SEQ_LEN:
                return False
            if len(item["target_input_ids"]) < SEQ_LEN:
                return False
            return True

        transfer_ds = transfer_ds.filter(filter)
        if is_debug():
            print(f"Filtered transfer dataset length: {len(transfer_ds)}")

        print("transfer ds length", len(transfer_ds), "saving to disk", flush=True)

        transfer_ds.save_to_disk(str(transfer_ds_path))
        print("done", flush=True)
    elif rank != 0 and not retain_ds_path.exists():
        # Wait for rank 0 to finish then user must re-run
        exit()

    transfer_ds = Dataset.load_from_disk(
        str(transfer_ds_path), keep_in_memory=False
    )
    retain_ds = Dataset.load_from_disk(str(retain_ds_path), keep_in_memory=False)

    total_examples = EXAMPLES_PER_PHASE * NUM_PHASES
    effective_batch_size = PAIRS_PER_BATCH * GRAD_ACCUMULATION * world_size
    max_steps = math.floor(total_examples / effective_batch_size)
    if is_debug():
        print(f"Total training steps: {max_steps}")

    STEPS_PER_PHASE = math.floor(
        EXAMPLES_PER_PHASE / (effective_batch_size)
    )
    print("steps per phase", STEPS_PER_PHASE)

    # Create alternating dataset that provides EXAMPLES_PER_PHASE
    # items per "epoch".
    train_dataset = AlternatingDataset(
        transfer_ds,
        retain_ds,
        rank,
        world_size,
        SEQ_LEN,
        NUM_PHASES,
        examples_per_phase=EXAMPLES_PER_PHASE,
    )
    phase_collator = get_ds_transfer_collator(
        pairs_per_batch=PAIRS_PER_BATCH,
        seq_len=SEQ_LEN,
        tokenizer=tokenizer,
        alignment_strategy=ALIGNMENT_STRATEGY,
    )

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
        **kwargs
    )

    callbacks_list = [
        EvalCallback(
            tokenizer=tokenizer,
            run_every_steps=EVAL_STEPS,
            ref_model=model,
            include_path=EVAL_INCLUDE_PATH,
            tasks=["wmdp_bio_robust", "wmdp_bio_cloze_verified", "mmlu"],
        ),
        PhaseUpdateCallback(STEPS_PER_PHASE),
    ]

    trainer_kwargs = {}
    if OPTIMIZER_TYPE == "muon":
        trainer_kwargs["optimizers"] = (
            get_optimizer(model, OPTIMIZER_TYPE, lr=LEARNING_RATE), 
            None
        )

    trainer = AlternatingDistillationTrainer(
        target_modules=TARGET_MODULES,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=phase_collator,
        callbacks=callbacks_list,
        lambda_mse=LAMBDA_MSE,
        **trainer_kwargs
    )

    if is_debug():
        print("Starting training...", flush=True)

    trainer.train()

    if hasattr(trainer, "hooks"):
        trainer.hooks.remove()

    trainer.save_model(os.path.join(OUTPUT_DIR, "aligned_model_final"))


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--wandb_run_name", type=str, help="WandB run name for logging", default="bio-transfer"
    )
    args = parser.parse_args()
    main(args)
