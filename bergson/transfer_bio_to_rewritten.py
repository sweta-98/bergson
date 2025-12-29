"""
Transfer activations from bio-forget to wmdp-lie-o-rewritten, then train on retain.

This script alternates between:
1. Transfer epoch: Transfer activations from bio-forget to wmdp-lie-o-rewritten
2. Retain epoch: Train on bio-retain set

Repeats for n=2 epochs total.
"""

from timeit import timeit
import os
import math
from pathlib import Path

import wandb
from torch.utils.data import IterableDataset as TorchIterableDataset
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    AutoTokenizer,
)
from datasets import load_from_disk, load_dataset
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM 

from bergson.utils.utils import assert_type
from bergson.unlearn.hook import ActivationCapture
from bergson.unlearn.token_alignment import SnapAlignmentStrategy

# Dataset paths - these match the paths used in rmu_data.py
# Datasets are saved to rmu/ subdirectory relative to project root
BIO_FORGET_PATH = "/projects/a5k/public/lucia/rmu/bio-forget"
WMDP_REWRITTEN_PATH = "/projects/a5k/public/lucia/rmu/wmdp-lie-o-rewritten"
BIO_RETAIN_PATH = "/projects/a5k/public/lucia/rmu/bio-retain"

STUDENT_MODEL_NAME = "EleutherAI/deep-ignorance-unfiltered"
OUTPUT_DIR = "/projects/a5k/public/lucia/runs/bio_to_rewritten_transfer_10_epochs"

SEQ_LEN = 1024
TARGET_MODULES = [
    "gpt_neox.layers.8.mlp.dense_4h_to_h",
    "gpt_neox.layers.16.mlp.dense_4h_to_h",
    "gpt_neox.layers.31.mlp.dense_4h_to_h",
    "embed_out",
]

PAIRS_PER_BATCH = 4
GRAD_ACCUMULATION = 1
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10  # Total epochs: 1 transfer + 1 retain, repeated
MAX_DS_LENGTH = 200_000

# Run evaluation every N steps
EVAL_STEPS = 50

LAMBDA_MSE = 0.8

# Token alignment strategy
ALIGNMENT_STRATEGY = SnapAlignmentStrategy()


class TransferDataCollator:
    """Data collator for transfer phase: bio-forget -> wmdp-lie-o-rewritten."""
    
    def __init__(self, alignment_strategy, pad_token_id, max_seq_len):
        self.alignment_strategy = alignment_strategy
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
    
    def _pad_tensor(self, tensor, pad_value):
        """Pad or truncate tensor to fixed max_seq_len."""
        curr_len = len(tensor)
        if curr_len >= self.max_seq_len:
            return tensor[:self.max_seq_len]
        
        padding = torch.full((self.max_seq_len - curr_len,), pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding])

    def __call__(self, features):
        bio_forget_ids = []
        rewritten_ids = []
        bio_forget_mask = []
        rewritten_mask = []

        def ensure_tensor(val):
            if isinstance(val, torch.Tensor):
                return val.clone().detach()
            return torch.tensor(val, dtype=torch.long)

        for f in features:
            # Align tokens between bio-forget and rewritten
            bio_tokens = f["bio_forget_input_ids"]
            rewritten_tokens = f["rewritten_input_ids"]
            bio_mask = f.get("bio_forget_attention_mask", [1] * len(bio_tokens))
            rewritten_mask_val = f.get("rewritten_attention_mask", [1] * len(rewritten_tokens))
            
            aligned_bio, aligned_rewritten, aligned_bio_mask, aligned_rewritten_mask = \
                ALIGNMENT_STRATEGY.align_tokens(
                    bio_tokens,
                    rewritten_tokens,
                    bio_mask,
                    rewritten_mask_val,
                )
            
            bio_forget_ids.append(ensure_tensor(aligned_bio))
            rewritten_ids.append(ensure_tensor(aligned_rewritten))
            bio_forget_mask.append(ensure_tensor(aligned_bio_mask))
            rewritten_mask.append(ensure_tensor(aligned_rewritten_mask))

        # Pad everything to strict fixed length (SEQ_LEN) to satisfy Accelerate's requirements
        combined_ids = bio_forget_ids + rewritten_ids
        combined_masks = bio_forget_mask + rewritten_mask

        padded_ids = torch.stack([self._pad_tensor(t, self.pad_token_id) for t in combined_ids])
        padded_masks = torch.stack([self._pad_tensor(t, 0) for t in combined_masks])

        # Split back into bio and rewritten batches
        split_idx = len(bio_forget_ids)
        bio_batch = padded_ids[:split_idx]
        rewritten_batch = padded_ids[split_idx:]
        
        bio_mask_batch = padded_masks[:split_idx]
        rewritten_mask_batch = padded_masks[split_idx:]

        # Structure: [Bio_1, ..., Bio_N, Rewritten_1, ..., Rewritten_N]
        input_ids = torch.cat([bio_batch, rewritten_batch], dim=0)
        attention_mask = torch.cat([bio_mask_batch, rewritten_mask_batch], dim=0)

        # Labels: -100 for bio-forget (source), actual tokens for rewritten (target)
        bio_labels = torch.full_like(bio_batch, -100)
        rewritten_labels = rewritten_batch.clone()
        
        # Mask padding in the labels
        rewritten_labels[rewritten_labels == self.pad_token_id] = -100
        
        labels = torch.cat([bio_labels, rewritten_labels], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class RetainDataCollator:
    """Data collator for retain phase: standard language modeling."""
    
    def __init__(self, pad_token_id, max_seq_len):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def _pad_tensor(self, tensor, pad_value):
        """Pad or truncate tensor to fixed max_seq_len."""
        curr_len = len(tensor)
        if curr_len >= self.max_seq_len:
            return tensor[:self.max_seq_len]
        
        padding = torch.full((self.max_seq_len - curr_len,), pad_value, dtype=tensor.dtype, device=tensor.device)
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
        input_ids_batch = torch.stack([self._pad_tensor(t, self.pad_token_id) for t in input_ids])
        attention_mask_batch = torch.stack([self._pad_tensor(t, 0) for t in attention_mask])
        
        labels_batch = input_ids_batch.clone()
        # Mask padding in labels
        labels_batch[labels_batch == self.pad_token_id] = -100

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
        }


def transfer_generator(bio_forget_set, rewritten_set, rank, world_size, max_seq_len):
    """Generator for transfer phase: pairs bio-forget with wmdp-lie-o-rewritten."""
    if world_size > 1:
        bio_forget_set = bio_forget_set.shard(num_shards=world_size, index=rank)
        rewritten_set = rewritten_set.shard(num_shards=world_size, index=rank)

    bio_iter = iter(bio_forget_set)
    rewritten_iter = iter(rewritten_set)

    for bio_sample, rewritten_sample in zip(bio_iter, rewritten_iter):
        # Extract input_ids - handle both tokenized and non-tokenized datasets
        bio_ids = bio_sample.get("input_ids", bio_sample.get("text", []))
        rewritten_ids = rewritten_sample.get("input_ids", rewritten_sample.get("text", []))
        
        # If not already tokenized, we'll need to tokenize (but assume they are for now)
        if isinstance(bio_ids, str):
            raise ValueError("bio-forget dataset must be tokenized (have 'input_ids' field)")
        if isinstance(rewritten_ids, str):
            raise ValueError("wmdp-lie-o-rewritten dataset must be tokenized (have 'input_ids' field)")
        
        yield {
            "bio_forget_input_ids": bio_ids[:max_seq_len] if isinstance(bio_ids, list) else bio_ids[:max_seq_len].tolist(),
            "bio_forget_attention_mask": [1] * min(len(bio_ids), max_seq_len),
            "rewritten_input_ids": rewritten_ids[:max_seq_len] if isinstance(rewritten_ids, list) else rewritten_ids[:max_seq_len].tolist(),
            "rewritten_attention_mask": [1] * min(len(rewritten_ids), max_seq_len),
        }


def retain_generator(retain_set, rank, world_size, max_seq_len):
    """Generator for retain phase: standard language modeling."""
    if world_size > 1:
        retain_set = retain_set.shard(num_shards=world_size, index=rank)

    for sample in retain_set:
        input_ids = sample.get("input_ids", sample.get("text", []))
        
        if isinstance(input_ids, str):
            raise ValueError("bio-retain dataset must be tokenized (have 'input_ids' field)")
        
        yield {
            "input_ids": input_ids[:max_seq_len] if isinstance(input_ids, list) else input_ids[:max_seq_len].tolist(),
            "attention_mask": [1] * min(len(input_ids), max_seq_len),
        }


class LmEvalHarnessCallback(TrainerCallback):
    def __init__(self, tokenizer, run_every_steps=50):
        self.tokenizer = tokenizer
        self.run_every_steps = run_every_steps

    def _run_evaluation(self, args, state, control, **kwargs):
        # Only run on Rank 0
        if state.is_world_process_zero:
            start_time = timeit()
            model = kwargs["model"]

            # 1. Switch to Eval Mode
            was_training = model.training
            model.eval()

            print(
                f"\n[Step {state.global_step}] Running WMDP-Bio and MMLU..."
            )

            try:
                # 2. Wrap model for LM Eval Harness
                lm_wrapper = HFLM(
                    pretrained=model,
                    tokenizer=self.tokenizer,
                    batch_size=PAIRS_PER_BATCH,
                )

                # 3. Run Evaluation
                results = simple_evaluate(
                    model=lm_wrapper,  # type: ignore
                    tasks=["wmdp_bio", "mmlu"],  # type: ignore
                    log_samples=False,  # type: ignore
                )
                results = assert_type(dict, results)

                metrics = {}

                # 4. Extract WMDP Score
                if "wmdp_bio" in results["results"]:
                    acc = results["results"]["wmdp_bio"].get(
                        "acc,none", results["results"]["wmdp_bio"].get("acc")
                    )
                    print(f"WMDP-Bio Acc: {acc}")
                    metrics["wmdp_bio_acc"] = acc

                # 5. Extract MMLU Score
                acc_mmlu = None
                if "groups" in results and "mmlu" in results["groups"]:
                    acc_mmlu = results["groups"]["mmlu"].get(
                        "acc,none", results["groups"]["mmlu"].get("acc")
                    )
                elif "mmlu" in results["results"]:
                    acc_mmlu = results["results"]["mmlu"].get(
                        "acc,none", results["results"]["mmlu"].get("acc")
                    )

                if acc_mmlu is not None:
                    print(f"MMLU Acc: {acc_mmlu}")
                    metrics["mmlu_acc"] = acc_mmlu

                # 6. Log to WandB
                if metrics:
                    wandb.log(metrics, step=state.global_step)

            except Exception as e:
                print(f"Warning: Evaluation failed: {e}")

            # 7. Revert to Training Mode
            if was_training:
                model.train()

            end_time = timeit()
            print(f"Evaluation time: {end_time - start_time} seconds")

    def on_train_begin(self, args, state, control, **kwargs):
        self._run_evaluation(args, state, control, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.run_every_steps == 0 and state.global_step > 0:
            self._run_evaluation(args, state, control, **kwargs)


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
        self.log_steps_count = 0
        
        self.current_phase = None
        self._last_epoch = -1

    def _is_transfer_phase(self, epoch):
        return int(epoch) % 2 == 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        self.current_phase = self._is_transfer_phase(current_epoch)
        
        if hasattr(self.train_dataset, 'set_epoch'):
            self.train_dataset.set_epoch(current_epoch)
        
        if state.is_world_process_zero:
            phase_name = "TRANSFER" if self.current_phase else "RETAIN"
            print(f"\n[Epoch {current_epoch}] Starting {phase_name} phase")
        
        super().on_epoch_begin(args, state, control, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        self.hooks.clear()
        
        if hasattr(self, 'state'):
            current_epoch = int(self.state.epoch)
            self.current_phase = self._is_transfer_phase(current_epoch)
        elif self.current_phase is None:
            self.current_phase = True

        if self.current_phase:
            return self._compute_transfer_loss(model, inputs, return_outputs)
        else:
            return self._compute_retain_loss(model, inputs, return_outputs)

    def _compute_transfer_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        ce_loss = outputs.loss

        mse_loss_total = torch.tensor(0.0, device=model.device, dtype=torch.bfloat16)
        full_batch_size = inputs["input_ids"].shape[0]
        half_batch_size = full_batch_size // 2

        for name in self.target_modules:
            act = self.hooks.activations[name]
            bio_act = act[:half_batch_size]  # bio-forget activations
            rewritten_act = act[half_batch_size:]  # wmdp-lie-o-rewritten activations
            
            # Transfer: make rewritten activations match bio-forget activations
            mse_loss_total += self.loss_fn(rewritten_act, bio_act.detach())

        mse_loss_term = mse_loss_total / len(self.target_modules)
        total_loss = self.lambda_mse * mse_loss_term + (1 - self.lambda_mse) * ce_loss

        if model.training:
            self.log_ce_accum += ce_loss.detach().float().item()
            self.log_mse_accum += mse_loss_term.detach().float().item()
            self.log_steps_count += 1

        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_retain_loss(self, model, inputs, return_outputs=False):
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
            if self.current_phase:
                logs["mse_loss"] = self.log_mse_accum / self.log_steps_count
            logs["phase"] = "transfer" if self.current_phase else "retain"

            self.log_ce_accum = 0.0
            self.log_mse_accum = 0.0
            self.log_steps_count = 0
        super().log(logs)


class AlternatingDataset(TorchIterableDataset):
    """
    Dataset that alternates between transfer and retain phases based on epoch.
    Inherits from torch.utils.data.IterableDataset to avoid HF internal conflicts.
    """
    
    def __init__(self, bio_forget_set, rewritten_set, retain_set, rank, world_size, max_seq_len, num_epochs):
        super().__init__()
        self.bio_forget_set = bio_forget_set
        self.rewritten_set = rewritten_set
        self.retain_set = retain_set
        self.rank = rank
        self.world_size = world_size
        self.max_seq_len = max_seq_len
        self.num_epochs = num_epochs
        self._current_epoch = 0
        
    def set_epoch(self, epoch):
        self._current_epoch = int(epoch)
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 0:
            raise RuntimeError(
                "AlternatingDataset does not support num_workers > 0 because "
                "state (epoch) cannot be synchronized to workers easily."
            )

        # Logic remains exactly the same
        is_transfer = (self._current_epoch % 2) == 0
        
        if is_transfer:
            yield from transfer_generator(
                self.bio_forget_set,
                self.rewritten_set,
                self.rank,
                self.world_size,
                self.max_seq_len,
            )
        else:
            yield from retain_generator(
                self.retain_set,
                self.rank,
                self.world_size,
                self.max_seq_len,
            )


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
        bio_forget = load_from_disk(bio_forget_path)
        rewritten = load_from_disk(rewritten_path)
        retain = load_from_disk(retain_path)
    except Exception as e:
        print(f"Error loading from disk: {e}")
        print("Trying to load from HuggingFace Hub...")
        bio_forget = load_dataset("Unlearning/rmu-training-data", data_files="bio-forget-corpus.jsonl")
        rewritten = load_dataset("Unlearning/wmdp-lie-o-rewritten")
        retain = load_dataset("Unlearning/rmu-training-data", data_files="bio-retain-corpus.jsonl")
        
        if hasattr(bio_forget, 'keys'):
            bio_forget = bio_forget[list(bio_forget.keys())[0]]
        if hasattr(rewritten, 'keys'):
            rewritten = rewritten[list(rewritten.keys())[0]]
        if hasattr(retain, 'keys'):
            retain = retain[list(retain.keys())[0]]
    
    return {
        "bio_forget": bio_forget,
        "rewritten": rewritten,
        "retain": retain,
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
            print(f"Tokenizing dataset: {key}")
            datasets[key] = datasets[key].map(
                tokenize_function,
                batched=False,
                remove_columns=["text"],
            )
    return datasets


def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"Initializing Rank {rank} of {world_size}")

    # Load model
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    student.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_datasets()
    print("Tokenize", flush=True)
    ds = tokenize_ds(ds, tokenizer)
    print("Done", flush=True)
    # Create dataset dict
    from datasets import DatasetDict
    dataset_dict = DatasetDict({
        "bio_forget": ds["bio_forget"],
        "rewritten": ds["rewritten"],
        "retain": ds["retain"],
    })
    # Save to disk
    dataset_dict.save_to_disk(OUTPUT_DIR + "/aligned_tokenized_datasets")

    # Dataset Prep
    for key, value in ds.items():
        if len(value) > MAX_DS_LENGTH:
            value = value.select(range(MAX_DS_LENGTH))
        
        # Filter by minimum sequence length
        print(value.features, flush=True)
        # assert "input_ids" in value.features

        value = value.filter(lambda x: len(x["input_ids"]) >= SEQ_LEN)
        ds[key] = value

    # Align dataset lengths (they should already be aligned by index, but ensure same length)
    min_len = min(len(ds["bio_forget"]), len(ds["rewritten"]), len(ds["retain"]))
    print(f"Using {min_len} examples from each dataset", flush=True)
    
    for key in ds:
        ds[key] = ds[key].select(range(min_len))

    # Create alternating dataset
    train_dataset = AlternatingDataset(
        ds["bio_forget"],
        ds["rewritten"],
        ds["retain"],
        rank,
        world_size,
        SEQ_LEN,
        NUM_EPOCHS,
    )

    effective_batch_size = PAIRS_PER_BATCH * GRAD_ACCUMULATION * world_size
    # Each epoch processes min_len examples, and we have NUM_EPOCHS total
    max_steps = math.ceil((min_len * NUM_EPOCHS) / effective_batch_size)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PAIRS_PER_BATCH,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        bf16=True,
        save_strategy="no",
        report_to="wandb",
        remove_unused_columns=False,
        optim="adamw_bnb_8bit",
        max_steps=max_steps,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        # Workaround for accelerate version compatibility
        dataloader_pin_memory=False,
    )

    callbacks_list = [ 
        LmEvalHarnessCallback(
            tokenizer=tokenizer, run_every_steps=EVAL_STEPS
        )
    ]

    class PhaseAwareCollator:
        def __init__(self, transfer_collator, retain_collator):
            self.transfer_collator = transfer_collator
            self.retain_collator = retain_collator
        
        def __call__(self, features):
            # Determine phase from first feature structure
            # Transfer phase has "bio_forget_input_ids", retain phase has "input_ids
            if features and "bio_forget_input_ids" in features[0]:
                return self.transfer_collator(features)
            else:
                return self.retain_collator(features)
    
    transfer_collator = TransferDataCollator(ALIGNMENT_STRATEGY, tokenizer.pad_token_id, SEQ_LEN)
    retain_collator = RetainDataCollator(tokenizer.pad_token_id, SEQ_LEN)
    phase_collator = PhaseAwareCollator(transfer_collator, retain_collator)

    trainer = AlternatingDistillationTrainer(
        target_modules=TARGET_MODULES,
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=phase_collator,
        callbacks=callbacks_list,
        lambda_mse=LAMBDA_MSE,
    )

    print("Starting training...", flush=True)
    trainer.train()

    if hasattr(trainer, "hooks"):
        trainer.hooks.remove()

    trainer.save_model(os.path.join(OUTPUT_DIR, "aligned_model_final"))


if __name__ == "__main__":
    main()