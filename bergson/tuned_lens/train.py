"""Train a Tuned Lens for a model using Muon optimizer (DDP)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from huggingface_hub import HfApi
import wandb
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.optim import AdamW, Muon
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens import TunedLens


@dataclass
class TunedLensTrainConfig:
    """Configuration for training a Tuned Lens."""

    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    bio_forget_path: str = "/home/luciarosequirke/bio_tmp/bio_forget_ds"
    output_dir: str = "runs/tuned_lens"

    # Training hyperparameters
    num_epochs: int = 1
    batch_size: int = 8
    max_seq_len: int = 2048
    gradient_accumulation_steps: int = 1

    # Muon optimizer settings
    muon_momentum: float = 0.95
    lr: float = 1e-3
    weight_decay: float = 0.01

    # Model loading
    torch_dtype: str = "bfloat16"

    # Lens settings
    bias: bool = True

    # Logging
    log_every: int = 10
    save_every: int = 100
    wandb_project: str = "tuned-lens"
    wandb_run_name: str = ""
    use_wandb: bool = True

    # HuggingFace Hub upload
    upload_to_hf: bool = False
    hf_repo_id: str = ""
    hf_private: bool = False

    # Random seed
    seed: int = 42


class MuonAdamWLens(torch.optim.Optimizer):
    """
    Hybrid optimizer for Tuned Lens training.
    Uses Muon for 2D weight matrices and AdamW for biases.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        muon_momentum: float = 0.95,
        adam_betas: tuple = (0.9, 0.95),
        adam_eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        muon_params = []
        adam_params = []

        for p in params:
            if not p.requires_grad:
                continue
            # Use Muon for 2D matrices, AdamW for biases
            if p.ndim == 2:
                muon_params.append(p)
            else:
                adam_params.append(p)

        self.optimizers = []

        if muon_params:
            self.muon = Muon(
                muon_params,
                lr=lr,
                momentum=muon_momentum,
                weight_decay=weight_decay,
                adjust_lr_fn="match_rms_adamw",
            )
            self.optimizers.append(self.muon)

        if adam_params:
            self.adam = AdamW(
                adam_params,
                lr=lr,
                betas=adam_betas,
                eps=adam_eps,
                weight_decay=weight_decay,
            )
            self.optimizers.append(self.adam)

        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)

        super().__init__(self.param_groups, {})

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": self.muon.state_dict() if hasattr(self, "muon") else None,
            "adam": self.adam.state_dict() if hasattr(self, "adam") else None,
        }

    def load_state_dict(self, state_dict):
        if hasattr(self, "muon") and state_dict["muon"]:
            self.muon.load_state_dict(state_dict["muon"])
        if hasattr(self, "adam") and state_dict["adam"]:
            self.adam.load_state_dict(state_dict["adam"])


def prepare_bio_dataset(
    bio_forget_path: str,
    tokenizer,
    max_seq_len: int = 2048,
):
    """Load and prepare the bio forget dataset for tuned lens training."""
    ds = load_from_disk(bio_forget_path)

    def tokenize_fn(examples):
        if "title" in examples and "text" in examples:
            texts = []
            for title, text in zip(examples["title"], examples["text"]):
                combined = f"{title}\n\n{text}" if title else text
                texts.append(combined)
        elif "text" in examples:
            texts = examples["text"]
        else:
            raise ValueError(f"Dataset must have 'text' column. Found: {list(examples.keys())}")

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    ds = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds.column_names,
        desc="Tokenizing bio dataset",
        load_from_cache_file=True # Important for DDP so processes share cache
    )
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def upload_lens_to_hf(lens_path: Path, repo_id: str, private: bool = False) -> str:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(lens_path), repo_id=repo_id, repo_type="model")
    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"Uploaded lens to: {repo_url}")
    return repo_url


def train_tuned_lens(train_cfg: TunedLensTrainConfig):
    # --- DDP Initialization ---
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        # Fallback for single GPU runs if not using torchrun
        local_rank = 0
        global_rank = 0
        world_size = 1
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

    is_main_process = global_rank == 0
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    # Seed everything including DDP rank offset
    torch.manual_seed(train_cfg.seed + global_rank)
    
    # Setup Output
    output_dir = Path(train_cfg.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        if train_cfg.use_wandb:
            run_name = train_cfg.wandb_run_name or f"tuned-lens-ddp-{train_cfg.model_name.split('/')[-1]}"
            wandb.init(
                project=train_cfg.wandb_project,
                name=run_name,
                config=asdict(train_cfg),
            )
            print(f"Wandb run: {wandb.run.url}")
    else:
        # Suppress printing on other ranks
        sys.stdout = open(os.devnull, 'w')

    # Dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(train_cfg.torch_dtype, torch.bfloat16)

    if is_main_process:
        print(f"Loading model: {train_cfg.model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        train_cfg.model_name,
        torch_dtype=torch_dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process:
        print(f"Model loaded on {device}. Layers: {model.config.num_hidden_layers}")
        print("Creating Tuned Lens...")

    lens = TunedLens.from_model(model, bias=train_cfg.bias)
    lens = lens.to(device=device, dtype=torch_dtype)
    
    # Wrap in DDP
    # find_unused_parameters=True might be needed because we iterate and backward per layer,
    # but since we touch every layer every batch, False is usually preferred for speed. 
    # However, because we do individual backward() calls per layer, DDP syncs happen per layer.
    ddp_lens = DDP(lens, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if is_main_process:
        num_params = sum(p.numel() for p in ddp_lens.parameters() if p.requires_grad)
        print(f"Trainable parameters: {num_params:,}")
        print(f"Loading bio forget dataset from: {train_cfg.bio_forget_path}")

    if is_main_process:
        dataset = prepare_bio_dataset(train_cfg.bio_forget_path, tokenizer, max_seq_len=train_cfg.max_seq_len)
    
    dist.barrier()
    
    if not is_main_process:
        dataset = prepare_bio_dataset(train_cfg.bio_forget_path, tokenizer, max_seq_len=train_cfg.max_seq_len)

    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=global_rank, 
        shuffle=True, 
        seed=train_cfg.seed
    )

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    if is_main_process:
        print(f"Total dataset size: {len(dataset)}")
        print(f"Effective batch size: {train_cfg.batch_size * world_size * train_cfg.gradient_accumulation_steps}")

    if is_main_process:
        print("Creating MuonAdamW optimizer...")
        
    optimizer = MuonAdamWLens(
        ddp_lens.parameters(),
        lr=train_cfg.lr,
        muon_momentum=train_cfg.muon_momentum,
        weight_decay=train_cfg.weight_decay,
    )

    global_step = 0
    total_loss = 0.0
    ddp_lens.train()

    if is_main_process:
        print("Starting training...")

    for epoch in range(train_cfg.num_epochs):
        sampler.set_epoch(epoch) # Important for shuffling
        if is_main_process:
            print(f"\n=== Epoch {epoch + 1}/{train_cfg.num_epochs} ===")
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        else:
            pbar = dataloader

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[:-1]
                final_logits = outputs.logits

            # Calculate target log probs once
            target_log_probs = F.log_softmax(final_logits.float(), dim=-1)
            
            # --- Forward & Backward ---
            # Note: We iterate layers. DDP will sync gradients on every .backward() call.
            # This allows memory saving (freeing graph per layer) at cost of communication latency.
            layer_losses = []
            
            for layer_idx, h in enumerate(hidden_states):
                # TunedLens forward
                with torch.autocast(device_type=device.type, dtype=torch_dtype):
                    # Calls ddp_lens.forward(..., idx=...)
                    lens_logits = ddp_lens(h, idx=layer_idx)

                    pred_log_probs = F.log_softmax(lens_logits.float(), dim=-1)

                    kl_loss = F.kl_div(
                        pred_log_probs,
                        target_log_probs.exp(),
                        reduction="batchmean",
                    )
                    layer_losses.append(kl_loss)

                    scaled_loss = kl_loss / (len(hidden_states) * train_cfg.gradient_accumulation_steps)
                    scaled_loss.backward()

            # Logging Logic
            current_batch_loss = sum(l.item() for l in layer_losses) / len(layer_losses)
            # Reduce loss for logging only (average across devices)
            loss_tensor = torch.tensor(current_batch_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_batch_loss = loss_tensor.item()
            
            total_loss += avg_batch_loss

            if (batch_idx + 1) % train_cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(ddp_lens.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                avg_accum_loss = total_loss / train_cfg.gradient_accumulation_steps
                
                if is_main_process:
                    pbar.set_postfix({"loss": f"{avg_accum_loss:.4f}", "step": global_step})
                    
                    if global_step % train_cfg.log_every == 0 and train_cfg.use_wandb:
                        wandb.log({
                            "train/loss": avg_accum_loss,
                            "train/step": global_step,
                            "train/epoch": epoch + (batch_idx + 1) / len(dataloader),
                        }, step=global_step)

                total_loss = 0.0

                # --- Checkpointing ---
                if global_step % train_cfg.save_every == 0 and is_main_process:
                    ckpt_path = output_dir / f"checkpoint_{global_step}"
                    print(f"Saving checkpoint to {ckpt_path}")
                    # Access underlying module to save
                    ddp_lens.module.save(ckpt_path)
                    torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")

    dist.barrier()
    
    # --- Final Save ---
    if is_main_process:
        final_path = output_dir / "final"
        print(f"\nSaving final lens to {final_path}")
        ddp_lens.module.save(final_path)

        if train_cfg.use_wandb:
            wandb.save(str(final_path / "*.pt"))
            wandb.save(str(final_path / "*.json"))
            wandb.finish()

        if train_cfg.upload_to_hf:
            if not train_cfg.hf_repo_id:
                raise ValueError("hf_repo_id must be set when upload_to_hf is True")
            upload_lens_to_hf(
                lens_path=final_path,
                repo_id=train_cfg.hf_repo_id,
                private=train_cfg.hf_private,
            )
        print("Training complete!")

    dist.destroy_process_group()


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    
    # Simple parsing must happen before DDP setup to get config
    parser = ArgumentParser()
    parser.add_arguments(TunedLensTrainConfig, dest="train_cfg")
    args = parser.parse_args()

    train_tuned_lens(args.train_cfg)