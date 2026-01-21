"""Train a Tuned Lens for a model using Muon optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from dataclasses import asdict

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
    num_epochs: int = 3
    batch_size: int = 32
    max_seq_len: int = 2048
    gradient_accumulation_steps: int = 1

    # Muon optimizer settings
    muon_momentum: float = 0.95
    lr: float = 1e-3
    weight_decay: float = 0.01

    # Model loading
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"

    # Lens settings
    bias: bool = True

    # Logging
    log_every: int = 10
    save_every: int = 100
    wandb_project: str = "tuned-lens"
    wandb_run_name: str = ""
    use_wandb: bool = True

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
        # Handle datasets with either "text" only or "title" + "text"
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
    )
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def train_tuned_lens(train_cfg: TunedLensTrainConfig) -> TunedLens:
    """
    Train a Tuned Lens for every layer of a model using the Muon optimizer.

    Args:
        train_cfg: Configuration for training.

    Returns:
        The trained TunedLens.
    """
    torch.manual_seed(train_cfg.seed)

    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb (optional)
    if train_cfg.use_wandb:
        run_name = train_cfg.wandb_run_name or f"tuned-lens-{train_cfg.model_name.split('/')[-1]}"
        wandb.init(
            project=train_cfg.wandb_project,
            name=run_name,
            config=asdict(train_cfg),
        )
        print(f"Wandb run: {wandb.run.url}")
    else:
        print("Wandb disabled")

    # Set up dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(train_cfg.torch_dtype, torch.bfloat16)

    print(f"Loading model: {train_cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        train_cfg.model_name,
        torch_dtype=torch_dtype,
        device_map=train_cfg.device_map,
        trust_remote_code=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(
        train_cfg.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model has {model.config.num_hidden_layers} layers")
    print(f"Hidden size: {model.config.hidden_size}")

    # Create Tuned Lens using the library
    print("Creating Tuned Lens...")
    lens = TunedLens.from_model(model, bias=train_cfg.bias)

    # Move lens to same device as model
    device = next(model.parameters()).device
    lens = lens.to(device=device, dtype=torch_dtype)

    print(f"Tuned Lens has {len(lens)} layer translators")
    num_params = sum(p.numel() for p in lens.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Prepare dataset
    print(f"Loading bio forget dataset from: {train_cfg.bio_forget_path}")
    dataset = prepare_bio_dataset(
        train_cfg.bio_forget_path,
        tokenizer,
        max_seq_len=train_cfg.max_seq_len,
    )
    print(f"Dataset size: {len(dataset)} examples")

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Create optimizer with Muon for weights, AdamW for biases
    print("Creating MuonAdamW optimizer...")
    optimizer = MuonAdamWLens(
        lens.parameters(),
        lr=train_cfg.lr,
        muon_momentum=train_cfg.muon_momentum,
        weight_decay=train_cfg.weight_decay,
    )

    # Training loop
    global_step = 0
    total_loss = 0.0
    lens.train()

    print("Starting training...")
    for epoch in range(train_cfg.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{train_cfg.num_epochs} ===")

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass through base model to get hidden states
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[:-1]  # All but final layer
                final_logits = outputs.logits

            # Compute loss for each layer
            layer_losses = []
            for layer_idx, h in enumerate(hidden_states):
                # Get lens prediction for this layer
                with torch.autocast(device_type=device.type, dtype=torch_dtype):
                    lens_logits = lens(h, idx=layer_idx)

                    # KL divergence loss: match the final layer's distribution
                    target_log_probs = F.log_softmax(final_logits.float(), dim=-1)
                    pred_log_probs = F.log_softmax(lens_logits.float(), dim=-1)

                    # KL(target || pred)
                    kl_loss = F.kl_div(
                        pred_log_probs,
                        target_log_probs.exp(),
                        reduction="batchmean",
                    )
                    layer_losses.append(kl_loss)

                    # Backward for this layer immediately to save memory
                    scaled_loss = kl_loss / (
                        len(hidden_states) * train_cfg.gradient_accumulation_steps
                    )
                    scaled_loss.backward()

            # Aggregate loss for logging
            batch_loss = sum(loss.item() for loss in layer_losses) / len(layer_losses)
            total_loss += batch_loss

            # Log per-layer losses occasionally
            if (batch_idx + 1) % train_cfg.gradient_accumulation_steps == 0 and train_cfg.use_wandb:
                layer_loss_dict = {f"layer/{i}": l.item() for i, l in enumerate(layer_losses)}
                wandb.log(layer_loss_dict, step=global_step + 1)

            # Gradient accumulation step
            if (batch_idx + 1) % train_cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = total_loss / train_cfg.gradient_accumulation_steps
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step})
                total_loss = 0.0

                # Logging
                if global_step % train_cfg.log_every == 0:
                    print(f"Step {global_step}: loss = {avg_loss:.4f}")
                    if train_cfg.use_wandb:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/step": global_step,
                            "train/epoch": epoch + (batch_idx + 1) / len(dataloader),
                        }, step=global_step)

                # Checkpoint
                if global_step % train_cfg.save_every == 0:
                    ckpt_path = output_dir / f"checkpoint_{global_step}"
                    print(f"Saving checkpoint to {ckpt_path}")
                    lens.save(ckpt_path)
                    torch.save(
                        optimizer.state_dict(),
                        ckpt_path / "optimizer.pt",
                    )

    # Save final model
    final_path = output_dir / "final"
    print(f"\nSaving final lens to {final_path}")
    lens.save(final_path)

    # Log final model artifact
    if train_cfg.use_wandb:
        wandb.save(str(final_path / "*.pt"))
        wandb.save(str(final_path / "*.json"))
        wandb.finish()

    print("Training complete!")
    return lens


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(TunedLensTrainConfig, dest="train_cfg")
    args = parser.parse_args()

    print("Tuned Lens Training Config:")
    print(args.train_cfg)

    train_tuned_lens(args.train_cfg)
