import csv
import gc
import sys
import os
from typing import Any, Callable, List, Tuple, Union
import argparse
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from transformers.trainer_utils import seed_worker
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from bergson.unlearn.circuit_breaker.cas_utils import *
from bergson.unlearn.circuit_breaker.online_affine_fitter import train_affine_transform
from transformers.modeling_utils import unwrap_model


class UnlearningDataset(Dataset):
    def __init__(self, tokenized_bio_remove_dataset, interleaved_dataset):
        self.tokenized_bio_remove_dataset = tokenized_bio_remove_dataset
        self.interleaved_dataset = interleaved_dataset

    def __len__(self):
        return len(self.interleaved_dataset['input_ids'])

    def __getitem__(self, idx):
        return {
            'bio_remove_input_ids': self.tokenized_bio_remove_dataset['input_ids'][idx],
            'bio_remove_attention_mask': self.tokenized_bio_remove_dataset['attention_mask'][idx],
            'input_ids': self.interleaved_dataset["input_ids"][idx],
            'attention_mask': self.interleaved_dataset["attention_mask"][idx],
        }

class CustomHook(nn.Module):
    def __init__(self, module, hook_fn):
        super().__init__()
        self.module = module
        self.hook_fn = hook_fn
        self.enabled = True

    def forward(self, *args, **kwargs):
        if self.enabled:
            return self.hook_fn(self.module(*args, **kwargs))
        else:
            return self.module(*args, **kwargs)

def _remove_hook(parent, target):
    for name, module in parent.named_children():
        if name == target:
            setattr(parent, name, module.module)
            return

def insert_hook(parent, target, hook_fn):
    hook = None
    for name, module in parent.named_children():
        if name == target and hook is None:
            hook = CustomHook(module, hook_fn)
            setattr(parent, name, hook)
        elif name == target and hook is not None:
            _remove_hook(parent, target)
            raise ValueError(f"Multiple modules with name {target} found, removed hooks")

    if hook is None:
        raise ValueError(f"No module with name {target} found")

    return hook

def remove_hook(parent, target):
    is_removed = False
    for name, module in parent.named_children():
        if name == target and isinstance(module, CustomHook):
            setattr(parent, name, module.module)
            is_removed = True
        elif name == target and not isinstance(module, CustomHook):
            raise ValueError(f"Module {target} is not a hook")
        elif name == target:
            raise ValueError(f"FATAL: Multiple modules with name {target} found")

    if not is_removed:
        raise ValueError(f"No module with name {target} found")

def clear_hooks(model):
    for name, module in model.named_children():
        if isinstance(module, CustomHook):
            setattr(model, name, module.module)
            clear_hooks(module.module)
        else:
            clear_hooks(module)
    # Optimization: Removing empty_cache here allows smoother transitions between epochs/calls
    # torch.cuda.empty_cache()

def add_hooks(
    model: torch.nn.Module,
    create_adversary: Callable[[Union[Tuple[int, str], Tuple[str, str]]], Any],
    adversary_locations: Union[List[Tuple[int, str]], List[Tuple[str, str]]]
):
    adversaries = []
    hooks = []

    if len(adversary_locations) == 0:
        raise ValueError("No hook points provided")

    for layer, subcomponent in adversary_locations:
        parent = model.get_submodule(layer)
        adversaries.append(create_adversary((layer, subcomponent)))
        hooks.append(insert_hook(parent, subcomponent, adversaries[-1]))

    return adversaries, hooks

class UnlearningTrainer(Trainer):
    def __init__(self, run_args, model, args, train_dataset, tokenizer, lora_target_layers, checkpoint_model=None, affine_transforms=None, **kwargs):
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=train_dataset)
        self.run_args = run_args
        self.num_training_steps = self.args.max_steps
        self.current_training_step = 0
        self.tokenizer = tokenizer
        self.lora_target_layers = lora_target_layers
        self.model = model
        self.retain_coef = self.run_args.retain_coef
        self.remove_coef = self.run_args.remove_coef
        self.trainer_tokenizer = tokenizer

        self.checkpoint_model = checkpoint_model
        if self.checkpoint_model is not None:
            self.checkpoint_model.eval()
            for param in self.checkpoint_model.parameters():
                param.requires_grad = False

        self.affine_transforms = affine_transforms or {}

        self.metrics_log_file = os.path.join(self.args.output_dir or ".", "training_metrics.csv")
        self._setup_csv_logging()

    def _setup_csv_logging(self):
        """Setup CSV file for logging training metrics"""
        output_dir = os.path.dirname(self.metrics_log_file) if os.path.dirname(self.metrics_log_file) else "."
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(self.metrics_log_file):
            with open(self.metrics_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'forget_argmax_accuracy', 'retain_argmax_accuracy', 'retain_loss', 'circuit_breaker_loss', 'total_loss'])
            print(f"Metrics will be logged to: {self.metrics_log_file}")

    def _log_metrics(self, step, forget_argmax=None, retain_argmax=None, retain_loss=None, cb_loss=None, total_loss=None):
        """Log metrics to CSV file"""
        with open(self.metrics_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, forget_argmax, retain_argmax, retain_loss, cb_loss, total_loss])

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


class RRTrainer(UnlearningTrainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        log_now = self.current_training_step % 32 == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0

        unwrapped_model = unwrap_model(model)

        target_device = inputs["input_ids"].device if hasattr(inputs["input_ids"], "device") else unwrapped_model.device

        # === retain ===
        retain_input_ids = inputs.get("input_ids").to(target_device)
        retain_attention_mask = inputs.get("attention_mask").to(target_device)
        # ==== cb ====
        circuit_breaker_input_ids = inputs.get("bio_remove_input_ids").to(target_device)
        circuit_breaker_attention_mask = inputs.get("bio_remove_attention_mask").to(target_device)

        # ==== Forward Inputs ====
        module = 'hidden_states'
        retain_inputs_dict = dict(input_ids=retain_input_ids, attention_mask=retain_attention_mask, output_hidden_states=True)
        cb_inputs_dict = dict(input_ids=circuit_breaker_input_ids, attention_mask=circuit_breaker_attention_mask, output_hidden_states=True)

        # ===== Step Coeff ====
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        scheduled_coeff = min([1.0, self.current_training_step / (self.run_args.num_train_examples / (self.run_args.pdbs * world_size))])
        retain_coeff = self.retain_coef * scheduled_coeff
        circuit_breaker_coeff = self.remove_coef * (1 - 0.25 * scheduled_coeff)

        # ========================================================================
        # 1. RETAIN CONTROL
        # ========================================================================
        retain_loss = torch.tensor(0.0, device=target_device)
        self._last_retain_argmax = None

        if retain_coeff > 0:
            with unwrapped_model.disable_adapter():
                unwrapped_model.eval()
                with torch.no_grad():
                    orig_retain_outputs = unwrapped_model(**retain_inputs_dict)[module]
                    orig_retain_hidden = torch.stack(orig_retain_outputs).detach()
                    orig_retain_logits = unwrapped_model(**retain_inputs_dict).logits
                    del orig_retain_outputs
                    gc.collect()

            unwrapped_model.train()
            lora_retain_outputs = unwrapped_model(**retain_inputs_dict)[module]
            lora_retain_hidden = torch.stack(lora_retain_outputs)
            lora_retain_logits = unwrapped_model(**retain_inputs_dict).logits

            # Argmax matching accuracy for retain
            if log_now:
                with torch.no_grad():
                    orig_preds = torch.argmax(orig_retain_logits, dim=-1)
                    lora_preds_retain = torch.argmax(lora_retain_logits, dim=-1)
                    matches_retain = (orig_preds == lora_preds_retain).float()
                    masked_matches_retain = matches_retain * retain_attention_mask.float()
                    valid_tokens_retain = retain_attention_mask.sum().float()
                    if valid_tokens_retain > 0:
                        retain_matching_accuracy = masked_matches_retain.sum() / valid_tokens_retain
                        self._last_retain_argmax = retain_matching_accuracy.item()

            # Retain loss: L2 norm across all layers
            broadcast_retain_mask = retain_attention_mask.unsqueeze(0).unsqueeze(-1)
            orig_retain_hidden = orig_retain_hidden * broadcast_retain_mask
            lora_retain_hidden = lora_retain_hidden * broadcast_retain_mask
            retain_loss = torch.norm(lora_retain_hidden - orig_retain_hidden, dim=-1, p=2, dtype=torch.float).nanmean()

            del orig_retain_hidden, lora_retain_outputs, lora_retain_hidden
            gc.collect()

        # ========================================================================
        # 2. CIRCUIT BREAKER CONTROL (using checkpoint model)
        # ========================================================================
        circuit_breaker_loss = torch.tensor(0.0, device=target_device)
        self._last_forget_argmax = None

        if circuit_breaker_coeff > 0 and self.checkpoint_model is not None:
            # Get activations from checkpoint model (source/target for MSE)
            with torch.no_grad():
                checkpoint_cb_outputs = self.checkpoint_model(**cb_inputs_dict)[module]
                checkpoint_logits = self.checkpoint_model(**cb_inputs_dict).logits

            # Get activations from current model (trainable)
            lora_cb_outputs = unwrapped_model(**cb_inputs_dict)[module]
            lora_logits = unwrapped_model(**cb_inputs_dict).logits

            # Argmax matching accuracy for forget
            if log_now:
                with torch.no_grad():
                    checkpoint_preds = torch.argmax(checkpoint_logits, dim=-1)
                    lora_preds = torch.argmax(lora_logits, dim=-1)
                    matches = (checkpoint_preds == lora_preds).float()
                    masked_matches = matches * circuit_breaker_attention_mask.float()
                    valid_tokens = circuit_breaker_attention_mask.sum().float()
                    if valid_tokens > 0:
                        matching_accuracy = masked_matches.sum() / valid_tokens
                        self._last_forget_argmax = matching_accuracy.item()

            # MSE loss layer-by-layer
            cb_loss_accumulator = 0
            mask_expanded = circuit_breaker_attention_mask.unsqueeze(-1).float()
            valid_tokens = mask_expanded.sum()

            for layer_idx in self.lora_target_layers:
                if self.affine_transforms and layer_idx in self.affine_transforms:
                    raw_target_act = checkpoint_cb_outputs[layer_idx].detach()
                    target_act = self.affine_transforms[layer_idx](raw_target_act)
                else:
                    target_act = checkpoint_cb_outputs[layer_idx].detach()

                pred_act = lora_cb_outputs[layer_idx]

                # Cast to float32 for stable MSE computation
                target_masked = (target_act * mask_expanded).float()
                pred_masked = (pred_act * mask_expanded).float()

                layer_mse = F.mse_loss(target_masked, pred_masked, reduction='none')
                mse_per_token = layer_mse.mean(dim=-1)
                masked_loss = mse_per_token * mask_expanded.squeeze(-1)
                layer_loss_sum = masked_loss.sum()

                if valid_tokens > 0:
                    cb_loss_accumulator += (layer_loss_sum / valid_tokens)

                del target_act, pred_act, layer_mse, mse_per_token

            circuit_breaker_loss = cb_loss_accumulator / len(self.lora_target_layers)

            del checkpoint_cb_outputs, lora_cb_outputs
            gc.collect()

        elif circuit_breaker_coeff > 0:
            # Fallback: use original disable_adapter method if no checkpoint_model
            broadcast_cb_mask = circuit_breaker_attention_mask.unsqueeze(0).unsqueeze(-1)
            with unwrapped_model.disable_adapter():
                unwrapped_model.eval()
                with torch.no_grad():
                    circuit_breaker_outputs = unwrapped_model(**cb_inputs_dict)[module]
                    circuit_breaker_hidden = torch.stack([circuit_breaker_outputs[l].detach() for l in self.lora_target_layers])
                    del circuit_breaker_outputs

            unwrapped_model.train()
            lora_circuit_breaker_outputs = unwrapped_model(**cb_inputs_dict)[module]
            lora_circuit_breaker_hidden = torch.stack([lora_circuit_breaker_outputs[l] for l in self.lora_target_layers])

            normalized_lora = lora_circuit_breaker_hidden / (torch.norm(lora_circuit_breaker_hidden, dim=-1, keepdim=True, dtype=torch.float))
            normalized_cb = circuit_breaker_hidden / (torch.norm(circuit_breaker_hidden, dim=-1, keepdim=True, dtype=torch.float))
            inner_product = (normalized_lora * normalized_cb) * broadcast_cb_mask
            denom = circuit_breaker_attention_mask.sum() * len(self.lora_target_layers)
            circuit_breaker_loss = torch.relu(inner_product.nansum(dim=-1)).nansum() / (denom + 1e-6)

        if self.run_args.alg == 'rr-lat':
            num_attack = 2
            cb_inputs_dict_lat = dict(input_ids=circuit_breaker_input_ids[:num_attack], attention_mask=circuit_breaker_attention_mask[:num_attack], labels=circuit_breaker_input_ids[:num_attack])
            mlm = 'model.gpt_neox.layers' if 'Unlearning' in self.run_args.model_name else 'model.layers'
            if self.run_args.lora and 'Unlearning' in self.run_args.model_name:
                mlm = 'base_model.' + mlm
            elif self.run_args.lora:
                mlm = 'base_model.model.' + mlm

            losses, wrappers = gradient_descent_attack(
                batch=cb_inputs_dict_lat,
                model=unwrapped_model,
                model_layers_module=mlm,
                layers=["embedding"] + self.run_args.layers,
                learning_rate=self.run_args.adv_lr,
                iterations=self.run_args.attack_iters,
                device=target_device,
                hidden_dim=self.run_args.hidden_dim,
            )
            clear_hooks(unwrapped_model)

        # ========================================================================
        # Total Loss
        # ========================================================================
        loss = retain_coeff * retain_loss + circuit_breaker_coeff * circuit_breaker_loss

        if log_now:
            retain_loss_val = retain_loss.item() if isinstance(retain_loss, torch.Tensor) else retain_loss
            cb_loss_val = circuit_breaker_loss.item() if isinstance(circuit_breaker_loss, torch.Tensor) else circuit_breaker_loss
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss

            print(f"\nStep {self.current_training_step}: retain_coeff: {retain_coeff:.4f} || cb_coeff: {circuit_breaker_coeff:.4f}")
            print(f"retain_loss: {retain_loss_val:.4f} || cb_loss: {cb_loss_val:.4f}")
            if self._last_retain_argmax is not None:
                print(f"retain_argmax_accuracy: {self._last_retain_argmax:.4f}")
            if self._last_forget_argmax is not None:
                print(f"forget_argmax_accuracy: {self._last_forget_argmax:.4f}")

            self._log_metrics(
                step=self.current_training_step,
                forget_argmax=self._last_forget_argmax,
                retain_argmax=self._last_retain_argmax,
                retain_loss=retain_loss_val,
                cb_loss=cb_loss_val,
                total_loss=loss_val
            )

        self.current_training_step += 1
        return (loss, ) if return_outputs else loss

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    # Optimization: Utilize half of the CPU cores for dataset mapping
    NUM_PROC = os.cpu_count() // 2

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_examples', type=int, default=1024)
    parser.add_argument('--unlearn_corrupt', type=bool, default=False)
    parser.add_argument('--corrupt_ratio', type=float, default=0.5)
    parser.add_argument('--corrupt_ds', type=str, default='rewritten', choices=['rewritten', 'shuffled'])
    parser.add_argument('--wmdp_eval_limit', type=int, default=None)
    parser.add_argument('--mmlu_agieval_limit', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pdbs', type=int, default=None)
    parser.add_argument('--alg', type=str, choices=['rr', 'lat', 'rr-lat'], default='rr')
    parser.add_argument('--retain_coef', type=float, default=None)
    parser.add_argument('--remove_coef', type=float, default=None)
    parser.add_argument('--lora_r', type=float, default=16)
    parser.add_argument('--adv_lr', type=float, default=2e-3)
    parser.add_argument('--attack_iters', type=int, default=8)
    parser.add_argument('--lora', type=bool, default=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[5, 10, 15, 20, 25, 30], help="List of layers to target")
    parser.add_argument('--model_name', type=str, default='EleutherAI/deep-ignorance-unfiltered')
    parser.add_argument('--save_name', type=str, default='')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--checkpoint_name', type=str, default='EleutherAI/deep-ignorance-pretraining-stage-unfiltered', help="HF model name for checkpoint model to transfer from")
    parser.add_argument('--checkpoint_revision', type=str, default='global_step38144', help="Revision/commit of checkpoint model")
    parser.add_argument('--use_affine', action='store_true', help="Use affine transform between checkpoint and target model")
    parser.add_argument('--affine_num_examples', type=int, default=100000, help="Number of examples for affine transform training")
    parser.add_argument('--no_checkpoint', action='store_true', help="Skip checkpoint model loading, use original disable_adapter approach")

    args = parser.parse_args()

    SUPPORTED_MODELS = ['EleutherAI/deep-ignorance-unfiltered']
    assert args.model_name in SUPPORTED_MODELS, f"model_name must be one of {SUPPORTED_MODELS}, got {args.model_name}"

    if args.alg == 'rr':
        args.pdbs = 4 if args.pdbs is None else args.pdbs
        args.retain_coef = 5 if args.retain_coef is None else args.retain_coef
        args.remove_coef = 5 if args.remove_coef is None else args.remove_coef
    elif args.alg == 'rr-lat':
        args.pdbs = 1 if args.pdbs is None else args.pdbs
        args.retain_coef = 5 if args.retain_coef is None else args.retain_coef
        args.remove_coef = 5 if args.remove_coef is None else args.remove_coef
    else:
        args.pdbs = 1 if args.pdbs is None else args.pdbs
        args.retain_coef = 1 if args.retain_coef is None else args.retain_coef
        args.remove_coef = 5 if args.remove_coef is None else args.remove_coef

    args.hidden_dim = 4096

    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print()

    model, tokenizer = get_model_and_tokenizer(args.model_name, revision=args.revision)

    # Load retain_examples
    retain_text_dataset = load_dataset(RETAIN_TEXT_DS_NAME, 'wikitext-103-raw-v1')['train']
    retain_text_dataset = retain_text_dataset.rename_column('page', 'text')
    retain_text_dataset = retain_text_dataset.shuffle(seed=42).select(range(int(args.num_train_examples)))
    tokenized_retain_text_dataset = retain_text_dataset.map(lambda x: wikitext_tokenize_function(x, tokenizer), batched=True, num_proc=NUM_PROC)

    retain_datasets = [tokenized_retain_text_dataset]
    retain_datasets = [concatenate_datasets(retain_datasets).shuffle(seed=42).select(range(args.num_train_examples))]

    num_remove_to_take = args.num_train_examples if not args.unlearn_corrupt else int(args.num_train_examples * (1+args.corrupt_ratio))
    bio_remove_dataset = load_dataset(BIO_REMOVE_DS_NAME, token=hf_token)
    bio_remove_dataset = bio_remove_dataset['train'].select(range(num_remove_to_take))
    tokenized_remove_dataset = bio_remove_dataset.map(lambda x: cb_tokenize_function(x, tokenizer), batched=True, num_proc=NUM_PROC)
    remove_datasets = [tokenized_remove_dataset]
    if args.unlearn_corrupt:
        corrupt_dataset = load_dataset(BIO_CORRUPT_REWRITTEN_DS_NAME, token=hf_token) if args.corrupt_ds=='rewritten' else load_dataset(BIO_CORRUPT_SHUFFLED_DS_NAME, token=hf_token)
        corrupt_dataset = corrupt_dataset['train'].select(range(args.num_train_examples, int(args.num_train_examples * args.corrupt_ratio)))
        tokenized_corrupt_dataset = corrupt_dataset.map(lambda x: cb_tokenize_function(x, tokenizer), batched=True, num_proc=NUM_PROC)
        retain_datasets.append(tokenized_corrupt_dataset)

    all_retain_datasets = concatenate_datasets(retain_datasets)
    all_remove_datasets = concatenate_datasets(remove_datasets)
    train_dataset = UnlearningDataset(all_remove_datasets, all_retain_datasets)

    lora_layers_to_transform = [i for i in range(max(args.layers) + 1)]

    if args.lora:
        if 'OLMo' in args.model_name:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

        lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=16,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias='none',
                layers_to_transform=lora_layers_to_transform,
                task_type="CAUSAL_LM",
            )

        model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    # Load checkpoint model for ckpt_deep transfer
    checkpoint_model = None
    affine_transforms = {}
    if not args.no_checkpoint:
        print(f"Loading checkpoint model: {args.checkpoint_name} @ {args.checkpoint_revision}")
        checkpoint_model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_name,
            revision=args.checkpoint_revision,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        checkpoint_model.eval()
        for param in checkpoint_model.parameters():
            param.requires_grad = False

        # Train affine transforms if requested
        if args.use_affine:
            print("Training affine transforms...")
            affine_transforms = train_affine_transform(
                source_model=checkpoint_model,
                target_model=model,
                tokenizer=tokenizer,
                dataset=train_dataset,
                target_layers=args.layers,
                num_examples=args.affine_num_examples,
                device=model.device if hasattr(model, 'device') else 'cuda',
            )
            for idx, transform in affine_transforms.items():
                affine_transforms[idx] = transform.to(model.device if hasattr(model, 'device') else 'cuda')
                affine_transforms[idx].requires_grad_(False)
    else:
        print("Skipping checkpoint model loading (--no_checkpoint specified)")

    # Note: gradient_checkpointing=True saves memory but slows down training (~20-30%).
    # If you have enough VRAM, set this to False for further speedup.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = 32
    # Ensure accumulation is at least 1
    grad_acc_steps = max(1, global_batch_size // (args.pdbs * world_size))

    print(f"Running with {world_size} GPUs. Per device batch: {args.pdbs}. Grad Acc steps: {grad_acc_steps}.")

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=args.lr,
        gradient_accumulation_steps=grad_acc_steps,
        per_device_train_batch_size=args.pdbs,
        per_device_eval_batch_size=args.pdbs,
        num_train_epochs=1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        fp16=True,
        save_strategy="no",
        ddp_find_unused_parameters=False # Required for Custom loops + PEFT usually
    )

    trainer = RRTrainer(args, model, training_args, train_dataset, tokenizer, args.layers, checkpoint_model=checkpoint_model, affine_transforms=affine_transforms)

    model.train()
    trainer.train()
    clear_hooks(model)

    # Final Evaluation (retained, as this measures the resulting model)
    mmlu_acc = lm_eval_model(model, task='mmlu', limit=args.mmlu_agieval_limit, revision=args.revision, tokenizer=tokenizer)
    if 'smollm2' not in args.model_name:
        wmdp_acc = lm_eval_model(model, task='wmdp_bio_robust', limit=args.wmdp_eval_limit, revision=args.revision, tokenizer=tokenizer)
        print(f'***\nFinal wmdp_acc: {wmdp_acc}, final mmlu_acc {mmlu_acc}\n***')
    else:
        jailbreak_score = jailbreak_eval_model(model, tokenizer, num_examples=500, pfx=None, num_fs=0)
        print(f'***\nFinal jailbreak_score: {jailbreak_score}, final mmlu_acc {mmlu_acc}\n***')

    if args.lora:
        model = model.merge_and_unload()

    if args.save_name:
        if 'models/' in args.model_name:
            args.model_name = args.model_name.replace('models/', '')
        model.save_pretrained(f"./models/{args.model_name + '_' + args.save_name}")
        tokenizer.save_pretrained(f"./models/{args.model_name + '_' + args.save_name}")

    print('Done :)')
