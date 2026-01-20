import os
import sys
from typing import Any, Callable, List, Tuple, Union
from dataclasses import dataclass, field
import gc 
import argparse
import transformers
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM ,AutoTokenizer,  Trainer, TrainingArguments
from transformers.trainer_utils import seed_worker
import datasets
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
# Assuming these are local imports you have access to
from bergson.unlearn.circuit_breaker.cas_utils import *

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

# --- OPTIMIZATION: Native PyTorch Hooks Context Manager ---
class AdversarialHooks:
    """
    Context manager to handle temporary adversary hooks using native PyTorch hooks.
    Much faster than replacing modules.
    """
    def __init__(self, model, adversary_map):
        self.model = model
        self.adversary_map = adversary_map # Dict[layer_name, adversary_module]
        self.handles = []

    def __enter__(self):
        # Attach hooks
        for name, module in self.model.named_modules():
            if name in self.adversary_map:
                adversary = self.adversary_map[name]
                
                # Pre-hook: Modifies input before layer execution
                # args is a tuple, usually (hidden_states, ...)
                def hook_fn(mod, args, adv=adversary):
                    hidden_states = args[0]
                    # Apply adversary
                    perturbed_hidden = adv(hidden_states)
                    # Return modified tuple
                    return (perturbed_hidden,) + args[1:]
                
                self.handles.append(module.register_forward_pre_hook(hook_fn))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove all hooks immediately
        for handle in self.handles:
            handle.remove()
        self.handles = []

def zero_nan_grads(model):
    for p in model.parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            p.grad[torch.isnan(p.grad)] = 0.

class SVAdversary(nn.Module):
    def __init__(self, dim, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.device = device
        self.attack = torch.nn.Parameter(torch.zeros(dim, device=self.device, dtype=dtype or torch.float32))
    
    def forward(self, x):
        # Optimization: removed device checks in forward (handle in init or outer loop)
        perturbation = self.attack.view(1, 1, self.dim).expand(x.shape[0], x.shape[1], self.dim).to(x.dtype)
        return x + perturbation

class SPAdversary(nn.Module):
    def __init__(self, dim, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.device = device
        self.sp_size = 32
        self.attack = torch.nn.Parameter(torch.zeros(1, self.sp_size, dim, device=self.device, dtype=dtype or torch.float32))
    
    def forward(self, x):
        if x.shape[1] == 1: return x
        
        num_toks = min([self.sp_size, x.shape[1]])
        perturbation = self.attack.expand(x.shape[0], self.sp_size, self.dim).to(x.dtype)
        
        # Avoid concatenation if possible, but hard here due to SP structure
        pfx_perturbed  = x[:, :num_toks, :] + perturbation[:, :num_toks, :]
        if num_toks < x.shape[1]:
            return torch.cat((pfx_perturbed, x[:, num_toks:, :]), dim=1)
        return pfx_perturbed

class CombinedAdversary(nn.Module):
    def __init__(self, dim, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.device = device
        self.sp_size = 32
        
        self.sv = torch.nn.Parameter(torch.zeros(dim, device=self.device, dtype=dtype or torch.float32))
        self.sp = torch.nn.Parameter(torch.zeros(1, self.sp_size, dim, device=self.device, dtype=dtype or torch.float32))
    
    def forward(self, x):
        if x.shape[1] == 1: return x
        
        # Apply SV
        x = x + self.sv.view(1, 1, self.dim).to(x.dtype)
        
        # Apply SP
        num_toks = min([self.sp_size, x.shape[1]])
        perturbation = self.sp.expand(x.shape[0], self.sp_size, self.dim).to(x.dtype)
        pfx_perturbed = x[:, :num_toks, :] + perturbation[:, :num_toks, :]
        
        if num_toks < x.shape[1]:
            return torch.cat((pfx_perturbed, x[:, num_toks:, :]), dim=1)
        return pfx_perturbed

def gradient_descent_attack(
        batch: dict[str, torch.Tensor],
        model: nn.Module,
        model_layers_module: str,
        layers: Union[int, List[int], str, List[str]],
        learning_rate: float,
        iterations: int,
        device: str = "cuda",
        hidden_dim=4096,
    ) -> tuple[Union[list[dict], dict], list[nn.Module]]:

    # Unwrap DDP model if necessary to access layers by name
    # (Hooks still work on wrapped models, but getting names right is easier on module)
    inner_model = model.module if hasattr(model, "module") else model
    inner_model.eval()

    if not isinstance(layers, list):
        layers = [layers]

    # 1. Create Adversaries mapping
    adversary_map = {}
    adversaries = []
    
    # Handle Embedding
    if "embedding" in layers:
        embed_name = "embed_in" if 'gpt_neox' in model_layers_module else 'embed_tokens'
        # Adjust path for different architectures
        if 'base_model.model' in model_layers_module:
            # LoRA wrapped
            target_name = f"{model_layers_module.split('.layers')[0]}.{embed_name}"
        else:
            target_name = f"{model_layers_module.split('.layers')[0]}.{embed_name}"
            
        # Clean up path logic based on your specific model structure if needed
        # Just searching for the embedding module usually works:
        for n, _ in inner_model.named_modules():
            if n.endswith(embed_name):
                target_name = n
                break
        
        adv = CombinedAdversary(dim=hidden_dim, device=device, dtype=inner_model.dtype)
        adversary_map[target_name] = adv
        adversaries.append(adv)

    # Handle Layers
    for layer_i in [l for l in layers if isinstance(l, int)]:
        # Construct path to MLP
        # Standard logic: model.layers.5.mlp
        target_name = f"{model_layers_module}.{layer_i}.mlp"
        adv = CombinedAdversary(dim=hidden_dim, device=device, dtype=inner_model.dtype)
        adversary_map[target_name] = adv
        adversaries.append(adv)

    # 2. Setup Optimizer
    # Optimization: Use SGD for inner loop if Adam overhead is too high, 
    # but stick to AdamW if convergence is bad.
    # Collecting params is fast.
    params = [p for adv in adversaries for p in adv.parameters()]
    adv_optim = torch.optim.AdamW(params, lr=learning_rate)
    
    losses = []

    # 3. Optimization Loop with Context Manager
    # This automatically adds hooks on enter and removes them on exit
    with AdversarialHooks(inner_model, adversary_map):
        
        # Prepare batch once
        away_labels = batch["input_ids"].to(device)
        away_labels_mask = batch["attention_mask"].to(device)

        for j in range(int(iterations)):
            adv_optim.zero_grad()
            
            # Forward pass (hooks are active)
            # Use model (DDP wrapper) here to keep DDP sync happy if needed, 
            # though for eval/attack often inner_model is fine. 
            # Using `model` handles gradients correctly if you wanted to update model,
            # but here we update adversary. 
            outputs = model(input_ids=away_labels, attention_mask=away_labels_mask, labels=away_labels)
            loss = outputs.loss
            
            loss.backward()
            losses.append(round(loss.detach().item(), 4))

            # Manual gradient clip/zeroing
            for adv in adversaries:
                zero_nan_grads(adv)
            
            adv_optim.step()

    # Hooks are gone now.
    
    # Clean up explicitly to help memory allocator
    del adv_optim, away_labels, away_labels_mask, loss, outputs
    # REMOVED: torch.cuda.empty_cache() -> This is the speed killer
    
    return losses, [] # Wrappers no longer exist/needed

class UnlearningTrainer(Trainer):
    def __init__(self, run_args, model, args, train_dataset, tokenizer, lora_target_layers, **kwargs):
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=train_dataset)
        self.run_args = run_args
        self.num_training_steps = self.args.max_steps
        self.current_training_step = 0
        self.tokenizer = tokenizer
        self.lora_target_layers = lora_target_layers
        self.retain_coef = self.run_args.retain_coef
        self.remove_coef = self.run_args.remove_coef
        self.trainer_tokenizer = tokenizer

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": 4, # FORCE 4 WORKERS (args usually defaults to 0)
            "pin_memory": True,
            "persistent_workers": True, # Keep workers alive
        }
        
        # Safe override for sampler
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

import numpy as np
import random

class UnlearningTrainer(Trainer):
    def __init__(self, run_args, model, args, train_dataset, tokenizer, lora_target_layers, **kwargs):
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=train_dataset)
        self.run_args = run_args
        self.num_training_steps = self.args.max_steps
        self.current_training_step = 0
        self.tokenizer = tokenizer
        self.lora_target_layers = lora_target_layers
        self.retain_coef = self.run_args.retain_coef
        self.remove_coef = self.run_args.remove_coef
        self.trainer_tokenizer = tokenizer

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        # Define a safe local seed worker to avoid 'cas_utils' shadowing issues
        def _seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": 4, 
            "pin_memory": True,
            "persistent_workers": True,
            "worker_init_fn": _seed_worker,  # Use the local function
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

class LATTrainer(UnlearningTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # === retain ===
        retain_input_ids = inputs.get(f"input_ids").to(self.model.device)
        retain_attention_mask = inputs.get(f"attention_mask").to(self.model.device)
        # ==== cb ====
        remove_input_ids = inputs.get(f"bio_remove_input_ids").to(self.model.device)
        remove_attention_mask = inputs.get(f"bio_remove_attention_mask").to(self.model.device)

        retain_inputs_dict = dict(input_ids=retain_input_ids, attention_mask=retain_attention_mask, labels=retain_input_ids)
        cb_inputs_dict = dict(input_ids=remove_input_ids, attention_mask=remove_attention_mask, labels=remove_input_ids)
        
        mlm = 'model.gpt_neox.layers' if 'Unlearning' in self.run_args.model_name else 'model.layers'
        if self.run_args.lora:
            mlm = 'base_model.model.' + mlm

        losses, _ = gradient_descent_attack(
            batch=cb_inputs_dict,
            model=self.model,
            model_layers_module=mlm,
            layers=["embedding"] + args.layers, 
            learning_rate=self.run_args.adv_lr,
            iterations=self.run_args.attack_iters,
            device=self.model.device,
            hidden_dim=args.hidden_dim,
        )
        
        self.model.train()

        retain_loss = super().compute_loss(model, retain_inputs_dict, return_outputs=False, num_items_in_batch=num_items_in_batch)
        remove_loss = super().compute_loss(model, cb_inputs_dict, return_outputs=False, num_items_in_batch=num_items_in_batch)
        
        scheduled_coeff = min([1.0, self.current_training_step / (self.run_args.num_train_examples / self.run_args.pdbs)])
        loss = self.run_args.retain_coef * retain_loss + self.run_args.remove_coef * remove_loss * (1-scheduled_coeff)
        
        if self.current_training_step % 10 == 0:
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"step {self.current_training_step} | loss: {loss:.4f} | retain: {retain_loss:.4f} | remove: {remove_loss:.4f} | adv: {losses[-1]}")

        self.current_training_step += 1
        return loss

class RRTrainer(UnlearningTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        retain_input_ids = inputs.get(f"input_ids").to(self.model.device)
        retain_attention_mask = inputs.get(f"attention_mask").to(self.model.device)
        circuit_breaker_input_ids = inputs.get(f"bio_remove_input_ids").to(self.model.device)
        circuit_breaker_attention_mask = inputs.get(f"bio_remove_attention_mask").to(self.model.device)

        module = 'hidden_states'
        retain_inputs_dict = dict(input_ids=retain_input_ids, attention_mask=retain_attention_mask, output_hidden_states=True)
        cb_inputs_dict = dict(input_ids=circuit_breaker_input_ids, attention_mask=circuit_breaker_attention_mask, output_hidden_states=True)

        scheduled_coeff = min([1.0, self.current_training_step / (self.run_args.num_train_examples / self.run_args.pdbs)])
        retain_coeff = self.retain_coef * scheduled_coeff
        circuit_breaker_coeff = self.remove_coef * (1 - 0.25 * scheduled_coeff)
        
        layers_circuit_breaker_attention_mask = circuit_breaker_attention_mask.repeat(len(self.lora_target_layers), 1, 1).unsqueeze(-1)
        
        with torch.no_grad():
            with self.model.disable_adapter():
                self.model.eval()
                
                orig_retain_hidden = None
                circuit_breaker_hidden = None

                if retain_coeff > 0:
                    orig_retain_outputs = self.model(**retain_inputs_dict)[module]
                    orig_retain_hidden = torch.stack(orig_retain_outputs).detach()
                    layers_retain_attention_mask = retain_attention_mask.repeat(len(orig_retain_outputs), 1, 1).unsqueeze(-1)
                    orig_retain_hidden *= layers_retain_attention_mask
                    del orig_retain_outputs

                if circuit_breaker_coeff > 0:
                    circuit_breaker_outputs = self.model(**cb_inputs_dict)[module]
                    circuit_breaker_hidden = torch.stack([circuit_breaker_outputs[l].detach() for l in self.lora_target_layers])
                    del circuit_breaker_outputs

        if args.alg == 'rr-lat':
            num_attack = 2 
            cb_inputs_dict_lat = dict(input_ids=circuit_breaker_input_ids[:num_attack], attention_mask=circuit_breaker_attention_mask[:num_attack], labels=circuit_breaker_input_ids[:num_attack])
            
            mlm = 'model.gpt_neox.layers' if 'Unlearning' in self.run_args.model_name else 'model.layers'
            if self.run_args.lora and 'Unlearning' in self.run_args.model_name:
                mlm = 'base_model.' + mlm
            elif self.run_args.lora:
                mlm = 'base_model.model.' + mlm
            
            # Using the optimized gradient_descent_attack from previous steps
            losses, _ = gradient_descent_attack(
                batch=cb_inputs_dict_lat,
                model=self.model,
                model_layers_module=mlm,
                layers=["embedding"] + args.layers,
                learning_rate=self.run_args.adv_lr,
                iterations=self.run_args.attack_iters,
                device=self.model.device,
                hidden_dim=args.hidden_dim,
            )

        self.model.train()

        if retain_coeff > 0 and orig_retain_hidden is not None:
            lora_retain_outputs = self.model(**retain_inputs_dict)[module]
            lora_retain_hidden = torch.stack(lora_retain_outputs) * layers_retain_attention_mask
            retain_loss = torch.norm(lora_retain_hidden - orig_retain_hidden, dim=-1, p=2, dtype=torch.float).nanmean()
        else:
            retain_loss = 0

        if circuit_breaker_coeff > 0 and circuit_breaker_hidden is not None:
            lora_circuit_breaker_outputs = self.model(**cb_inputs_dict)[module]
            lora_circuit_breaker_hidden = torch.stack([lora_circuit_breaker_outputs[l] for l in self.lora_target_layers])
            
            normalized_lora_circuit_breaker_outputs = lora_circuit_breaker_hidden / (torch.norm(lora_circuit_breaker_hidden, dim=-1, keepdim=True, dtype=torch.float) + 1e-6)
            normalized_circuit_breaker_outputs = circuit_breaker_hidden / (torch.norm(circuit_breaker_hidden, dim=-1, keepdim=True, dtype=torch.float) + 1e-6)
            
            inner_product = (normalized_lora_circuit_breaker_outputs * normalized_circuit_breaker_outputs) * layers_circuit_breaker_attention_mask
            circuit_breaker_loss = torch.relu(inner_product.nansum(dim=-1)).nansum() / (layers_circuit_breaker_attention_mask.sum() + 1e-6)
        else:
            circuit_breaker_loss = 0

        loss = retain_coeff * retain_loss + circuit_breaker_coeff * circuit_breaker_loss
        
        self.current_training_step += 1
        return (loss, ) if return_outputs else loss

if __name__ == "__main__":

    assert torch.cuda.is_available(), "CUDA is not available"

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
    parser.add_argument('--model_name', type=str, default='allenai/OLMo-2-1124-7B-Instruct')
    parser.add_argument('--save_name', type=str, default='')
    parser.add_argument('--revision', type=str, default='main')
    args = parser.parse_args()

    # Default logic for coefficients
    if args.alg == 'rr':
        args.pdbs = 4 if args.pdbs is None else args.pdbs
        args.retain_coef = 5 if args.retain_coef is None else args.retain_coef
        args.remove_coef = 5 if args.remove_coef is None else args.remove_coef
    elif args.alg == 'rr-lat':
        args.pdbs = 1 if args.pdbs is None else args.pdbs
        args.retain_coef = 5 if args.retain_coef is None else args.retain_coef
        args.remove_coef = 5 if args.remove_coef is None else args.remove_coef
    else:  # lat
        args.pdbs = 1 if args.pdbs is None else args.pdbs
        args.retain_coef = 1 if args.retain_coef is None else args.retain_coef
        args.remove_coef = 5 if args.remove_coef is None else args.remove_coef

    if 'smollm2' not in args.model_name:
        args.hidden_dim = 4096
    else:
        args.hidden_dim = 2048
        args.layers = [l for l in args.layers if l < 24]

    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print()

    model, tokenizer = get_model_and_tokenizer(args.model_name, revision=args.revision)

    # Load retain_examples
    retain_text_dataset = load_dataset(RETAIN_TEXT_DS_NAME, 'wikitext-103-raw-v1')['train']
    retain_text_dataset = retain_text_dataset.rename_column('page', 'text')
    retain_text_dataset = retain_text_dataset.shuffle(seed=42).select(range(int(args.num_train_examples)))
    tokenized_retain_text_dataset = retain_text_dataset.map(lambda x: wikitext_tokenize_function(x, tokenizer), batched=True)
    
    retain_datasets = [tokenized_retain_text_dataset]
    if args.model_name == 'allenai/OLMo-2-1124-7B-Instruct' or 'Unlearning' in args.model_name:
        bio_retain_dataset = load_dataset(BIO_RETAIN_DS_NAME, 'bio-retain-corpus')
        bio_retain_dataset = bio_retain_dataset['train'].shuffle(seed=42).select(range(int(args.num_train_examples * 0.25))) 
        tokenized_bio_retain_dataset = bio_retain_dataset.map(lambda x: cb_retain_tokenize_function(x, tokenizer), batched=True)
        retain_datasets.append(tokenized_bio_retain_dataset) 
    retain_datasets = [concatenate_datasets(retain_datasets).shuffle(seed=42).select(range(args.num_train_examples))]

    # Load remove examples
    num_remove_to_take = args.num_train_examples if not args.unlearn_corrupt else int(args.num_train_examples * (1+args.corrupt_ratio))
    if 'smollm2' not in args.model_name:
        bio_remove_dataset = load_dataset(BIO_REMOVE_DS_NAME, token=hf_token)
        bio_remove_dataset = bio_remove_dataset['train'].select(range(num_remove_to_take))
        tokenized_remove_dataset = bio_remove_dataset.map(lambda x: cb_tokenize_function(x, tokenizer), batched=True)
        remove_datasets = [tokenized_remove_dataset]
        if args.unlearn_corrupt:
            corrupt_dataset = load_dataset(BIO_CORRUPT_REWRITTEN_DS_NAME, token=hf_token) if args.corrupt_ds=='rewritten' else load_dataset(BIO_CORRUPT_SHUFFLED_DS_NAME, token=hf_token)
            corrupt_dataset = corrupt_dataset['train'].select(range(args.num_train_examples, int(args.num_train_examples * args.corrupt_ratio)))
            tokenized_corrupt_dataset = corrupt_dataset.map(lambda x: cb_tokenize_function(x, tokenizer), batched=True)
            retain_datasets.append(tokenized_corrupt_dataset)
    else:
        remove_refusal_compliance_dataset = load_dataset(RETAIN_REFUSAL_COMPLIANCE_DS_NAME)['train']
        remove_refusal_compliance_dataset = remove_refusal_compliance_dataset.shuffle(seed=42).select(range(args.num_train_examples))
        tokenized_remove_compliance_dataset = remove_refusal_compliance_dataset.map(lambda x: refusal_compliance_tokenize_function(x, tokenizer, refuse=False), batched=True)
        remove_datasets = [tokenized_remove_compliance_dataset]
        if args.unlearn_corrupt:
            corrupt_dataset = load_dataset(RETAIN_INCOMPETENT_COMPLIANCE_DS_NAME, token=hf_token)['train']
            corrupt_dataset = corrupt_dataset.select(range(args.num_train_examples, int(args.num_train_examples * args.corrupt_ratio)))
            tokenized_corrupt_dataset = corrupt_dataset.map(lambda x: incompetent_compliance_tokenize_function(x, tokenizer, refuse=False), batched=True)
            retain_datasets.append(tokenized_corrupt_dataset)

    all_retain_datasets = concatenate_datasets(retain_datasets)
    all_remove_datasets = concatenate_datasets(remove_datasets)
    train_dataset = UnlearningDataset(all_remove_datasets, all_retain_datasets)

    # Model Wrapping and Trainer Setup
    if args.alg == 'rr' or args.alg == 'rr-lat':
        lora_layers_to_transform = [i for i in range(max(args.layers) + 1)]
        if args.lora:
            lora_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=16,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if 'OLMo' in args.model_name else None,
                    lora_dropout=0.05,
                    bias='none',
                    layers_to_transform=lora_layers_to_transform,
                    task_type="CAUSAL_LM",
                )
            model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()

        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=args.lr, 
            gradient_accumulation_steps=(32 // args.pdbs),  
            per_device_train_batch_size=args.pdbs,  
            per_device_eval_batch_size=args.pdbs,
            num_train_epochs=1, 
            weight_decay=0.01,
            gradient_checkpointing=True,
            fp16=True,
            save_strategy="no"
        )
        trainer = RRTrainer(args, model, training_args, train_dataset, tokenizer, args.layers)
        print("rr trainer")

    elif args.alg == 'lat':
        lora_layers_to_transform = [i for i in range(max(args.layers) + 1)]
        if args.lora:
            lora_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=16,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if 'OLMo' in args.model_name else None,
                    lora_dropout=0.05,
                    bias='none',
                    layers_to_transform=lora_layers_to_transform,
                    task_type="CAUSAL_LM",
                )
            model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()

        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=args.lr, 
            gradient_accumulation_steps=(32 // args.pdbs),  
            per_device_train_batch_size=args.pdbs,  
            per_device_eval_batch_size=args.pdbs,
            num_train_epochs=1, 
            weight_decay=0.01,
            gradient_checkpointing=True,
            fp16=True,
            save_strategy="no"
        )
        trainer = LATTrainer(args, model, training_args, train_dataset, tokenizer, args.layers)
        print("lat trainer")

    # Start Training
    model.train()
    trainer.train()

    # --- FIX START ---
    
    # 1. Removed clear_hooks(model) as hooks are now managed by context managers.
    
    # 2. Add Barrier: Ensure all GPUs finish training before Rank 0 starts evaluation/saving
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()

    # 3. Fix Rank Logic: Cast environment variable to int
    rank = int(os.environ.get('LOCAL_RANK', 0))
    print(f"rank: {rank}")
    
    if rank == 0:
        # if 'smollm2' not in args.model_name:
        #     # mmlu_acc = lm_eval_model(model, task='mmlu', limit=args.mmlu_eval_limit, revision=args.revision, tokenizer=tokenizer)
        #     wmdp_acc = lm_eval_model(model, task='wmdp_bio_robust', limit=args.wmdp_eval_limit, revision=args.revision, tokenizer=tokenizer)
        #     # print(f'***\nFinal wmdp_acc: {wmdp_acc}, final mmlu_acc {mmlu_acc}\n***')
        #     print(wmdp_acc)
        # else:
        #     jailbreak_score = jailbreak_eval_model(model, tokenizer, num_examples=500, pfx=None, num_fs=0)
        #     print(f'***\nFinal jailbreak_score: {jailbreak_score}, final mmlu_acc {mmlu_acc}\n***')

        if args.lora:
            # Merge LoRA before saving
            model = model.merge_and_unload()
        
        if args.save_name:
            save_path_name = args.model_name
            if 'models/' in save_path_name:
                save_path_name = save_path_name.replace('models/', '')
            
            save_dir = f"./models/{save_path_name}_{args.save_name}"
            print(f"Saving model to {save_dir}")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            # print('FINAL MODEL:')
            # # Launch the eval as a subprocess - python -m scripts.eval_mmlu_stem --model_path $OUTPUT_DIR --batch_size 8
            # # python -m scripts.eval_wmdp_robust --model_path $OUTPUT_DIR --batch_size 8
            # import subprocess
            # subprocess.run(['python', '-m', 'scripts.eval_mmlu_stem', '--model_path', save_dir, '--batch_size', '8'])
            # subprocess.run(['python', '-m', 'scripts.eval_wmdp_robust', '--model_path', save_dir, '--batch_size', '8'])
        
        # mmlu_acc = lm_eval_model(model, task='mmlu', limit=args.mmlu_agieval_limit, revision=args.revision, tokenizer=tokenizer)
            
    # Optional: Barrier at the very end to prevent premature exit of non-zero ranks
    if dist.is_initialized():
        dist.barrier()
        
    print('Done :)')