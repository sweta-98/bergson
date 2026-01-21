"""Unlearning via entropy maximization at frozen tuned lens layers."""

import sys
import os
from typing import Any, Callable, List, Tuple, Union
import argparse
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import seed_worker
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from bergson.unlearn.circuit_breaker.cas_utils import *
from transformers.modeling_utils import unwrap_model
from tuned_lens import TunedLens


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
    def __init__(self, run_args, model, args, train_dataset, tokenizer, lora_target_layers, lens=None, **kwargs):
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
        self.lens = lens

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
        # 1. SAFELY UNWRAP MODEL
        # This works for both DDP (returns inner model) and Single GPU (returns model as-is).
        # We need this to access .disable_adapter() and specific layers.
        unwrapped_model = unwrap_model(model)
        
        # Determine device from inputs (safest way in DDP)
        # If inputs aren't on device yet, fall back to model device
        target_device = inputs["input_ids"].device if hasattr(inputs["input_ids"], "device") else unwrapped_model.device

        # === retain ===
        retain_input_ids = inputs.get(f"input_ids").to(target_device)
        retain_attention_mask = inputs.get(f"attention_mask").to(target_device)
        # ==== cb ====
        circuit_breaker_input_ids = inputs.get(f"bio_remove_input_ids").to(target_device)
        circuit_breaker_attention_mask = inputs.get(f"bio_remove_attention_mask").to(target_device)

        # ==== Forward Inputs ====
        module = 'hidden_states'
        retain_inputs_dict = dict(input_ids=retain_input_ids, attention_mask=retain_attention_mask, output_hidden_states=True)
        cb_inputs_dict = dict(input_ids=circuit_breaker_input_ids, attention_mask=circuit_breaker_attention_mask, output_hidden_states=True)

        # ===== Step Coeff ====
        # Recalculate global batch size for scheduling to be accurate in DDP
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        # Note: self.run_args.pdbs is per-device.
        
        scheduled_coeff = min([1.0, self.current_training_step / (self.run_args.num_train_examples / (self.run_args.pdbs * world_size))])
        retain_coeff = self.retain_coef * scheduled_coeff 
        circuit_breaker_coeff = self.remove_coef * (1 - 0.25 * scheduled_coeff)

        # Optimization: Broadcasting masks (Batch, Seq) -> (1, Batch, Seq, 1)
        broadcast_retain_mask = retain_attention_mask.unsqueeze(0).unsqueeze(-1)
        broadcast_cb_mask = circuit_breaker_attention_mask.unsqueeze(0).unsqueeze(-1)

        # Use unwrapped_model for context manager (DDP wrapper doesn't have disable_adapter)
        with unwrapped_model.disable_adapter():
            unwrapped_model.eval()
            with torch.no_grad():
                ### Retain control
                if retain_coeff > 0:
                    orig_retain_outputs = unwrapped_model(**retain_inputs_dict)[module]
                    orig_retain_hidden = torch.stack(orig_retain_outputs).detach()
                    orig_retain_hidden *= broadcast_retain_mask
                    del orig_retain_outputs

        unwrapped_model.train()

        ### Retain control
        if retain_coeff > 0:
            # We use unwrapped_model here because we are accessing specific hidden states.
            # DDP usually requires using 'model' to sync gradients, but since we are 
            # effectively doing a manual loss calculation on sub-components, 
            # and Accelerate/Trainer handles the backward pass sync, this is generally safe.
            # If you see hanging, switch these forward passes to use 'model' (but you lose direct access to [module] output structure if wrapped).
            lora_retain_outputs = unwrapped_model(**retain_inputs_dict)[module]
            lora_retain_hidden = torch.stack(lora_retain_outputs) * broadcast_retain_mask
            retain_loss = torch.norm(lora_retain_hidden - orig_retain_hidden, dim=-1, p=2, dtype=torch.float).nanmean()
        else:
            retain_loss = 0

        ### Forget loss - entropy maximization via tuned lens (memory-efficient CE proxy)
        if circuit_breaker_coeff > 0:
            lora_circuit_breaker_outputs = unwrapped_model(**cb_inputs_dict)[module]

            # Use cross-entropy with random targets as memory-efficient entropy proxy
            # Minimizing CE against random tokens spreads probability mass -> maximizes entropy
            layer_losses = []
            lens_device = next(self.lens.parameters()).device
            for layer_idx in self.lora_target_layers:
                hidden = lora_circuit_breaker_outputs[layer_idx]  # [batch, seq, hidden]
                hidden_bf16 = hidden.to(device=lens_device, dtype=torch.bfloat16)

                lens_logits = self.lens(hidden_bf16, idx=layer_idx)  # [batch, seq, vocab]
                batch_size, seq_len, vocab_size = lens_logits.shape

                # Random target tokens
                random_targets = torch.randint(
                    0, vocab_size, (batch_size, seq_len), device=lens_logits.device
                )

                # Compute CE loss only on non-padded tokens
                mask = circuit_breaker_attention_mask.bool()
                logits_flat = lens_logits[mask]  # [num_valid_tokens, vocab]
                targets_flat = random_targets[mask]  # [num_valid_tokens]

                if logits_flat.numel() > 0:
                    ce_loss = F.cross_entropy(logits_flat.float(), targets_flat, reduction='mean')
                    layer_losses.append(ce_loss)

            # Minimize CE against random targets to maximize entropy
            # Normalize by ln(vocab_size) to keep loss in similar scale
            log_vocab = torch.log(torch.tensor(float(vocab_size), device=target_device))
            if layer_losses:
                mean_ce = torch.stack(layer_losses).mean()
                circuit_breaker_loss = mean_ce / log_vocab
                mean_entropy = -mean_ce  # For logging compatibility (approx)
            else:
                circuit_breaker_loss = torch.tensor(0.0, device=target_device)
                mean_entropy = torch.tensor(0.0)
        else:
            circuit_breaker_loss = torch.tensor(0.0, device=target_device)
            mean_entropy = torch.tensor(0.0)

        loss = retain_coeff * retain_loss + circuit_breaker_coeff * circuit_breaker_loss

        if self.current_training_step % 32 == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            entropy_val = mean_entropy.item() if isinstance(mean_entropy, torch.Tensor) else mean_entropy
            print(f"retain_coeff: {retain_coeff:.4f} || forget_coeff: {circuit_breaker_coeff:.4f} || retain_loss: {retain_loss:.4f} || forget_loss: {circuit_breaker_loss:.4f} || mean_entropy: {entropy_val:.4f}")
        
        # Optimization: Moved heavy eval out of loop
        
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
    parser.add_argument('--pdbs', type=int, default=4)
    parser.add_argument('--retain_coef', type=float, default=5.0)
    parser.add_argument('--remove_coef', type=float, default=5.0)
    parser.add_argument('--lora_r', type=float, default=16)
    parser.add_argument('--lora', type=bool, default=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[5, 10, 15, 20, 25, 30], help="List of layers to target")
    parser.add_argument('--model_name', type=str, default='EleutherAI/deep-ignorance-unfiltered')
    parser.add_argument('--save_name', type=str, default='')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--lens_path', type=str, required=True, help="Path to tuned lens weights")
    parser.add_argument('--skip_eval', action='store_true', help="Skip final evaluation (for faster tuning)")
    
    args = parser.parse_args()

    if 'smollm2' in args.model_name:
        args.layers = [l for l in args.layers if l < 24]

    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print()

    model, tokenizer = get_model_and_tokenizer(args.model_name, revision=args.revision)

    # Load frozen tuned lens
    print(f"Loading tuned lens from: {args.lens_path}")
    device = next(model.parameters()).device
    lens = TunedLens.from_model(model, bias=True)
    lens_state_dict = torch.load(f"{args.lens_path}/params.pt", map_location=device)

    # Map saved keys (e.g., "0.weight") to expected keys (e.g., "layer_translators.0.weight")
    mapped_state_dict = {}
    num_layers = len(lens)
    for i in range(num_layers):
        src_weight_key = f"{i}.weight"
        src_bias_key = f"{i}.bias"
        dst_weight_key = f"layer_translators.{i}.weight"
        dst_bias_key = f"layer_translators.{i}.bias"
        if src_weight_key in lens_state_dict:
            mapped_state_dict[dst_weight_key] = lens_state_dict[src_weight_key]
        if src_bias_key in lens_state_dict:
            mapped_state_dict[dst_bias_key] = lens_state_dict[src_bias_key]

    # Copy unembed params from initialized lens
    current_state = lens.state_dict()
    for key in current_state:
        if key.startswith("unembed"):
            mapped_state_dict[key] = current_state[key]

    lens.load_state_dict(mapped_state_dict)

    # Remove accelerate hooks from lens submodules (they get copied via deepcopy)
    from accelerate.hooks import remove_hook_from_module
    for submodule in lens.modules():
        remove_hook_from_module(submodule)

    lens = lens.to(device=device, dtype=torch.bfloat16)
    lens.eval()
    for param in lens.parameters():
        param.requires_grad = False
    print(f"Loaded lens with {len(lens)} layer translators (frozen)")

    # Load retain_examples
    retain_text_dataset = load_dataset(RETAIN_TEXT_DS_NAME, 'wikitext-103-raw-v1')['train']
    retain_text_dataset = retain_text_dataset.rename_column('page', 'text')
    retain_text_dataset = retain_text_dataset.shuffle(seed=42).select(range(int(args.num_train_examples)))
    tokenized_retain_text_dataset = retain_text_dataset.map(lambda x: wikitext_tokenize_function(x, tokenizer), batched=True, num_proc=NUM_PROC)

    retain_datasets = [tokenized_retain_text_dataset]
    if args.model_name == 'allenai/OLMo-2-1124-7B-Instruct' or 'Unlearning' in args.model_name:
        bio_retain_dataset = load_dataset(BIO_RETAIN_DS_NAME, 'bio-retain-corpus')
        bio_retain_dataset = bio_retain_dataset['train'].shuffle(seed=42).select(range(int(args.num_train_examples * 0.25))) 
        tokenized_bio_retain_dataset = bio_retain_dataset.map(lambda x: cb_retain_tokenize_function(x, tokenizer), batched=True, num_proc=NUM_PROC)
        retain_datasets.append(tokenized_bio_retain_dataset) 
    retain_datasets = [concatenate_datasets(retain_datasets).shuffle(seed=42).select(range(args.num_train_examples))]

    num_remove_to_take = args.num_train_examples if not args.unlearn_corrupt else int(args.num_train_examples * (1+args.corrupt_ratio))
    if 'smollm2' not in args.model_name:
        # remove data is wmdp bio remove papers
        bio_remove_dataset = load_dataset(BIO_REMOVE_DS_NAME, token=hf_token)
        bio_remove_dataset = bio_remove_dataset['train'].select(range(num_remove_to_take))
        tokenized_remove_dataset = bio_remove_dataset.map(lambda x: cb_tokenize_function(x, tokenizer), batched=True, num_proc=NUM_PROC)
        remove_datasets = [tokenized_remove_dataset]
        if args.unlearn_corrupt:
            corrupt_dataset = load_dataset(BIO_CORRUPT_REWRITTEN_DS_NAME, token=hf_token) if args.corrupt_ds=='rewritten' else load_dataset(BIO_CORRUPT_SHUFFLED_DS_NAME, token=hf_token)
            corrupt_dataset = corrupt_dataset['train'].select(range(args.num_train_examples, int(args.num_train_examples * args.corrupt_ratio)))
            tokenized_corrupt_dataset = corrupt_dataset.map(lambda x: cb_tokenize_function(x, tokenizer), batched=True, num_proc=NUM_PROC)
            retain_datasets.append(tokenized_corrupt_dataset)
    else:
        # remove data is compliances with harmful requests 
        remove_refusal_compliance_dataset = load_dataset(RETAIN_REFUSAL_COMPLIANCE_DS_NAME)['train']
        remove_refusal_compliance_dataset = remove_refusal_compliance_dataset.shuffle(seed=42).select(range(args.num_train_examples))
        tokenized_remove_compliance_dataset = remove_refusal_compliance_dataset.map(lambda x: refusal_compliance_tokenize_function(x, tokenizer, refuse=False), batched=True, num_proc=NUM_PROC)
        remove_datasets = [tokenized_remove_compliance_dataset]
        if args.unlearn_corrupt:
            corrupt_dataset = load_dataset(RETAIN_INCOMPETENT_COMPLIANCE_DS_NAME, token=hf_token)['train']
            corrupt_dataset = corrupt_dataset.select(range(args.num_train_examples, int(args.num_train_examples * args.corrupt_ratio)))
            tokenized_corrupt_dataset = corrupt_dataset.map(lambda x: incompetent_compliance_tokenize_function(x, tokenizer, refuse=False), batched=True, num_proc=NUM_PROC)
            retain_datasets.append(tokenized_corrupt_dataset)

    all_retain_datasets = concatenate_datasets(retain_datasets)
    all_remove_datasets = concatenate_datasets(remove_datasets)
    train_dataset = UnlearningDataset(all_remove_datasets, all_retain_datasets)

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
        bf16=True,
        max_grad_norm=1.0,
        save_strategy="no",
        ddp_find_unused_parameters=False # Required for Custom loops + PEFT usually
    )

    trainer = RRTrainer(args, model, training_args, train_dataset, tokenizer, args.layers, lens=lens)

    model.train()
    trainer.train()
    clear_hooks(model)

    # Final Evaluation
    if not args.skip_eval:
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