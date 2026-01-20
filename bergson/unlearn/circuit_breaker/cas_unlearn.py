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


def zero_nan_grads(model):
    for _, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                p.grad[torch.isnan(p.grad)] = 0.


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
    torch.cuda.empty_cache()


def add_hooks(
    model: torch.nn.Module,
    create_adversary: Callable[[Union[Tuple[int, str], Tuple[str, str]]], Any],
    adversary_locations: Union[List[Tuple[int, str]], List[Tuple[str, str]]]
):

    # adverseries is a list of things created by `create_adversary()`
    # hooks is a list of the hooks we've added to the model
    adversaries = []
    hooks = []

    if len(adversary_locations) == 0:
        raise ValueError("No hook points provided")

    for layer, subcomponent in adversary_locations:
        # to add an adversary at (layer, subcomponent), we first get the layer
        parent = model.get_submodule(layer)
        # then we call `create_adversary()` to make a new adversary
        adversaries.append(create_adversary((layer, subcomponent)))
        # then we use `insert_hook` to add the adversary to the model
        hooks.append(insert_hook(parent, subcomponent, adversaries[-1]))
        # internally, `insert_hook` creates a new `CustomHook` and sets it as the subcomponent of `parent`

    return adversaries, hooks


class SVAdversary(nn.Module):
    
    def __init__(self, dim, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.device = device
        if dtype is not None:
            self.attack = torch.nn.Parameter(torch.zeros(dim, device=self.device, dtype=dtype))
        else:
            self.attack = torch.nn.Parameter(torch.zeros(dim, device=self.device))
    
    def forward(self, x):
        if self.device is None or self.device != x.device:
            with torch.no_grad():
                self.device = x.device
                self.attack.data = self.attack.data.to(self.device)

        perturbation = (self.attack.view(1, 1, self.dim)).expand(x.shape[0], x.shape[1], self.dim).to(x.dtype)
        x = x + perturbation
        return x


class SPAdversary(nn.Module):
    
    def __init__(self, dim, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.device = device
        self.sp_size = 32  # how many tokens to perturb up to
        if dtype is not None:
            self.attack = torch.nn.Parameter(torch.zeros(1, self.sp_size, dim, device=self.device, dtype=dtype))
        else:
            self.attack = torch.nn.Parameter(torch.zeros(1, self.sp_size, dim, device=self.device))
    
    def forward(self, x):
        if x.shape[1] == 1:  # generation mode (perturbation already applied)
            return x
        if self.device is None or self.device != x.device:
            with torch.no_grad():
                self.device = x.device
                self.attack.data = self.attack.data.to(self.device)
        sys.stdout.flush()
        num_toks = min([self.sp_size, x.shape[1]])  # perturb up to num_toks tokens in x
        perturbation = self.attack.expand(x.shape[0], self.sp_size, self.dim).to(x.dtype)
        pfx_perturbed  = x[:, :num_toks, :] + perturbation[:, :num_toks, :]
        new_x = torch.cat((pfx_perturbed, x[:, num_toks:, :]), dim=1)
        return new_x


class CombinedAdversary(nn.Module):
    
    def __init__(self, dim, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.device = device
        self.sp_size = 32  # how many tokens to perturb up to
        if dtype is not None:
            self.sv = torch.nn.Parameter(torch.zeros(dim, device=self.device, dtype=dtype))
            self.sp = torch.nn.Parameter(torch.zeros(1, self.sp_size, dim, device=self.device, dtype=dtype))
        else:
            self.sv = torch.nn.Parameter(torch.zeros(dim, device=self.device))
            self.sp = torch.nn.Parameter(torch.zeros(1, self.sp_size, dim, device=self.device))
    
    def forward(self, x):
        if x.shape[1] == 1:  # generation mode (perturbation already applied)
            return x
        if self.device is None or self.device != x.device:
            with torch.no_grad():
                self.device = x.device
                self.sv.data = self.sv.data.to(self.device)
                self.sp.data = self.sp.data.to(self.device)
        
        # add steering vector attack
        sv_perturbation = (self.sv.view(1, 1, self.dim)).expand(x.shape[0], x.shape[1], self.dim).to(x.dtype)
        x = x + sv_perturbation
        
        # add soft prompt attack
        num_toks = min([self.sp_size, x.shape[1]])  # perturb up to num_toks tokens in x
        sp_perturbation = self.sp.expand(x.shape[0], self.sp_size, self.dim).to(x.dtype)
        pfx_perturbed  = x[:, :num_toks, :] + sp_perturbation[:, :num_toks, :]
        new_x = torch.cat((pfx_perturbed, x[:, num_toks:, :]), dim=1)

        return new_x


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

    model.eval()

    # Clear and initialize the adversary
    clear_hooks(model)
    if type(layers) != list:
        layers = [layers,]

    create_adversary = lambda x: CombinedAdversary(dim=hidden_dim, device=device, dtype=model.dtype,)

    adversary_locations = [
        (f"{model_layers_module}.{layer_i}", "mlp") for layer_i in layers if type(layer_i) == int
    ]
    if "embedding" in layers:
        embed_name = "embed_in" if 'gpt_neox' in model_layers_module else 'embed_tokens'
        adversary_locations += [(model_layers_module.replace(".layers", ""), embed_name)]

    adversaries, wrappers = add_hooks(
        model,
        create_adversary=create_adversary,
        adversary_locations=adversary_locations
    )

    params = []
    for adv in adversaries:
        params += list(adv.parameters())
    
    # Define optimization utils
    adv_optim = torch.optim.AdamW(params, lr=learning_rate)
    losses = []

    # Optimize adversary to elicit attack labels
    for j in range(int(iterations)):
        adv_optim.zero_grad()

        # Compute the adversary loss
        away_labels = batch["input_ids"].to(device)
        away_labels_mask = batch["attention_mask"].to(device)
        loss = model(input_ids=away_labels, attention_mask=away_labels_mask, labels=away_labels).loss
        loss.backward()
        losses.append(round(loss.detach().item(), 4))

        zero_nan_grads(adv)
        adv_optim.step()

    del adversaries, adv_optim, away_labels, away_labels_mask, loss
    torch.cuda.empty_cache() 
    return losses, wrappers


class UnlearningTrainer(Trainer):

    def __init__(self, run_args, model, args, train_dataset, tokenizer, lora_target_layers, **kwargs):
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


class LATTrainer(UnlearningTrainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        # === retain ===
        retain_input_ids = inputs.get(f"input_ids").to(self.model.device)
        retain_attention_mask = inputs.get(f"attention_mask").to(self.model.device)
        # ==== cb ====
        remove_input_ids = inputs.get(f"bio_remove_input_ids").to(self.model.device)
        remove_attention_mask = inputs.get(f"bio_remove_attention_mask").to(self.model.device)

        # ==== Forward Inputs ====
        retain_inputs_dict = dict(input_ids=retain_input_ids, attention_mask=retain_attention_mask, labels=retain_input_ids)
        cb_inputs_dict = dict(input_ids=remove_input_ids, attention_mask=remove_attention_mask, labels=remove_input_ids)
        
        # ===== loss components =====
        mlm = 'model.gpt_neox.layers' if 'Unlearning' in self.run_args.model_name else 'model.layers'
        if self.run_args.lora:
            mlm = 'base_model.model.' + mlm
        losses, wrappers = gradient_descent_attack(
            batch=cb_inputs_dict,
            model=self.model,
            model_layers_module=mlm,
            layers=["embedding"] + args.layers, 
            learning_rate=self.run_args.adv_lr,
            iterations=self.run_args.attack_iters,
            device=self.model.device,
            hidden_dim=args.hidden_dim,
        )
        # print(f"adv losses: {losses}")
        self.model.train()

        retain_loss = super().compute_loss(model, retain_inputs_dict, return_outputs=False, num_items_in_batch=num_items_in_batch)
        remove_loss = super().compute_loss(model, cb_inputs_dict, return_outputs=False, num_items_in_batch=num_items_in_batch)
        scheduled_coeff = min([1.0, self.current_training_step / (self.run_args.num_train_examples / self.run_args.pdbs)])
        loss = self.run_args.retain_coef * retain_loss + self.run_args.remove_coef * remove_loss * (1-scheduled_coeff)
        clear_hooks(self.model)
        # print(f"model loss: {loss}")
        torch.cuda.empty_cache()
        if self.current_training_step % 10 == 0:
            print(f"model loss: {loss:.4f}, retain loss: {retain_loss:.4f}, remove loss: {remove_loss:.4f}, adv losses: {losses}")

        if self.current_training_step % 100 == 0:
            ask_simple_questions(model, self.trainer_tokenizer)
            model.train()

        self.current_training_step += 1
        sys.stdout.flush()

        return loss


class RRTrainer(UnlearningTrainer):
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        # === retain ===
        retain_input_ids = inputs.get(f"input_ids").to(self.model.device)
        retain_attention_mask = inputs.get(f"attention_mask").to(self.model.device)
        # ==== cb ====
        circuit_breaker_input_ids = inputs.get(f"bio_remove_input_ids").to(self.model.device)
        circuit_breaker_attention_mask = inputs.get(f"bio_remove_attention_mask").to(self.model.device)

        # ==== Forward Inputs ====
        module = 'hidden_states'
        retain_inputs_dict = dict(input_ids=retain_input_ids, attention_mask=retain_attention_mask, output_hidden_states=True)
        cb_inputs_dict = dict(input_ids=circuit_breaker_input_ids, attention_mask=circuit_breaker_attention_mask, output_hidden_states=True)

        # ===== Step Coeff ====
        scheduled_coeff = min([1.0, self.current_training_step / (self.run_args.num_train_examples / self.run_args.pdbs)])
        # retain_coeff, circuit_breaker_coeff = self.retain_coef * scheduled_coeff, self.remove_coef * (1-scheduled_coeff)
        retain_coeff = self.retain_coef * scheduled_coeff  # goes from 0 to 1 (used to be 0.1 to 1)
        circuit_breaker_coeff = self.remove_coef * (1 - 0.25 * scheduled_coeff)  # goes from 1 to 0.75
        # print(f"retain_coeff: {retain_coeff:.4f} || circuit_breaker_coeff: {circuit_breaker_coeff:.4f}")
        
        # ===== loss components =====
        layers_circuit_breaker_attention_mask = circuit_breaker_attention_mask.repeat(len(self.lora_target_layers), 1, 1).unsqueeze(-1)
        with self.model.disable_adapter():
            self.model.eval()
            with torch.no_grad():
                ### Retain control
                if retain_coeff > 0:
                    orig_retain_outputs = self.model(**retain_inputs_dict)[module]
                    orig_retain_hidden = torch.stack(orig_retain_outputs).detach()
                    layers_retain_attention_mask = retain_attention_mask.repeat(len(orig_retain_outputs), 1, 1).unsqueeze(-1)
                    orig_retain_hidden *= layers_retain_attention_mask
                    del orig_retain_outputs
                    gc.collect()

                ### Circuit Breaker control
                if circuit_breaker_coeff > 0:
                    circuit_breaker_outputs = self.model(**cb_inputs_dict)[module]
                    circuit_breaker_hidden = torch.stack([circuit_breaker_outputs[l].detach() for l in self.lora_target_layers])
                    del circuit_breaker_outputs
                    gc.collect()

        if args.alg == 'rr-lat':
            num_attack = 2  # for avoiding oom errors
            torch.cuda.empty_cache()
            cb_inputs_dict_lat = dict(input_ids=circuit_breaker_input_ids[:num_attack], attention_mask=circuit_breaker_attention_mask[:num_attack], labels=circuit_breaker_input_ids[:num_attack])
            mlm = 'model.gpt_neox.layers' if 'Unlearning' in self.run_args.model_name else 'model.layers'
            if self.run_args.lora and 'Unlearning' in self.run_args.model_name:
                mlm = 'base_model.' + mlm
            elif self.run_args.lora:
                mlm = 'base_model.model.' + mlm
            losses, wrappers = gradient_descent_attack(
                batch=cb_inputs_dict_lat,
                model=self.model,
                model_layers_module=mlm,
                layers=["embedding"] + args.layers,
                learning_rate=self.run_args.adv_lr,
                iterations=self.run_args.attack_iters,
                device=self.model.device,
                hidden_dim=args.hidden_dim,
            )
            # print(losses)
        self.model.train()

        ### Retain control
        if retain_coeff > 0:
            lora_retain_outputs = self.model(**retain_inputs_dict)[module]
            lora_retain_hidden = torch.stack(lora_retain_outputs) * layers_retain_attention_mask
            retain_loss = torch.norm(lora_retain_hidden - orig_retain_hidden, dim=-1, p=2, dtype=torch.float).nanmean()
        else:
            retain_loss = 0

        ### Circuit Breaker control
        if circuit_breaker_coeff > 0:
            lora_circuit_breaker_outputs = self.model(**cb_inputs_dict)[module]
            lora_circuit_breaker_hidden = torch.stack([lora_circuit_breaker_outputs[l] for l in self.lora_target_layers])
            normalized_lora_circuit_breaker_outputs = lora_circuit_breaker_hidden / (torch.norm(lora_circuit_breaker_hidden, dim=-1, keepdim=True, dtype=torch.float))
            normalized_circuit_breaker_outputs = circuit_breaker_hidden / (torch.norm(circuit_breaker_hidden, dim=-1, keepdim=True, dtype=torch.float))
            inner_product = (normalized_lora_circuit_breaker_outputs * normalized_circuit_breaker_outputs) * layers_circuit_breaker_attention_mask
            circuit_breaker_loss = torch.relu(inner_product.nansum(dim=-1)).nansum() / layers_circuit_breaker_attention_mask.sum()
        else:
            circuit_breaker_loss = 0

        loss = retain_coeff * retain_loss + circuit_breaker_coeff * circuit_breaker_loss
        if self.run_args.alg == 'rr-lat':
            clear_hooks(self.model)

        if self.current_training_step % 32 == 0:
            print(f"retain_coeff: {retain_coeff:.4f} || cb_coeff: {circuit_breaker_coeff:.4f} || retain_loss: {retain_loss:.4f} || cb_loss: {circuit_breaker_loss:.4f}")
            if args.alg == 'rr-lat':
                print(f"adv losses: {losses}")                
        # print('\nRETAIN', self.tokenizer.decode(retain_input_ids[0])[:1000])
        # print()
        # print('\nREMOVE', self.tokenizer.decode(circuit_breaker_input_ids[0])[:1000])
        # print('\n\n\n')
        if self.current_training_step % 128 == 0:
            ask_simple_questions(model, self.trainer_tokenizer)
            with torch.no_grad():
                wmdp_acc = lm_eval_model(self.model, task='wmdp_bio_robust', limit=self.run_args.wmdp_eval_limit, revision=self.run_args.revision, tokenizer=self.tokenizer)
                print(f'***\n wmdp_acc: {wmdp_acc}')
            model.train()
        
        self.current_training_step += 1
        sys.stdout.flush()

        return (loss, ) if return_outputs else loss


if __name__ == "__main__":

    assert torch.cuda.is_available(), "CUDA is not available"

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_examples', type=int, default=1024)
    parser.add_argument('--unlearn_corrupt', type=bool, default=False)
    parser.add_argument('--corrupt_ratio', type=float, default=0.5)
    parser.add_argument('--corrupt_ds', type=str, default='rewritten', choices=['rewritten', 'shuffled'])  # only matters for bio unlearning
    parser.add_argument('--wmdp_eval_limit', type=int, default=None)
    parser.add_argument('--mmlu_agieval_limit', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pdbs', type=int, default=None)
    parser.add_argument('--alg', type=str, choices=['rr', 'lat', 'rr-lat'], default='rr')
    parser.add_argument('--retain_coef', type=float, default=None) 
    parser.add_argument('--remove_coef', type=float, default=None)  
    parser.add_argument('--lora_r', type=float, default=16)  
    parser.add_argument('--adv_lr', type=float, default=2e-3)  # for lat
    parser.add_argument('--attack_iters', type=int, default=8)  # for lat
    parser.add_argument('--lora', type=bool, default=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[5, 10, 15, 20, 25, 30], help="List of layers to target")
    parser.add_argument('--model_name', type=str, default='allenai/OLMo-2-1124-7B-Instruct')
    parser.add_argument('--save_name', type=str, default='')
    parser.add_argument('--revision', type=str, default='main')
    args = parser.parse_args()
    if args.alg == 'rr':
        args.pdbs = 4 if args.pdbs is None else args.pdbs
        args.retain_coef = 5 if args.retain_coef is None else args.retain_coef
        args.remove_coef = 5 if args.remove_coef is None else args.remove_coef
    elif args.alg == 'rr-lat':
        args.pdbs = 1 if args.pdbs is None else args.pdbs
        args.retain_coef = 5 if args.retain_coef is None else args.retain_coef
        args.remove_coef = 5 if args.remove_coef is None else args.remove_coef
    else:  # lat
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
    # retain_chat_dataset = load_dataset(RETAIN_CHAT_DS_NAME)['train_sft']
    # retain_chat_dataset = retain_chat_dataset.shuffle(seed=42).select(range(int(args.num_train_examples * 0.25)))
    # tokenized_retain_chat_dataset = retain_chat_dataset.map(lambda x: ultrachat_tokenize_function(x, tokenizer), batched=True)
    # retain_datasets = [tokenized_retain_text_dataset, tokenized_retain_chat_dataset]
    retain_datasets = [tokenized_retain_text_dataset]
    if args.model_name == 'allenai/OLMo-2-1124-7B-Instruct' or 'Unlearning' in args.model_name:
        bio_retain_dataset = load_dataset(BIO_RETAIN_DS_NAME, 'bio-retain-corpus')
        bio_retain_dataset = bio_retain_dataset['train'].shuffle(seed=42).select(range(int(args.num_train_examples * 0.25))) 
        tokenized_bio_retain_dataset = bio_retain_dataset.map(lambda x: cb_retain_tokenize_function(x, tokenizer), batched=True)
        retain_datasets.append(tokenized_bio_retain_dataset) 
    retain_datasets = [concatenate_datasets(retain_datasets).shuffle(seed=42).select(range(args.num_train_examples))]

    num_remove_to_take = args.num_train_examples if not args.unlearn_corrupt else int(args.num_train_examples * (1+args.corrupt_ratio))
    if 'smollm2' not in args.model_name:
        # remove data is wmdp bio remove papers
        bio_remove_dataset = load_dataset(BIO_REMOVE_DS_NAME, token=hf_token)
        bio_remove_dataset = bio_remove_dataset['train'].select(range(num_remove_to_take))
        tokenized_remove_dataset = bio_remove_dataset.map(lambda x: cb_tokenize_function(x, tokenizer), batched=True)
        remove_datasets = [tokenized_remove_dataset]
        if args.unlearn_corrupt:
            # add (args.num_train_examples * args.corrupt_ratio) sets of unshuffled paired corrupt and remove examples to the retain and remove datasets
            corrupt_dataset = load_dataset(BIO_CORRUPT_REWRITTEN_DS_NAME, token=hf_token) if args.corrupt_ds=='rewritten' else load_dataset(BIO_CORRUPT_SHUFFLED_DS_NAME, token=hf_token)
            corrupt_dataset = corrupt_dataset['train'].select(range(args.num_train_examples, int(args.num_train_examples * args.corrupt_ratio)))
            tokenized_corrupt_dataset = corrupt_dataset.map(lambda x: cb_tokenize_function(x, tokenizer), batched=True)
            retain_datasets.append(tokenized_corrupt_dataset)
    else:
        # remove data is compliances with harmful requests 
        remove_refusal_compliance_dataset = load_dataset(RETAIN_REFUSAL_COMPLIANCE_DS_NAME)['train']
        remove_refusal_compliance_dataset = remove_refusal_compliance_dataset.shuffle(seed=42).select(range(args.num_train_examples))
        tokenized_remove_compliance_dataset = remove_refusal_compliance_dataset.map(lambda x: refusal_compliance_tokenize_function(x, tokenizer, refuse=False), batched=True)
        remove_datasets = [tokenized_remove_compliance_dataset]
        if args.unlearn_corrupt:
            # add (args.num_train_examples * args.corrupt_ratio) sets of unshuffled paired corrupt and remove examples to the retain and remove datasets
            corrupt_dataset = load_dataset(RETAIN_INCOMPETENT_COMPLIANCE_DS_NAME, token=hf_token)['train']
            corrupt_dataset = corrupt_dataset.select(range(args.num_train_examples, int(args.num_train_examples * args.corrupt_ratio)))
            tokenized_corrupt_dataset = corrupt_dataset.map(lambda x: incompetent_compliance_tokenize_function(x, tokenizer, refuse=False), batched=True)
            retain_datasets.append(tokenized_corrupt_dataset)

    all_retain_datasets = concatenate_datasets(retain_datasets)
    all_remove_datasets = concatenate_datasets(remove_datasets)
    train_dataset = UnlearningDataset(all_remove_datasets, all_retain_datasets)

    if args.alg == 'rr' or args.alg == 'rr-lat':

        lora_layers_to_transform = [i for i in range(max(args.layers) + 1)]
        # full_layers = False

        if args.lora:
            lora_config = LoraConfig(
                    r=args.lora_r,  # used to be 16
                    lora_alpha=16,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if 'OLMo' in args.model_name else None,
                    lora_dropout=0.05,
                    bias='none',
                    layers_to_transform=lora_layers_to_transform,
                    task_type="CAUSAL_LM",
                )

            model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()

        # drop_layers_after = max(args.layers) if not full_layers else None

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

    elif args.alg == 'lat':

        lora_layers_to_transform = [i for i in range(max(args.layers) + 1)]
        # full_layers = False

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

    model.train()
    trainer.train()
    clear_hooks(model)

    # print('FINAL MODEL:')
    # ask_simple_questions(model, tokenizer)
    # if args.lora and args.alg == 'rr':
    #     print('CONTROL WITH NO ADAPTERS:')
    #     with model.disable_adapter():
    #         ask_simple_questions(model, tokenizer)

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


