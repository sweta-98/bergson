from timeit import timeit
import traceback
import json
import os

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict
from lm_eval.tasks import TaskManager
from torch import Tensor
from transformers import TrainerCallback
import wandb

from bergson.utils.utils import assert_type

# #region agent log
DEBUG_LOG_PATH = "/home/a5k/lucia.a5k/bergson/.cursor/debug.log"
# #endregion

@torch.inference_mode()
def stable_rank(A: Tensor) -> float:
    eps = torch.finfo(A.dtype).eps

    # Spectral norm
    _, S, _ = torch.svd_lowrank(A, q=1)
    spec = S[0]

    # Frobenius norm
    frob = torch.linalg.matrix_norm(A, ord="fro")

    # Ratio of the squares
    return (frob.pow(2) / (spec.pow(2) + eps)).item()


class EvalCallback(TrainerCallback):
    def __init__(
        self, 
        tokenizer, 
        pairs_per_batch: int, 
        run_every_steps=50, 
        ref_model=None,
        tasks=["wmdp_bio", "mmlu"],
        include_path=None,
    ):
        self.tokenizer = tokenizer
        self.run_every_steps = run_every_steps
        self.pairs_per_batch = pairs_per_batch

        self.ref_state_dict = {}
        if ref_model:
            for name, param in ref_model.named_parameters():
                if param.dim() > 1:  # Only track matrices (weights), not biases
                    self.ref_state_dict[name] = (
                        param.detach().cpu().clone().to(torch.float32)
                    )
        self.include_path = include_path
        self.tasks = tasks

        # Calculate the stable rank of the initial parameters
        ranks = self._compute_module_ranks()
        mean = torch.tensor(ranks, dtype=torch.float32).mean().item()
        std = torch.tensor(ranks, dtype=torch.float32).std().item()
        print(
            f"Initial Mean Stable Rank of Modules: {mean:.4f}, Std: {std:.4f}"
            f"Max: {max(ranks):.4f}, Min: {min(ranks):.4f}"
        )

    def _compute_module_ranks(self):
        """Computes the stable rank of each module."""
        ranks = []
        with torch.no_grad():
            for param in self.ref_state_dict.values():
                # Only track matrices (weights), not biases
                if param.dim() <= 1:  
                    continue

                ranks.append(stable_rank(param))
        return ranks

    def _compute_param_diffs(self, current_model):
        """Computes the average stable rank of (W_now - W_init)
        and the frobenius norm of the difference."""
        mod_stable_ranks = []

        # For frobenius norm
        ssqs = torch.tensor(0.0, dtype=torch.float32)

        with torch.no_grad():
            for name, param in current_model.named_parameters():
                if name in self.ref_state_dict:
                    ref_param = self.ref_state_dict[name].to(device=param.device)
                    if param.dim() <= 1:
                        continue

                    diff = param.to(torch.float32) - ref_param

                    mod_stable_ranks.append(stable_rank(diff))

                    # Accumulate sum of squares
                    ssq = diff.pow(2).sum()
                    ssqs += ssq.to(ssqs.device)

        frob_norm = ssqs.sqrt()

        print(f"Frobenius Norm: {frob_norm :.4f}")

        return {
            "stable_ranks": mod_stable_ranks,
            "frob_norm ": frob_norm,
        }

    def _run_evaluation(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            start_time = timeit()
            model = kwargs["model"]
            was_training = model.training
            model.eval()

            print("Calculating Stable Rank of parameter update...")
            if self.ref_state_dict:
                diffs = self._compute_param_diffs(model)
                ranks = diffs["stable_ranks"]
                frob_norm = diffs["frob_norm "]
                avg_stable_rank = torch.tensor(ranks, dtype=torch.float32).mean().item()

                print(f"Avg Stable Rank of Diff: {avg_stable_rank:.4f}", flush=True)
                wandb.log(
                    {
                        "update_stable_rank": avg_stable_rank,
                        "update_stable_rank_dist": wandb.Histogram(ranks),
                        "update_frob_norm ": frob_norm,
                    },
                    step=state.global_step,
                )

            print(
                f"\n[Step {state.global_step}] Running WMDP-Bio and MMLU (limit=40)...",
                flush=True
            )
            
            assert self.include_path is not None, "include_path must be set to load tasks"
            task_manager = (
                TaskManager(include_path=str(self.include_path)) 
            )

            lm_wrapper = HFLM(model) # type: ignore

            requested_tasks = ["wmdp_bio_cloze_verified", "wmdp_bio_robust"]
            for task in requested_tasks:
                config = task_manager._get_config(task)
                print("got config for", task, ":", config, flush=True)

            # If this fails it may be an LM eval versioning issue - 0.4 came with a
            # breaking change
            results = evaluator.simple_evaluate(
                model=lm_wrapper,
                model_args="",
                tasks=requested_tasks,
                log_samples=True,
                task_manager=TaskManager(include_path=str(self.include_path)),
            )
            results = assert_type(dict, results)

            print("results", results, flush=True)

            # Dynamically extract metrics for all tasks
            metrics = {}
            for task_name in self.tasks:
                acc = None
                
                # Check groups first (for group tasks like wmdp_bio_robust)
                if "groups" in results and task_name in results["groups"]:
                    task_results = results["groups"][task_name]
                    acc = task_results.get("acc,none", task_results.get("acc"))
                # Check results (for individual tasks)
                elif "results" in results and task_name in results["results"]:
                    task_results = results["results"][task_name]
                    acc = task_results.get("acc,none", task_results.get("acc"))
                
                if acc is not None:
                    # Convert task name to metric name (replace special chars)
                    metric_name = task_name.replace("-", "_") + "_acc"
                    print(f"{task_name} Acc: {acc}", flush=True)
                    metrics[metric_name] = acc
                else:
                    print(f"Warning: Could not find accuracy metric for task '{task_name}'", flush=True)

            if metrics:
                wandb.log(metrics, step=state.global_step)

            if was_training:
                model.train()
            print(f"Evaluation time: {timeit() - start_time} seconds", flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        self._run_evaluation(args, state, control, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.run_every_steps == 0 and state.global_step > 0:
            self._run_evaluation(args, state, control, **kwargs)


# Hypothesis: unlearning would work better if we induced a high rank parameter update.


# class PairedDataCollator:
#     def __call__(self, features):
#         retain_ids = []
#         forget_ids = []
#         retain_mask = []
#         forget_mask = []

#         def ensure_tensor(val):
#             if isinstance(val, torch.Tensor):
#                 return val.clone().detach()
#             return torch.tensor(val, dtype=torch.long)

#         for f in features:
#             retain_ids.append(ensure_tensor(f["retain_input_ids"]))
#             forget_ids.append(ensure_tensor(f["forget_input_ids"]))
#             retain_mask.append(ensure_tensor(f["retain_attention_mask"]))
#             forget_mask.append(ensure_tensor(f["forget_attention_mask"]))

#         retain_batch = torch.stack(retain_ids)
#         forget_batch = torch.stack(forget_ids)
#         retain_mask_batch = torch.stack(retain_mask)
#         forget_mask_batch = torch.stack(forget_mask)

#         # Structure: [Retain_1, ..., Retain_N, Forget_1, ..., Forget_N]
#         input_ids = torch.cat([retain_batch, forget_batch], dim=0)
#         attention_mask = torch.cat([retain_mask_batch, forget_mask_batch], dim=0)

#         retain_labels = retain_batch.clone()
#         # Set forget labels to -100 so they are ignored in CE Loss
#         forget_labels = torch.full_like(forget_batch, -100)
#         labels = torch.cat([retain_labels, forget_labels], dim=0)

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": labels,
#         }


# def paired_generator(forget_set, retain_set, rank, world_size, max_seq_len):
#     if world_size > 1:
#         forget_set = forget_set.shard(num_shards=world_size, index=rank)
#         retain_set = retain_set.shard(num_shards=world_size, index=rank)

#     forget_iter = iter(forget_set)
#     retain_iter = iter(retain_set)

#     for f_sample, r_sample in zip(forget_iter, retain_iter):
#         yield {
#             "forget_input_ids": f_sample["input_ids"][:max_seq_len],
#             "forget_attention_mask": [1] * max_seq_len,
#             "retain_input_ids": r_sample["input_ids"][:max_seq_len],
#             "retain_attention_mask": [1] * max_seq_len,
#         }

