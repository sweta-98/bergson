#!/usr/bin/env python3
"""LoRA finetune OLMo-2-7B-Instruct on pile + WMDP bio (forget + retain) in FP32.

Usage::

    # Single GPU
    python scripts/train_olmo_wmdp_fp32.py

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=4 scripts/train_olmo_wmdp_fp32.py
"""

import os
from datetime import datetime

import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from peft import LoraConfig
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = f"runs/olmo_wmdp_lora_fp32/{timestamp}"
MODEL_NAME = "allenai/OLMo-2-1124-7B-Instruct"


class NoShuffleSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, train_dataset):
        return SequentialSampler(train_dataset)


def prepare_dataset() -> Dataset:
    """Combine pile-10k + WMDP bio forget + WMDP bio retain."""
    # Load the existing mixed dataset (forget + retain)
    wmdp_mixed = load_from_disk("data/wmdp_mixed")
    if not isinstance(wmdp_mixed, Dataset):
        raise TypeError(f"Expected Dataset, got {type(wmdp_mixed)}")

    # Load pile-10k
    pile = load_dataset("NeelNanda/pile-10k", split="train")

    # Add source column to pile
    pile = pile.add_column("source", ["pile"] * len(pile))

    # Keep only the 'text' and 'source' columns in both
    wmdp_mixed = wmdp_mixed.select_columns(["text", "source"])
    pile = pile.select_columns(["text", "source"])

    combined = concatenate_datasets([wmdp_mixed, pile])
    print(f"Combined dataset: {len(combined)} examples")
    print(f"  WMDP mixed: {len(wmdp_mixed)} (forget + retain)")
    print(f"  Pile-10k: {len(pile)}")
    return combined


def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))

    ds = prepare_dataset()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map={"": f"cuda:{rank}"},
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    peft_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj",
        ],
        lora_dropout=0.1,
        use_rslora=True,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = NoShuffleSFTTrainer(
        model=model,
        train_dataset=ds,
        args=SFTConfig(
            ddp_find_unused_parameters=False,
            bf16=False,
            fp16=False,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            lr_scheduler_type="cosine",
            max_length=1024,
            max_steps=-1,
            num_train_epochs=4,
            optim="adamw_8bit",
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=16,
            report_to="wandb",
            run_name=f"olmo_wmdp_lora_fp32_{timestamp}",
            save_steps=500,
            warmup_steps=50,
            weight_decay=0.01,
        ),
        peft_config=peft_config,
    )

    trainer.train()

    if rank == 0:
        adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
        trainer.model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        print(f"\nAdapter saved to {adapter_path}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
