#!/usr/bin/env python3
"""LoRA finetune OLMo-2-7B-Instruct on the mixed WMDP bio dataset.

Expects the mixed dataset to already exist at data/wmdp_mixed (created by
prepare_wmdp_mixed.py). Saves the LoRA adapter locally.

Usage::

    # Single GPU
    python scripts/train_olmo_wmdp.py

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=4 scripts/train_olmo_wmdp.py
"""

import os

import torch
import torch.distributed as dist
from datasets import Dataset, load_from_disk
from peft import LoraConfig
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


OUTPUT_DIR = "runs/olmo_wmdp_lora"
MODEL_NAME = "allenai/OLMo-2-1124-7B-Instruct"
DATASET_DIR = "data/wmdp_mixed"


class NoShuffleSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, train_dataset):
        return SequentialSampler(train_dataset)


def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))

    # Load the mixed dataset
    ds = load_from_disk(DATASET_DIR)
    if not isinstance(ds, Dataset):
        raise TypeError(f"Expected Dataset, got {type(ds)}")
    print(f"Loaded {len(ds)} examples from {DATASET_DIR}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map={"": f"cuda:{rank}"},
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    peft_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj"
        ], # "up_proj", "down_proj",
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
            bf16=True,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            lr_scheduler_type="cosine",
            max_length=1024,
            max_steps=-1,
            num_train_epochs=4,
            optim="adamw_8bit",
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=32,
            report_to="wandb",
            run_name="olmo_wmdp_lora",
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
