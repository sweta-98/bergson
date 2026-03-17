#!/usr/bin/env python3
"""Full SFT of OLMo-2-7B-Instruct on forget+retain in FP32 with full Adam.

Usage::
    torchrun --nproc_per_node=4 scripts/train_olmo_wmdp_sft_fp32.py
"""

import os
from datetime import datetime

import torch
import torch.distributed as dist
from datasets import Dataset, load_from_disk
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = f"runs/olmo_wmdp_sft_fp32/{timestamp}"
MODEL_NAME = "allenai/OLMo-2-1124-7B-Instruct"
DATASET_DIR = "data/wmdp_mixed"


class NoShuffleSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, train_dataset):
        return SequentialSampler(train_dataset)


def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))

    ds = load_from_disk(DATASET_DIR)
    if not isinstance(ds, Dataset):
        raise TypeError(f"Expected Dataset, got {type(ds)}")
    print(f"Loaded {len(ds)} examples from {DATASET_DIR}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map={"": f"cuda:{rank}"},
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    trainer = NoShuffleSFTTrainer(
        model=model,
        train_dataset=ds,
        args=SFTConfig(
            ddp_find_unused_parameters=False,
            bf16=False,
            fp16=False,
            gradient_checkpointing=True,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            logging_steps=1,
            lr_scheduler_type="cosine",
            max_length=1024,
            max_steps=-1,
            num_train_epochs=4,
            optim="adamw_torch",
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=2,
            report_to="wandb",
            run_name=f"olmo_wmdp_sft_fp32_{timestamp}",
            save_steps=500,
            warmup_steps=50,
            weight_decay=0.01,
        ),
    )

    trainer.train()

    if rank == 0:
        model_path = os.path.join(OUTPUT_DIR, "final_model")
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"\nModel saved to {model_path}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
