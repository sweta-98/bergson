import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import argparse
from transformers import Trainer, TrainingArguments, TrainerCallback
from datasets import concatenate_datasets, load_dataset
from datasets import Dataset as hf_dataset
from peft import get_peft_model, LoraConfig
from tampering_utils import *
import gc
import wandb


class WMDPEvalCallback(TrainerCallback):
    """Custom callback that runs WMDP evals on a fixed cadence and logs to W&B."""

    def __init__(self, model, eval_every, args, tokenizer):
        self.model = model
        self.eval_every = eval_every
        self.run_args = args
        self.tokenizer = tokenizer
        self.eval_results = []

    @staticmethod
    def _flatten(d, parent_key=''):
        items = {}
        for k, v in d.items():
            nk = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(WMDPEvalCallback._flatten(v, nk))
            else:
                items[nk] = v
        return items

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_every == 0:
            eval_tasks = ["wmdp_bio_aisi", "wmdp_bio_aisi_cloze_verified", "wmdp_bio_retrieval"]
            harness_results = lm_eval_model_multi_task(
                self.model,
                tasks=eval_tasks,
                revision=self.run_args.revision,
                tokenizer=self.tokenizer,
                limit=self.run_args.wmdp_eval_limit
            )

            eval_results = {}
            for task_name in harness_results["results"]:
                task_results = harness_results["results"][task_name]
                accuracy = None
                if "acc,none" in task_results:
                    accuracy = task_results["acc,none"]
                elif "acc_norm,none" in task_results:
                    accuracy = task_results["acc_norm,none"]
                else:
                    print(f"Misisng metrics for {task_name}")
                    continue

                eval_results[task_name] = accuracy

            print(f"Step {state.global_step}: Evaluation Results: {eval_results}")
            self.eval_results.append({
                'step': state.global_step,
                'results': eval_results
            })

            # ---- W&B logging ----
            # Only log scalars; prefix with eval/
            flat = self._flatten(eval_results)
            logs = {f"eval/{k}": v for k, v in flat.items() if isinstance(v, (int, float))}
            try:
                wandb.log(logs, step=state.global_step)
            except Exception as e:
                print(f"[W&B] Logging failed at step {state.global_step}: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        # Optionally replay all eval snapshots to ensure they're in the run history
        if self.eval_results:
            try:
                for rec in self.eval_results:
                    flat = self._flatten(rec['results'])
                    logs = {f"eval/{k}": v for k, v in flat.items() if isinstance(v, (int, float))}
                    wandb.log(logs, step=rec['step'])
            except Exception as e:
                print(f"[W&B] Final logging failed: {e}")


class LogSpaceCheckpointCallback(TrainerCallback):
    """Save checkpoints at log-spaced intervals for early steps."""

    def __init__(self, log_steps_until=100, regular_save_steps=None):
        """
        Args:
            log_steps_until: Use log spacing until this step (e.g., 100)
            regular_save_steps: After log_steps_until, save every N steps (optional)
        """
        self.log_steps_until = log_steps_until
        self.regular_save_steps = regular_save_steps

        # Generate log-spaced steps: 0, 1, 2, 4, 8, 16, 32, 64, ...
        self.log_steps = {0, 1}  # Always include step 0 and 1
        power = 1
        while 2**power <= log_steps_until:
            self.log_steps.add(2**power)
            power += 1

        print(f"Will save checkpoints at log-spaced steps: {sorted(self.log_steps)}")

    def on_step_end(self, args, state, control, **kwargs):
        """Trigger saves at log-spaced intervals."""
        step = state.global_step

        # Check if we should save at this step
        should_save = False

        if step <= self.log_steps_until and step in self.log_steps:
            should_save = True
            print(f"[Checkpoint] Step {step}: Triggering save (log-spaced)")
        elif self.regular_save_steps and step > self.log_steps_until:
            if step % self.regular_save_steps == 0:
                should_save = True
                print(f"[Checkpoint] Step {step}: Triggering save (linear interval)")

        if not should_save and step % 100 == 0:
            print(f"[Checkpoint] Step {step}: No save scheduled")

        control.should_save = should_save  # <-- KEY CHANGE: explicitly set True or False

        return control


if __name__ == '__main__':

    assert torch.cuda.is_available(), "CUDA is not available"

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_examples', type=int, default=512)
    parser.add_argument('--wmdp_eval_limit', type=int, default=None)
    parser.add_argument('--jailbreak_eval_limit', type=int, default=200)
    parser.add_argument('--eval_every', type=int, default=-1)
    parser.add_argument('--mmlu_agieval_limit', type=int, default=None)
    parser.add_argument('--include_bio_retain', type=bool, default=False)
    parser.add_argument('--include_bio_remove', type=bool, default=False)
    parser.add_argument('--include_bio_filtered_docs', type=bool, default=False)
    parser.add_argument('--include_chat_retain', type=bool, default=False)
    parser.add_argument('--include_text_retain', type=bool, default=False)
    parser.add_argument('--include_refusal_retain', type=bool, default=False)
    parser.add_argument('--include_compliance_remove', type=bool, default=False)
    parser.add_argument('--include_incompetent_compliance_retain', type=bool, default=False)
    parser.add_argument('--include_corrupt_rewritten', type=bool, default=False)
    parser.add_argument('--include_corrupt_shuffled', type=bool, default=False)
    parser.add_argument('--include_corrupt_deepfried', type=bool, default=False)
    parser.add_argument('--lora', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_warmup', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max_chunks', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--save_checkpoint', action='store_true', help='Save final HF checkpoint')

    # ---- W&B specific args ----
    parser.add_argument('--wandb_project', type=str, default='midtraining_safety_tampering')
    parser.add_argument('--wandb_entity', type=str, default='eleutherai')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default='online')  # 'online' | 'offline' | 'disabled'

    args = parser.parse_args()

    if 'smollm2' not in args.model_name:
        args.hidden_dim = 4096
    else:
        args.hidden_dim = 2048

    # ---- Create experiment ID with timestamp ----
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_method = "lora" if args.lora else "fp"
    model_short_name = args.model_name.split('/')[-1]  # Get model name without org prefix

    # Determine dataset type for experiment ID
    if args.include_bio_remove:
        args.dataset_type = "adversarial"
    elif args.include_text_retain:
        args.dataset_type = "benign"
    elif args.include_bio_filtered_docs:
        args.dataset_type = "filtered"
    else:
        args.dataset_type = "custom"

    experiment_id = f"{model_short_name}-{training_method}-{args.dataset_type}-{timestamp}"

    # Create checkpoint directory if saving
    checkpoint_dir = None
    if args.save_checkpoint:
        checkpoint_dir = f"/projects/a5k/public/checkpoints/tampering/{experiment_id}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint will be saved to: {checkpoint_dir}")

    # ---- Initialize W&B (optional) ----
    # Respect --wandb_mode and feed full config
    if args.wandb_mode == 'disabled':
        os.environ['WANDB_DISABLED'] = 'true'

    try:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name or experiment_id,  # Use experiment_id as default
            config=vars(args),
            mode=args.wandb_mode if args.wandb_mode in ('online', 'offline', 'disabled') else 'online',
            reinit=True,
        )
        # Disable parameter/gradient watching to reduce overhead
        wandb.watch(models=None, log=None)
    except Exception as e:
        print(f"[W&B] init failed, proceeding without active run: {e}")

    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    model, tokenizer = get_model_and_tokenizer(args.model_name, revision=args.revision, dm='auto')
    training_datasets = []

    if args.include_chat_retain:
        retain_chat_dataset = load_dataset(RETAIN_CHAT_DS_NAME)['train_sft']
        retain_chat_dataset = retain_chat_dataset.shuffle(seed=args.seed).select(range(args.num_train_examples))
        tokenized_retain_chat_dataset = retain_chat_dataset.map(lambda x: ultrachat_tokenize_function(x, tokenizer, truncate=False), batched=True)
        training_datasets.append(tokenized_retain_chat_dataset)

    if args.include_text_retain:
        retain_text_dataset = load_dataset(RETAIN_TEXT_DS_NAME, 'wikitext-103-raw-v1')['train']
        retain_text_dataset = retain_text_dataset.rename_column('page', 'text')
        retain_text_dataset = retain_text_dataset.shuffle(seed=args.seed).select(range(args.num_train_examples))
        tokenized_retain_text_dataset = retain_text_dataset.map(lambda x: wikitext_tokenize_function(x, tokenizer, truncate=False), batched=True)
        training_datasets.append(tokenized_retain_text_dataset)

    if args.include_refusal_retain:
        retain_refusal_compliance_dataset = load_dataset(RETAIN_REFUSAL_COMPLIANCE_DS_NAME)['train']
        retain_refusal_compliance_dataset = retain_refusal_compliance_dataset.shuffle(seed=args.seed).select(range(args.num_train_examples))
        tokenized_retain_refusal_dataset = retain_refusal_compliance_dataset.map(lambda x: refusal_compliance_tokenize_function(x, tokenizer, refuse=True), batched=True)
        training_datasets.append(tokenized_retain_refusal_dataset)

    if args.include_compliance_remove:
        remove_refusal_compliance_dataset = load_dataset(RETAIN_REFUSAL_COMPLIANCE_DS_NAME)['train']
        remove_refusal_compliance_dataset = remove_refusal_compliance_dataset.shuffle(seed=args.seed).select(range(args.num_train_examples))
        tokenized_remove_compliance_dataset = remove_refusal_compliance_dataset.map(lambda x: refusal_compliance_tokenize_function(x, tokenizer, refuse=False), batched=True)
        training_datasets.append(tokenized_remove_compliance_dataset)

    if args.include_incompetent_compliance_retain:
        retain_incompetent_compliance_dataset = load_dataset(RETAIN_INCOMPETENT_COMPLIANCE_DS_NAME, token=hf_token)['train']
        retain_incompetent_compliance_dataset = retain_incompetent_compliance_dataset.shuffle(seed=args.seed).select(range(args.num_train_examples))
        tokenized_incompetent_compliance_dataset = retain_incompetent_compliance_dataset.map(lambda x: incompetent_compliance_tokenize_function(x, tokenizer, refuse=False), batched=True)
        training_datasets.append(tokenized_incompetent_compliance_dataset)

    if args.include_bio_retain:
        bio_retain_dataset = load_dataset(BIO_RETAIN_DS_NAME, 'bio-retain-corpus')
        bio_retain_dataset = bio_retain_dataset['train'].shuffle(seed=args.seed).select(range(args.num_train_examples))
        tokenized_bio_retain_dataset = bio_retain_dataset.map(lambda x: cb_retain_tokenize_function(x, tokenizer, truncate=False), batched=True)
        training_datasets.append(tokenized_bio_retain_dataset)

    if args.include_bio_remove:
        bio_remove_dataset = load_dataset(BIO_REMOVE_DS_NAME, token=hf_token)
        bio_remove_dataset = bio_remove_dataset['train'].shuffle(seed=args.seed).select(range(args.num_train_examples))
        tokenized_bio_remove_dataset = bio_remove_dataset.map(lambda x: cb_tokenize_function(x, tokenizer, truncate=False), batched=True)
        training_datasets.append(tokenized_bio_remove_dataset)

    if args.include_bio_filtered_docs:
        bio_filtered_docs_dataset = load_dataset(BIO_FILTERED_DOCS_DS_NAME, token=hf_token)
        bio_filtered_docs_dataset = bio_filtered_docs_dataset['train'].shuffle(seed=args.seed).select(range(args.num_train_examples))
        tokenized_bio_filtered_docs_dataset = bio_filtered_docs_dataset.map(lambda x: wikitext_tokenize_function(x, tokenizer, truncate=False), batched=True, num_proc=50)
        training_datasets.append(tokenized_bio_filtered_docs_dataset)

    if args.include_corrupt_rewritten:
        corrupt_dataset = load_dataset(BIO_CORRUPT_REWRITTEN_DS_NAME, token=hf_token)
        corrupt_dataset = corrupt_dataset['train'].shuffle(seed=args.seed).select(range(args.num_train_examples))
        tokenized_corrupt_dataset = corrupt_dataset.map(lambda x: cb_tokenize_function(x, tokenizer, truncate=False), batched=True)
        training_datasets.append(tokenized_corrupt_dataset)

    if args.include_corrupt_shuffled:
        corrupt_dataset = load_dataset(BIO_CORRUPT_SHUFFLED_DS_NAME, token=hf_token)
        corrupt_dataset = corrupt_dataset['train'].shuffle(seed=args.seed).select(range(args.num_train_examples))
        tokenized_corrupt_dataset = corrupt_dataset.map(lambda x: cb_tokenize_function(x, tokenizer, truncate=False), batched=True)
        training_datasets.append(tokenized_corrupt_dataset)

    if args.include_corrupt_deepfried:
        corrupt_dataset = load_dataset(BIO_CORRUPT_DEEPFRIED_DS_NAME, token=hf_token)
        corrupt_dataset = corrupt_dataset['train'].shuffle(seed=args.seed).select(range(args.num_train_examples))
        tokenized_corrupt_dataset = corrupt_dataset.map(lambda x: cb_tokenize_function(x, tokenizer, truncate=False), batched=True)
        training_datasets.append(tokenized_corrupt_dataset)

    interleaved_dataset = concatenate_datasets(training_datasets).shuffle()
    interleaved_dataset = interleaved_dataset.remove_columns(
        [col for col in interleaved_dataset.column_names if col not in ["input_ids", "token_type_ids", "attention_mask", "labels"]]
    )

    # Chunk interleaved_dataset into pieces of size 2048 tokens
    def chunk_example(example, chunk_size=MAX_LENGTH, pad_token_id=tokenizer.pad_token_id, max_chunks=5):
        input_ids = example["input_ids"]
        attention_mask = example.get("attention_mask", [1] * len(input_ids))
        labels = example.get("labels", input_ids.copy())
        token_type_ids = example.get("token_type_ids", [0] * len(input_ids))
        chunks = []
        for i in range(0, len(input_ids), chunk_size):
            chunk_input_ids = input_ids[i:i+chunk_size]
            chunk_attention_mask = attention_mask[i:i+chunk_size]
            chunk_labels = labels[i:i+chunk_size]
            chunk_token_type_ids = token_type_ids[i:i+chunk_size]
            # Pad if needed
            pad_len = chunk_size - len(chunk_input_ids)
            if pad_len > 0:
                chunk_input_ids += [pad_token_id] * pad_len
                chunk_attention_mask += [0] * pad_len
                chunk_labels += [-100] * pad_len
                chunk_token_type_ids += [0] * pad_len
            chunks.append({
                "input_ids": chunk_input_ids,
                "attention_mask": chunk_attention_mask,
                "labels": chunk_labels,
                "token_type_ids": chunk_token_type_ids,
            })
            if i >= ((max_chunks-1) * chunk_size):  # Limit the number of chunks to avoid memory issues
                break
        return chunks

    # if args.chunk:
    chunked_examples = []
    for i, example in enumerate(interleaved_dataset):
        chunked_examples.extend(chunk_example(example, chunk_size=MAX_LENGTH, max_chunks=args.max_chunks))
        if (i+1) % 1000 == 0:
            print(f"Processed {i+1} examples, total chunks: {len(chunked_examples)}")
    del interleaved_dataset
    interleaved_dataset = hf_dataset.from_list(chunked_examples).shuffle(seed=args.seed)
    # shorten to 85k examples if there are more than 85k
    if len(interleaved_dataset) > 85000:
        interleaved_dataset = interleaved_dataset.select(range(85000))
    del chunked_examples

    print(interleaved_dataset)
    gc.collect()
    torch.cuda.empty_cache()
    # quit()

    if args.lora:
        lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if 'OLMo' in args.model_name else None,
            )
        model = get_peft_model(model, lora_config)

    callbacks = []

    # Add WMDP eval callback if needed
    if args.eval_every > 0:
        callbacks.append(WMDPEvalCallback(model, args.eval_every, args, tokenizer))

    # Add log-space checkpoint callback
    callbacks.append(LogSpaceCheckpointCallback(
        log_steps_until=500,      # Log spacing for first 500 steps
        regular_save_steps=2000    # Then save every 500 steps (optional)
    ))

    # Effective Batch Size: 2048 * 16
    training_args = TrainingArguments(
        output_dir=checkpoint_dir if args.save_checkpoint else "./tampering_results",
        learning_rate=args.lr,
        gradient_accumulation_steps=1, # used to be 32
        per_device_train_batch_size=16,
        per_device_eval_batch_size=1,
        num_train_epochs=args.epochs,
        include_num_input_tokens_seen=True,
        weight_decay=0.01,
        gradient_checkpointing=False,  # to avoid oom errors
        fp16=True,  # to avoid oom errors,
        save_strategy="steps",        # Enable step-based saving controlled by callback
        save_steps=10000,              # Large value - actual saves controlled by LogSpaceCheckpointCallback
        save_only_model=True,          # Only save model weights (saves space)
        warmup_steps=args.lr_warmup,  # Warmup steps for learning rate scheduler
        logging_strategy="steps",  # Enable logging during training
        logging_steps=1,
        report_to=["wandb"],              # <— W&B integration
        run_name=args.wandb_run_name       # <— W&B run name (optional)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=interleaved_dataset,
        eval_dataset=interleaved_dataset,
        callbacks=callbacks,
    )

    for param in model.parameters():
        param.requires_grad = True

    model.train()
    trainer.train()

    if args.eval_every > 0:
        print(f'***\nWMDP eval performance over time: {callbacks[0].eval_results}\n***')

    # mmlu_acc = lm_eval_model(model, task='mmlu', limit=args.mmlu_agieval_limit, revision=args.revision, tokenizer=tokenizer)

    if 'smollm2' not in args.model_name and args.eval_every < 0:
        wmdp_acc = lm_eval_model(model, task='wmdp_bio_aisi', limit=args.wmdp_eval_limit, revision=args.revision, tokenizer=tokenizer)
        print(f'***\nFinal wmdp_acc: {wmdp_acc}\n***')
        pass

    elif 'smollm2' in args.model_name:
        jailbreak_score = jailbreak_eval_model(model, tokenizer, num_examples=args.jailbreak_eval_limit, pfx=None, num_fs=0)
        print(f'***\nFinal jailbreak_score: {jailbreak_score}\n***')

    # if args.open_book_eval:
    #     custom_eval_acc = open_book_bio_eval(model, tokenizer, open_book=True)
    #     print(f'***\nCustom bio eval acc: {custom_eval_acc}\n***')

    if args.lora:
        model = model.merge_and_unload()

    # Save checkpoint if requested
    if args.save_checkpoint and checkpoint_dir:
        print(f"Saving final checkpoint to {checkpoint_dir}...")
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved successfully!")

        # Push to HuggingFace Hub
        # print(f"Pushing checkpoint to HuggingFace Hub...")
        # repo_id = f"EleutherAI/{experiment_id}"
        # model.push_to_hub(repo_id, private=False)
        # tokenizer.push_to_hub(repo_id, private=False)
        # print(f"Successfully pushed to hub: {repo_id}")

        # Log checkpoint path and hub repo to W&B
        if wandb.run is not None:
            wandb.run.summary["checkpoint_dir"] = checkpoint_dir
            # wandb.run.summary["hf_repo"] = repo_id

    # ---- Finish W&B run ----
    try:
        wandb.finish()
    except Exception:
        pass

    print('Done :)\n\n\n')