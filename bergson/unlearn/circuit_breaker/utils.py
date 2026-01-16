import json
import os
import gc

import torch
from transformers import AutoModelForCausalLM, LlavaNextForConditionalGeneration


def save_model_and_tokenizer(
    model_name_or_path, model, tokenizer, drop_layers_after, output_dir, trainer
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n\nModel and tokenizer saving to {output_dir}\n\n")

    # merge lora
    merged_model = model.merge_and_unload()

    # merge original layers
    if drop_layers_after is not None:
        # Move merged model to CPU first to free GPU memory
        merged_model = merged_model.to("cpu")
        gc.collect()

        # Load anchor model on CPU to avoid device distribution issues
        anchor_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=merged_model.dtype, device_map="cpu"
        )
        # Handle different model architectures - keep everything on CPU for saving
        if hasattr(merged_model, 'model') and hasattr(merged_model.model, 'layers'):
            # Llama-style models
            restored_layers = anchor_model.model.layers[drop_layers_after + 1 :]
            merged_model.model.layers = merged_model.model.layers + restored_layers
        elif hasattr(merged_model, 'gpt_neox') and hasattr(merged_model.gpt_neox, 'layers'):
            # GPTNeoX-style models
            restored_layers = anchor_model.gpt_neox.layers[drop_layers_after + 1 :]
            merged_model.gpt_neox.layers = merged_model.gpt_neox.layers + restored_layers
        merged_model.config = anchor_model.config
        # Update config to reflect the actual number of layers after restoration
        if hasattr(merged_model, 'gpt_neox') and hasattr(merged_model.gpt_neox, 'layers'):
            merged_model.config.num_hidden_layers = len(merged_model.gpt_neox.layers)
            merged_model.gpt_neox.config = merged_model.config
        elif hasattr(merged_model, 'model') and hasattr(merged_model.model, 'layers'):
            merged_model.config.num_hidden_layers = len(merged_model.model.layers)

    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    lorra_config_path = os.path.join(output_dir, "lorra_config.json")
    with open(lorra_config_path, "w", encoding="utf-8") as file:
        json.dump(trainer.lorra_args.to_dict(), file, indent=2)

    torch.use_deterministic_algorithms(False)
    if trainer.training_args.do_eval:
        trainer.evaluate()


def save_llava_model_and_tokenizer(
    model_name_or_path, model, processor, drop_layers_after, output_dir, trainer
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"MModel and processor saving to {output_dir}")

    # merge lora
    merged_model = model.merge_and_unload()
    # merge original layers

    anchor_model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name_or_path, device_map="auto", torch_dtype=merged_model.dtype
    )
    merged_model.language_model.model.layers = (
        merged_model.language_model.model.layers
        + anchor_model.language_model.model.layers[drop_layers_after + 1 :]
    )
    merged_model.config = anchor_model.config

    merged_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    lorra_config_path = os.path.join(output_dir, "lorra_config.json")
    with open(lorra_config_path, "w", encoding="utf-8") as file:
        json.dump(trainer.lorra_args.to_dict(), file, indent=2)

    torch.use_deterministic_algorithms(False)
    if trainer.training_args.do_eval:
        trainer.evaluate()
