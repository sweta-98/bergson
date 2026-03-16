#!/usr/bin/env python3
"""Verify MCQA query gradient correctness for WMDP trackstar.

This script checks:
1. Tokenization: Are labels placed on the correct answer token?
2. Gradients: Do per-example query gradients have reasonable cosine similarity?

Saves an on-disk gradient index (float32 memmap) for further analysis.

Usage:
    python scripts/verify_query_grads.py [--model MODEL] [--subset SUBSET]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── bergson imports ──────────────────────────────────────────────────────────
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bergson.data import tokenize
from bergson.config import DataConfig
from bergson.format import apply_format


def verify_tokenization(tokenizer, ds_formatted, data_cfg: DataConfig, n_print: int = 5):
    """Check that labels land on exactly the answer token for every example."""
    # Tokenize the full formatted dataset
    ds_tok = ds_formatted.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=data_cfg, tokenizer=tokenizer, max_length=2048),
    )

    n_ok = 0
    n_bad_no_label = 0
    n_bad_wrong_token = 0
    n_bad_multi_token = 0

    for idx in range(len(ds_tok)):
        input_ids = ds_tok[idx]["input_ids"]
        labels = ds_tok[idx].get("labels", input_ids)
        completion = ds_formatted[idx]["completion"]

        # Find labelled positions (where label != -100)
        labelled_positions = [i for i, l in enumerate(labels) if l != -100]

        if len(labelled_positions) == 0:
            n_bad_no_label += 1
            if n_bad_no_label <= n_print:
                print(f"  [NO LABEL] idx={idx} completion='{completion}'")
                # Debug: show the chat-formatted string
                convo = [
                    {"role": "user", "content": ds_formatted[idx]["prompt"]},
                    {"role": "assistant", "content": completion},
                ]
                formatted = tokenizer.apply_chat_template(convo, tokenize=False)
                print(f"    Chat string (last 100 chars): ...{formatted[-100:]}")
                # Show where rfind would find the answer
                pos = formatted.rfind(completion)
                print(f"    rfind('{completion}') = {pos}, len(formatted) = {len(formatted)}")
            continue

        # Check that the labelled tokens decode to the expected answer
        labelled_token_ids = [labels[p] for p in labelled_positions]
        decoded_label = tokenizer.decode(labelled_token_ids, skip_special_tokens=False)

        # The completion for MCQA should be a single letter (A/B/C/D)
        if len(labelled_positions) > 1:
            n_bad_multi_token += 1
            if n_bad_multi_token <= n_print:
                print(
                    f"  [MULTI-TOKEN] idx={idx} completion='{completion}' "
                    f"decoded='{decoded_label}' n_tokens={len(labelled_positions)} "
                    f"positions={labelled_positions}"
                )

        # Strip whitespace for comparison since tokenizers may include leading space
        if decoded_label.strip() != completion.strip():
            n_bad_wrong_token += 1
            if n_bad_wrong_token <= n_print:
                print(
                    f"  [WRONG TOKEN] idx={idx} expected='{completion}' "
                    f"got='{decoded_label}' positions={labelled_positions}"
                )
        else:
            n_ok += 1

    total = len(ds_tok)
    print(f"\n=== Tokenization Verification ===")
    print(f"Total examples:          {total}")
    print(f"OK (correct label):      {n_ok}")
    print(f"NO LABEL (labels=-100):  {n_bad_no_label}")
    print(f"WRONG TOKEN:             {n_bad_wrong_token}")
    print(f"MULTI-TOKEN label:       {n_bad_multi_token}")

    if n_bad_no_label > 0 or n_bad_wrong_token > 0:
        print("\n*** TOKENIZATION ISSUES DETECTED ***")
    else:
        print("\nTokenization looks correct.")

    return ds_tok


def compute_query_gradients(
    model,
    tokenizer,
    ds_tok,
    output_dir: Path,
    batch_size: int = 4,
    proj_dim: int = 512,
    seed: int = 42,
):
    """Compute per-example query gradients, project, and save to disk.

    We collect the gradient of the CE loss w.r.t. all trainable parameters,
    flatten into a single vector, and apply a random projection to `proj_dim`
    for tractable cosine-similarity analysis.
    """
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()

    # Collect parameter names we'll differentiate through
    param_names = []
    param_list = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            param_names.append(name)
            param_list.append(p)

    total_params = sum(p.numel() for p in param_list)
    print(f"Differentiating through {len(param_list)} parameters ({total_params:,} elements)")

    # Random projection matrix (chunked to save memory)
    rng = np.random.RandomState(seed)
    # We'll project on-the-fly rather than materializing the full matrix

    n = len(ds_tok)
    output_dir.mkdir(parents=True, exist_ok=True)
    grad_path = output_dir / "query_grads.bin"
    info_path = output_dir / "info.json"

    # Pre-allocate memmap
    grads_mm = np.memmap(grad_path, dtype="float32", mode="w+", shape=(n, proj_dim))

    # Build a stable random projection seed per chunk
    chunk_seeds = rng.randint(0, 2**31, size=100)

    print(f"Computing gradients for {n} examples (proj_dim={proj_dim})...")

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_indices = list(range(start, end))
        bs = len(batch_indices)

        # Pad batch manually
        input_ids_list = [ds_tok[i]["input_ids"] for i in batch_indices]
        labels_list = [
            ds_tok[i].get("labels", ds_tok[i]["input_ids"]) for i in batch_indices
        ]
        max_len = max(len(ids) for ids in input_ids_list)

        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        padded_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids_list]
        padded_labels = [lab + [-100] * (max_len - len(lab)) for lab in labels_list]

        x = torch.tensor(padded_ids, dtype=torch.long, device=device)
        y = torch.tensor(padded_labels, dtype=torch.long, device=device)

        # Per-example gradients via sequential backward
        for bi in range(bs):
            model.zero_grad()
            xi = x[bi : bi + 1]
            yi = y[bi : bi + 1]

            logits = model(xi).logits[:, :-1]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                yi[:, 1:].flatten(),
                reduction="sum",
            )

            # Check loss is non-zero (label exists)
            if loss.item() == 0.0:
                print(f"  WARNING: zero loss at idx={batch_indices[bi]}")
                grads_mm[batch_indices[bi]] = 0.0
                continue

            loss.backward()

            # Gather flat gradient and project
            flat_grad = torch.cat([p.grad.flatten() for p in param_list])

            # Random projection: project in chunks to save memory
            projected = torch.zeros(proj_dim, device=device)
            offset = 0
            for ci, p in enumerate(param_list):
                g = p.grad.flatten()
                chunk_rng = np.random.RandomState(chunk_seeds[ci % len(chunk_seeds)])
                # Gaussian random projection for this chunk
                proj = torch.tensor(
                    chunk_rng.randn(proj_dim, g.numel()).astype(np.float32)
                    / np.sqrt(proj_dim),
                    device=device,
                )
                projected += proj @ g
                offset += g.numel()

            grads_mm[batch_indices[bi]] = projected.cpu().numpy()

        if (start // batch_size) % 10 == 0:
            print(f"  Processed {end}/{n}")

    grads_mm.flush()

    # Save metadata
    with open(info_path, "w") as f:
        json.dump(
            {
                "num_grads": n,
                "proj_dim": proj_dim,
                "dtype": "float32",
                "total_params": total_params,
                "param_names": param_names,
            },
            f,
            indent=2,
        )

    print(f"Saved gradient index to {output_dir}")
    return grads_mm


def compute_query_gradients_fast(
    model,
    tokenizer,
    ds_tok,
    output_dir: Path,
    proj_dim: int = 512,
    seed: int = 42,
):
    """Compute per-example query gradients using embedding gradient only.

    This is much faster than full-parameter gradients and still diagnostic:
    the embedding gradient at the answer position tells us whether the loss
    is flowing through the correct token.
    """
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()

    # We'll collect gradient of loss w.r.t. the input embeddings
    embed = model.get_input_embeddings()
    embed_dim = embed.weight.shape[1]

    n = len(ds_tok)
    output_dir.mkdir(parents=True, exist_ok=True)

    # We'll store the embedding gradient at the answer position
    # Shape: (n, embed_dim)
    grad_path = output_dir / "query_embed_grads.bin"
    info_path = output_dir / "embed_grads_info.json"
    grads_mm = np.memmap(grad_path, dtype="float32", mode="w+", shape=(n, embed_dim))

    print(f"Computing embedding gradients for {n} examples...")

    for idx in range(n):
        input_ids = ds_tok[idx]["input_ids"]
        labels = ds_tok[idx].get("labels", input_ids)

        # Find the answer position (last non-(-100) label, shifted by 1 for next-token pred)
        labelled_positions = [i for i, l in enumerate(labels) if l != -100]
        if not labelled_positions:
            grads_mm[idx] = 0.0
            continue

        # The valid_mask position is one before the labelled position
        # (because valid_masks[:, :-1] = padded_labels[:, 1:] != -100)
        answer_predict_pos = labelled_positions[0] - 1
        if answer_predict_pos < 0:
            grads_mm[idx] = 0.0
            continue

        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        y = torch.tensor([labels], dtype=torch.long, device=device)

        # Enable gradient on embeddings
        model.zero_grad()
        embeds = embed(x)
        embeds.requires_grad_(True)
        embeds.retain_grad()

        # Forward from embeddings
        outputs = model(inputs_embeds=embeds)
        logits = outputs.logits[:, :-1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y[:, 1:].flatten(),
            reduction="sum",
        )

        if loss.item() == 0.0:
            print(f"  WARNING: zero loss at idx={idx}")
            grads_mm[idx] = 0.0
            continue

        loss.backward()

        # Get the embedding gradient at the answer prediction position
        embed_grad = embeds.grad[0, answer_predict_pos].float().cpu().numpy()
        grads_mm[idx] = embed_grad

        if idx % 50 == 0:
            print(f"  Processed {idx}/{n} (loss={loss.item():.4f})")

    grads_mm.flush()

    with open(info_path, "w") as f:
        json.dump(
            {
                "num_grads": n,
                "grad_dim": embed_dim,
                "dtype": "float32",
                "method": "embed_grad_at_answer_position",
            },
            f,
            indent=2,
        )

    print(f"Saved embedding gradient index to {output_dir}")
    return grads_mm


def analyze_cosine_similarity(grads: np.ndarray, max_pairs: int = 5000):
    """Compute and report cosine similarity statistics."""
    # Filter out zero rows
    norms = np.linalg.norm(grads, axis=1)
    nonzero_mask = norms > 1e-8
    n_nonzero = nonzero_mask.sum()
    n_zero = len(grads) - n_nonzero
    print(f"\n=== Cosine Similarity Analysis ===")
    print(f"Total examples:   {len(grads)}")
    print(f"Non-zero grads:   {n_nonzero}")
    print(f"Zero grads:       {n_zero}")

    if n_zero > 0:
        print(f"  WARNING: {n_zero} examples have zero gradient (likely label issues)")

    if n_nonzero < 2:
        print("Not enough non-zero gradients for similarity analysis.")
        return

    valid_grads = grads[nonzero_mask]
    # Normalize
    valid_norms = np.linalg.norm(valid_grads, axis=1, keepdims=True)
    normed = valid_grads / valid_norms

    # Compute pairwise cosine similarities (subsample if too many)
    n = len(normed)
    if n * (n - 1) // 2 > max_pairs:
        rng = np.random.RandomState(0)
        indices = rng.choice(n, size=int(np.sqrt(max_pairs * 2)) + 1, replace=False)
        normed_sub = normed[indices]
    else:
        normed_sub = normed

    sim_matrix = normed_sub @ normed_sub.T
    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(len(normed_sub), k=1)
    sims = sim_matrix[triu_idx]

    print(f"\nPairwise cosine similarity ({len(sims)} pairs):")
    print(f"  Mean:   {sims.mean():.4f}")
    print(f"  Std:    {sims.std():.4f}")
    print(f"  Min:    {sims.min():.4f}")
    print(f"  Max:    {sims.max():.4f}")
    print(f"  Median: {np.median(sims):.4f}")

    # Distribution
    for threshold in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        frac = (sims > threshold).mean()
        print(f"  > {threshold:.1f}: {frac:.1%}")

    # Norm statistics
    print(f"\nGradient norm statistics:")
    print(f"  Mean:   {norms[nonzero_mask].mean():.6f}")
    print(f"  Std:    {norms[nonzero_mask].std():.6f}")
    print(f"  Min:    {norms[nonzero_mask].min():.6f}")
    print(f"  Max:    {norms[nonzero_mask].max():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Verify WMDP query gradients")
    parser.add_argument(
        "--model",
        default="allenai/OLMo-2-1124-7B-Instruct",
        help="Model name or path",
    )
    parser.add_argument("--subset", default="wmdp-bio", help="WMDP subset")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument(
        "--output_dir",
        default="runs/verify_query_grads",
        help="Output directory for gradient index",
    )
    parser.add_argument(
        "--tokenize_only",
        action="store_true",
        help="Only verify tokenization, skip gradient computation",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="Use fast embedding-gradient method (default: True)",
    )
    parser.add_argument(
        "--no_fast",
        action="store_true",
        help="Use full parameter gradient method (slow but thorough)",
    )
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=512,
        help="Projection dim for full gradient method",
    )
    args = parser.parse_args()

    format_template = str(
        Path(__file__).resolve().parents[1] / "bergson" / "templates" / "mcqa.yaml"
    )
    output_dir = Path(args.output_dir)

    # ── Step 1: Load and format dataset ──────────────────────────────────────
    print("Loading WMDP dataset...")
    ds = load_dataset("cais/wmdp", args.subset, split=args.split)
    print(f"Loaded {len(ds)} examples from cais/wmdp/{args.subset}/{args.split}")

    print("Applying MCQA format template...")
    ds_formatted = apply_format(ds, format_template)

    # Quick sanity check on a few examples
    print("\nSample formatted examples:")
    for i in range(min(3, len(ds_formatted))):
        row = ds_formatted[i]
        print(f"  [{i}] prompt[-50:]: ...{row['prompt'][-50:]}")
        print(f"       completion: '{row['completion']}'")

    # ── Step 2: Load tokenizer and verify tokenization ───────────────────────
    print(f"\nLoading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    data_cfg = DataConfig(
        prompt_column="prompt",
        completion_column="completion",
        truncation=True,
    )

    ds_tok = verify_tokenization(tokenizer, ds_formatted, data_cfg)

    if args.tokenize_only:
        print("\n--tokenize_only specified, skipping gradient computation.")
        return

    # ── Step 3: Load model and compute gradients ─────────────────────────────
    print(f"\nLoading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    model.requires_grad_(False)
    model.get_input_embeddings().requires_grad_(True)

    if args.no_fast:
        grads = compute_query_gradients(
            model, tokenizer, ds_tok, output_dir, proj_dim=args.proj_dim
        )
    else:
        grads = compute_query_gradients_fast(
            model, tokenizer, ds_tok, output_dir
        )

    # ── Step 4: Analyze cosine similarities ──────────────────────────────────
    analyze_cosine_similarity(np.array(grads))


if __name__ == "__main__":
    main()
