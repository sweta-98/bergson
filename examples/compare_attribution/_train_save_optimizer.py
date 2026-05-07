"""Train gpt2 on WikiText train[:32110] with a hand-rolled AdamW loop using a
SINGLE param group, so the saved optimizer.pt's flat state indexing matches
``model.named_parameters()`` order. HF Trainer's two-group (decay / no_decay)
split otherwise produces an indexing offset that mis-maps optimizer state to
parameters when bergson's load_from_optimizer iterates by index.

Outputs runs/compare_gpt2_wikitext/hf_optim/optimizer.pt with the standard
``{"state": {idx: {"exp_avg_sq": tensor, ...}}, "param_groups": [...]}`` shape
that ``bergson.utils.load_from_optimizer.load_from_optimizer`` expects.
"""

import math
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = Path("runs/compare_gpt2_wikitext/hf_optim")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda:0"


def main():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32).to(DEVICE)

    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:32110]"
    )

    def chunk(batch):
        all_ids: list[int] = []
        for text in batch["text"]:
            ids = tok(text, add_special_tokens=False)["input_ids"]
            all_ids.extend(ids + [tok.eos_token_id])
        n = len(all_ids) // 512
        all_ids = all_ids[: n * 512]
        return {
            "input_ids": [all_ids[i : i + 512] for i in range(0, n * 512, 512)],
            "labels": [all_ids[i : i + 512] for i in range(0, n * 512, 512)],
        }

    chunked = ds.map(
        chunk,
        batched=True,
        batch_size=2140,
        remove_columns=ds.column_names,
        num_proc=8,
    ).with_format("torch")
    print(f"chunked: {len(chunked)} sequences of 512 tokens", flush=True)

    # Effective bs 80 to match MAGIC. Use micro_bs=10 + accum=8 because a
    # single A40 can't hold bs=80 lm_head logits (7.7 GiB just for those).
    micro_bs = 10
    accum = 8
    loader = DataLoader(chunked, batch_size=micro_bs, shuffle=False)

    # Single param group → flat optimizer state indexing matches
    # list(model.named_parameters()) order, which is what
    # bergson.utils.load_from_optimizer relies on.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        betas=(0.95, 0.975),
        eps=1e-8,
        weight_decay=0.01,
    )

    total_steps = math.ceil(len(loader) / accum)
    print(f"total steps: {total_steps}", flush=True)

    # Polynomial schedule with 25% warmup, lr_start 1e-6, lr_end 8e-5 — matches MAGIC config.
    def lr_at(step: int) -> float:
        warmup_steps = max(1, int(0.25 * total_steps))
        peak_lr = 8e-4
        start_lr = 1e-6
        end_lr = 8e-5
        if step < warmup_steps:
            return start_lr + (peak_lr - start_lr) * (step / warmup_steps)
        # Polynomial decay (linear, power=1) from peak to end_lr over remaining steps.
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return peak_lr + (end_lr - peak_lr) * progress

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    accum_count = 0
    accum_loss = 0.0
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        out = model(input_ids=ids, labels=labels)
        loss = out.loss / accum
        loss.backward()
        accum_loss += loss.item()
        accum_count += 1
        if accum_count == accum:
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step % 5 == 0 or step == total_steps - 1:
                print(f"step {step:3d}/{total_steps}  loss={accum_loss:.3f}  lr={optimizer.param_groups[0]['lr']:.3e}", flush=True)
            step += 1
            accum_count = 0
            accum_loss = 0.0
    # Flush any partial accumulation
    if accum_count > 0:
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"step {step:3d}/{total_steps} (partial accum={accum_count})  loss={accum_loss:.3f}", flush=True)

    optimizer_state = optimizer.state_dict()
    out_path = OUT_DIR / "optimizer.pt"
    torch.save(optimizer_state, out_path)
    print(f"saved {out_path}", flush=True)
    print(f"  num_state_entries: {len(optimizer_state['state'])}", flush=True)
    print(f"  num_param_groups: {len(optimizer_state['param_groups'])}", flush=True)
    for idx, st in list(optimizer_state["state"].items())[:3]:
        print(f"  state[{idx}] keys={list(st.keys())} exp_avg_sq.shape={tuple(st['exp_avg_sq'].shape)}", flush=True)

    # Sanity: verify mapping aligns by checking shape against model.named_parameters() order
    named = list(model.named_parameters())
    print(f"  model has {len(named)} params total; first 3 names+shapes:", flush=True)
    for name, p in named[:3]:
        print(f"    {name}: {tuple(p.shape)}", flush=True)


if __name__ == "__main__":
    main()
