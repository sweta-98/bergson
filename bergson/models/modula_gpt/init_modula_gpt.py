"""Materialize a random-init ModulaGPT checkpoint ready for bergson magic.

The saved directory includes a copy of modeling_modula_gpt.py so bergson
(and any other HF consumer) can load the model via
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
without depending on bergson being import-visible at load time.

Default shape matches the magic_scaling_gpt2 research rig:
  4 layers, 128 embedding, 4 heads, 512 context, GPT-2 BPE vocab.

Usage:
    python -m bergson.models.modula_gpt.init_modula_gpt ./my_modula_init
    python -m bergson.models.modula_gpt.init_modula_gpt ./my_modula_init \\
        --seed 42 --d-embed 256 --num-blocks 6
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

from .modeling_modula_gpt import ModulaGPTConfig, ModulaGPTForCausalLM

THIS_DIR = Path(__file__).resolve().parent
MODELING_SRC = THIS_DIR / "modeling_modula_gpt.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Directory to save the init model to.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-embed", type=int, default=128)
    parser.add_argument("--d-query", type=int, default=32)
    parser.add_argument("--d-value", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--blocks-mass", type=float, default=5.0)
    parser.add_argument(
        "--tokenizer", type=str, default="gpt2",
        help="HF tokenizer id to copy into the output dir (default: gpt2)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    config = ModulaGPTConfig(
        vocab_size=args.vocab_size,
        context=args.context,
        num_heads=args.num_heads,
        d_embed=args.d_embed,
        d_query=args.d_query,
        d_value=args.d_value,
        num_blocks=args.num_blocks,
        blocks_mass=args.blocks_mass,
    )
    # `auto_map` tells HF's trust_remote_code loader which classes to
    # look up in the local modeling_modula_gpt.py copy we bundle below.
    config.auto_map = {
        "AutoConfig": "modeling_modula_gpt.ModulaGPTConfig",
        "AutoModelForCausalLM": "modeling_modula_gpt.ModulaGPTForCausalLM",
    }

    model = ModulaGPTForCausalLM(config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built ModulaGPT:")
    print(f"  total params:  {n_params:>12,}")
    print(f"  atoms:         {len(model.atoms):>12,}")
    for i, a in enumerate(model.atoms):
        shape = tuple(model.weights[i].shape)
        print(f"    [{i:>2}] {type(a).__name__:<12} shape={shape}  mass={a.mass}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    shutil.copy2(MODELING_SRC, args.out_dir / "modeling_modula_gpt.py")

    tok = GPT2TokenizerFast.from_pretrained(args.tokenizer)
    tok.save_pretrained(args.out_dir)

    print(f"\nSaved random-init ModulaGPT to {args.out_dir}")
    print(f"  (load with AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True))")


if __name__ == "__main__":
    main()
