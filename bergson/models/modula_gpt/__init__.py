"""HuggingFace-compatible wrapper around a torch-Modula GPT.

Pairs with bergson's `optimizer: modula` branch (see
`bergson.magic.modula_optim`) to replicate the training loop from
modula-torch examples/train-gpt.py (pre-JAX commit b3a89a2 of
github.com/modula-systems/modula).

Typical usage:

    # 1. Build and save a random-init checkpoint
    python -m bergson.models.modula_gpt.init_modula_gpt ./my_modula_init

    # 2. Point a bergson MAGIC config at it
    # config.yaml:
    #   model: ./my_modula_init
    #   model_kwargs: "trust_remote_code=True"
    #   optimizer: modula
    bergson magic config.yaml
"""

from .modeling_modula_gpt import ModulaGPTConfig, ModulaGPTForCausalLM

__all__ = ["ModulaGPTConfig", "ModulaGPTForCausalLM"]
