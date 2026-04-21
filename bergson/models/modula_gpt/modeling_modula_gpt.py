"""Self-contained HF-compatible wrapper around the torch-Modula GPT.

Vendors Modula's torch implementation (pre-JAX commit b3a89a2 of
github.com/modula-systems/modula) inline so the saved model directory
can be loaded by `AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)` without any import dependency on the Modula
repo itself.

Exposes:
    ModulaGPTConfig       - PretrainedConfig subclass
    ModulaGPTForCausalLM  - PreTrainedModel subclass with a
                            `modula_optim_spec()` method consumed by
                            bergson.magic.modula_optim.modula_adamw.

Pair with bergson's `optimizer: modula` (in config.yaml). With a stock
AdamW/Muon/SGD optimizer the weights will train fine but Modula's
per-atom normalize + regularize discipline won't be applied, and the
attribution experiments in the magic_scaling_gpt2 repo will not
reproduce.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


# ---------------------------------------------------------------------
# Vendored modula.vector — minimal Vector used by compound.forward
# ---------------------------------------------------------------------


class _Vector:
    """A list of tensors supporting indexing/slicing for compound forwards.

    Only the subset used during forward is needed. Slicing returns a
    new _Vector so recursive compound.forward works.
    """

    def __init__(self, tensor_or_list=()):
        if isinstance(tensor_or_list, torch.Tensor):
            self.tensor_list = [tensor_or_list]
        elif isinstance(tensor_or_list, (list, tuple)):
            self.tensor_list = list(tensor_or_list)
        else:
            raise TypeError(type(tensor_or_list))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return _Vector(self.tensor_list[item])
        return self.tensor_list[item]

    def __len__(self):
        return len(self.tensor_list)

    def __iter__(self):
        return iter(self.tensor_list)

    def __and__(self, other):
        return _Vector(self.tensor_list + other.tensor_list)


# ---------------------------------------------------------------------
# Vendored modula.abstract — Module hierarchy
# ---------------------------------------------------------------------


class _Module:
    def __init__(self):
        self.mass = None
        self.sensitivity = None
        self.length = None
        self.children = []

    def forward(self, x, w):
        raise NotImplementedError

    def initialize(self, device, dtype):
        raise NotImplementedError

    def tare(self, absolute=1, relative=None):
        if relative is not None:
            self.mass *= relative
            for child in self.children:
                child.tare(relative=relative)
        else:
            self.tare(relative=absolute / self.mass)

    def __call__(self, x, w):
        return self.forward(x, w)

    def __matmul__(self, other):
        if isinstance(other, tuple):
            other = _TupleModule(other)
        return _CompositeModule(self, other)

    def __rmatmul__(self, other):
        if isinstance(other, tuple):
            other = _TupleModule(other)
        return other @ self

    def __add__(self, other):
        return _Add() @ (self, other)

    def __mul__(self, other):
        assert other != 0
        return self @ _Mul(other)

    def __rmul__(self, other):
        assert other != 0
        return _Mul(other) @ self

    def __truediv__(self, other):
        assert other != 0
        return self * (1 / other)

    def __pow__(self, other):
        assert other >= 0 and other % 1 == 0
        if other > 0:
            return copy.deepcopy(self) @ self ** (other - 1)
        return _Mul(1.0)


class _CompositeModule(_Module):
    def __init__(self, m1, m0):
        super().__init__()
        self.children = (m0, m1)
        self.length = m0.length + m1.length
        self.mass = m0.mass + m1.mass
        self.sensitivity = m1.sensitivity * m0.sensitivity

    def forward(self, x, w):
        m0, m1 = self.children
        w0 = w[: m0.length]
        w1 = w[m0.length :]
        return m1.forward(m0.forward(x, w0), w1)

    def initialize(self, device, dtype=torch.float32):
        m0, m1 = self.children
        return m0.initialize(device, dtype=dtype) & m1.initialize(device, dtype=dtype)


class _TupleModule(_Module):
    def __init__(self, tuple_of_modules):
        super().__init__()
        self.children = tuple_of_modules
        self.length = sum(c.length for c in self.children)
        self.mass = sum(c.mass for c in self.children)
        self.sensitivity = sum(c.sensitivity for c in self.children)

    def forward(self, x, w):
        output = []
        for child in self.children:
            w_child = w[: child.length]
            output.append(child.forward(x, w_child))
            w = w[child.length :]
        return output

    def initialize(self, device, dtype=torch.float32):
        vec = _Vector()
        for c in self.children:
            vec = vec & c.initialize(device, dtype=dtype)
        return vec


class _Mul(_Module):
    def __init__(self, alpha):
        super().__init__()
        self.mass = 0
        self.sensitivity = abs(alpha)
        self.length = 0
        self.initialize = lambda device, dtype=torch.float32: _Vector()
        self.alpha = alpha

    def forward(self, x, _):
        if isinstance(x, list):
            return [self.forward(xi, _) for xi in x]
        return self.alpha * x


class _Add(_Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.length = 0
        self.initialize = lambda device, dtype=torch.float32: _Vector()
        self.forward = lambda x, w: sum(x)


# ---------------------------------------------------------------------
# Vendored modula.atom — weight-bearing atoms
# ---------------------------------------------------------------------


class _Linear(_Module):
    def __init__(self, out_features, in_features, mass=1):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1
        self.length = 1
        self.out_features = out_features
        self.in_features = in_features
        self.scale = math.sqrt(out_features / in_features)

    def forward(self, x, w):
        return self.scale * F.linear(x, w[0])

    def initialize(self, device, dtype=torch.float32):
        weight = torch.empty(
            (self.out_features, self.in_features),
            device=device,
            requires_grad=True,
        )
        torch.nn.init.orthogonal_(weight)
        weight.data = weight.data.to(dtype=dtype)
        return _Vector(weight)


class _Embedding(_Module):
    def __init__(self, num_embedding, embedding_dim, mass=1):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1
        self.length = 1
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.scale = math.sqrt(embedding_dim)

    def forward(self, x, w):
        return self.scale * F.embedding(x, w[0])

    def initialize(self, device, dtype=torch.float32):
        weight = torch.empty(
            (self.num_embedding, self.embedding_dim),
            device=device,
            requires_grad=True,
        )
        torch.nn.init.normal_(weight)
        weight.data /= weight.norm(dim=1, keepdim=True)
        weight.data = weight.data.to(dtype=dtype)
        return _Vector(weight)


# ---------------------------------------------------------------------
# Vendored modula.bond — stateless bonds
# ---------------------------------------------------------------------


class _Bond(_Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.length = 0
        self.initialize = lambda device, dtype=torch.float32: _Vector()


class _Identity(_Bond):
    def __init__(self):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x


class _AddHeads(_Bond):
    def __init__(self, num_heads):
        super().__init__()
        self.sensitivity = 1
        self.num_heads = num_heads

    def forward(self, x, w):
        B, T, C = x.size()
        return x.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)


class _RemoveHeads(_Bond):
    def __init__(self):
        super().__init__()
        self.sensitivity = 1

    def forward(self, x, w):
        B, nh, T, hs = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, nh * hs)


class _Enumerate(_Bond):
    def __init__(self):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: torch.arange(
            0, x.size()[1], dtype=torch.long, device=x.device,
        )


class _GELU(_Bond):
    def __init__(self):
        super().__init__()
        self.sensitivity = 1 / math.sqrt(2)
        self.forward = lambda x, w: F.gelu(x)


def _ScaledGELU():
    return math.sqrt(2) * _GELU()


class _MeanSubtract(_Bond):
    def __init__(self, dim=-1):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x - x.mean(dim=dim, keepdim=True)


class _RMSDivide(_Bond):
    def __init__(self, dim=-1):
        super().__init__()
        self.sensitivity = 1
        self.forward = lambda x, w: x / x.square().mean(dim=dim, keepdim=True).sqrt()


def _LayerNorm(dim=-1):
    return _RMSDivide(dim) @ _MeanSubtract(dim)


class _FunctionalAttention(_Bond):
    def __init__(self, causal):
        super().__init__()
        self.sensitivity = 1
        self.causal = causal

    def forward(self, x, w):
        q, k, v = x
        # Bergson traces training steps with create_graph=True
        # (double-backward); efficient/flash SDPA kernels don't support
        # that, so force the math kernel.
        with sdpa_kernel([SDPBackend.MATH]):
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=self.causal, scale=1 / q.shape[-1],
            )


# ---------------------------------------------------------------------
# Vendored modula.compound — GPT builder
# ---------------------------------------------------------------------


def _build_modula_gpt(
    vocab_size, context, num_heads, d_embed, d_query, d_value,
    num_blocks, blocks_mass=5,
):
    def Attention(nh, de, dq, dv, ctx, causal):
        Q = _AddHeads(nh) @ _Linear(nh * dq, de)
        K = _AddHeads(nh) @ _Linear(nh * dq, de)
        V = _AddHeads(nh) @ _Linear(nh * dv, de)
        W = _Linear(de, dv * nh) @ _RemoveHeads()
        return W @ _FunctionalAttention(causal) * (1 / 3) @ (Q, K, V)

    token_embedding = _Embedding(vocab_size, d_embed)
    position_embedding = _Embedding(context, d_embed) @ _Enumerate()
    initial = (1 / 2) * token_embedding + (1 / 2) * position_embedding
    initial.tare()

    attention = (
        Attention(num_heads, d_embed, d_query, d_value, context, causal=True)
        @ _LayerNorm()
    )
    mlp = (
        _Linear(d_embed, 4 * d_embed)
        @ _ScaledGELU()
        @ _Linear(4 * d_embed, d_embed)
        @ _LayerNorm()
    )
    attention_block = (1 - 1 / (2 * num_blocks)) * _Identity() + (
        1 / (2 * num_blocks)
    ) * attention
    mlp_block = (1 - 1 / (2 * num_blocks)) * _Identity() + (
        1 / (2 * num_blocks)
    ) * mlp
    blocks = (mlp_block @ attention_block) ** num_blocks
    blocks.tare(absolute=blocks_mass)

    final = _Linear(vocab_size, d_embed) @ _LayerNorm()

    root = final @ blocks @ initial
    root._token_embedding = token_embedding
    return root


def _collect_atoms(root):
    """Return weight-bearing atoms in the same order as root.initialize()."""
    atoms = []

    def visit(mod):
        if isinstance(mod, (_Linear, _Embedding)):
            atoms.append(mod)
            return
        if isinstance(mod, _CompositeModule):
            m0, m1 = mod.children
            visit(m0)
            visit(m1)
        elif isinstance(mod, _TupleModule):
            for c in mod.children:
                visit(c)
        # bonds have no weights, skip

    visit(root)
    return atoms


def _compute_atom_target_norms(root, target_norm: float = 1.0):
    """Simulate modula's compound.normalize() recursion once to bake per-atom
    mass-weighted target-norm scales. Returns list ordered like `_collect_atoms`.
    """
    scales: dict[int, float] = {}

    def recurse(mod, tn: float):
        if isinstance(mod, (_Linear, _Embedding)):
            scales[id(mod)] = tn
            return
        if isinstance(mod, _CompositeModule):
            if mod.mass > 0:
                m0, m1 = mod.children
                m1_sens = m1.sensitivity if m1.sensitivity else 1.0
                recurse(m0, m0.mass / mod.mass * tn / m1_sens)
                recurse(m1, m1.mass / mod.mass * tn)
        elif isinstance(mod, _TupleModule):
            if mod.mass > 0:
                for c in mod.children:
                    recurse(c, c.mass / mod.mass * tn)

    recurse(root, float(target_norm))
    return [scales.get(id(a), 0.0) for a in _collect_atoms(root)]


# ---------------------------------------------------------------------
# HuggingFace-compatible wrapper
# ---------------------------------------------------------------------


class ModulaGPTConfig(PretrainedConfig):
    model_type = "modula_gpt"

    def __init__(
        self,
        vocab_size: int = 50257,
        context: int = 512,
        num_heads: int = 4,
        d_embed: int = 128,
        d_query: int = 32,
        d_value: int = 32,
        num_blocks: int = 4,
        blocks_mass: float = 5.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.context = context
        self.num_heads = num_heads
        self.d_embed = d_embed
        self.d_query = d_query
        self.d_value = d_value
        self.num_blocks = num_blocks
        self.blocks_mass = blocks_mass
        super().__init__(**kwargs)


class _EmbeddingProxy(nn.Module):
    """Surfaces the token-embedding weight as `.weight` for bergson's
    `model.get_input_embeddings().requires_grad_(True)` path.
    """

    def __init__(self, param: nn.Parameter):
        super().__init__()
        self.weight = param


class ModulaGPTForCausalLM(PreTrainedModel):
    config_class = ModulaGPTConfig
    base_model_prefix = "modula_gpt"
    supports_gradient_checkpointing = False
    _no_split_modules: list[str] = []
    _tied_weights_keys: list[str] = []
    all_tied_weights_keys: dict = {}

    def __init__(self, config: ModulaGPTConfig):
        super().__init__(config)
        self.modula_gpt = _build_modula_gpt(
            vocab_size=config.vocab_size,
            context=config.context,
            num_heads=config.num_heads,
            d_embed=config.d_embed,
            d_query=config.d_query,
            d_value=config.d_value,
            num_blocks=config.num_blocks,
            blocks_mass=config.blocks_mass,
        )
        weight_vec = self.modula_gpt.initialize(device="cpu")
        self.weights = nn.ParameterList(
            [nn.Parameter(t.detach().clone()) for t in weight_vec]
        )
        self.atoms = _collect_atoms(self.modula_gpt)
        assert len(self.atoms) == len(self.weights), (
            f"atom/weight count mismatch: {len(self.atoms)} vs {len(self.weights)}"
        )
        self._token_emb_idx = self.atoms.index(self.modula_gpt._token_embedding)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        example_weight: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        vec = _Vector(list(self.weights))
        logits = self.modula_gpt.forward(input_ids, vec)

        loss = None
        if labels is not None:
            loss_fn = getattr(self, "loss_function", None)
            if loss_fn is not None:
                loss = loss_fn(
                    logits=logits,
                    labels=labels,
                    example_weight=example_weight,
                    vocab_size=self.config.vocab_size,
                )
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

        return CausalLMOutput(loss=loss, logits=logits)

    def get_input_embeddings(self):
        return _EmbeddingProxy(self.weights[self._token_emb_idx])

    def set_input_embeddings(self, value):
        self.weights[self._token_emb_idx] = value.weight

    def gradient_checkpointing_enable(self, *args, **kwargs):
        pass  # no-op; the small model doesn't need it

    def modula_optim_spec(self) -> dict[str, tuple[str, float]]:
        """Spec consumed by `bergson.magic.modula_optim.modula_adamw`.

        Returns a dict keyed by the parameter name as it appears in
        `self.named_parameters()` (e.g. "weights.0"), mapping to
        `(atom_kind, target_norm_scale)`:
          - atom_kind: "linear" or "embedding"
          - target_norm_scale: mass-ratio-weighted share of a unit
            target norm, computed once at model-build time. A value of
            0.0 marks a mass-0 atom whose update should be zeroed.
        """
        target_norms = _compute_atom_target_norms(self.modula_gpt, target_norm=1.0)
        atom_infos = []
        for atom, tn in zip(self.atoms, target_norms):
            if isinstance(atom, _Linear):
                kind = "linear"
            elif isinstance(atom, _Embedding):
                kind = "embedding"
            else:
                kind = "unknown"
            atom_infos.append((kind, float(tn)))

        weight_ids = {id(p) for p in self.weights}
        names_in_order: list[str] = []
        for name, p in self.named_parameters():
            if id(p) in weight_ids:
                names_in_order.append(name)
        if len(names_in_order) != len(atom_infos):
            raise RuntimeError(
                f"Found {len(names_in_order)} weight params but {len(atom_infos)} atoms"
            )
        return {name: info for name, info in zip(names_in_order, atom_infos)}
