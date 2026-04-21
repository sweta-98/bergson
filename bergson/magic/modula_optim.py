"""Functional torchopt optimizer matching modula's Adam-with-normalize loop.

Intended to exactly replicate the training step from modula-torch
examples/train-gpt.py:

    mom1 += (1-beta1)**(step/(step+1)) * (grad    - mom1)
    mom2 += (1-beta2)**(step/(step+1)) * (grad**2 - mom2)
    update = mom1 / mom2**0.5
    update.zero_nans()
    gpt.normalize(update, target_norm = init_lr * schedule)
    weights -= update
    gpt.regularize(weights, strength = init_lr * schedule * wd)

Mapping to this file:
    1. Reference's Adam EMA uses a time-varying coefficient
       (1-beta)**(step/(step+1)) rather than a fixed beta with bias
       correction; we use the identical formula (no bias correction,
       no eps).
    2. Per-atom spectral normalize: `_Linear.normalize` one-step power
       iteration with persistent `u`; `_Embedding.normalize` per-row.
    3. Linear regularize: `weight *= 1 - strength` — applied post-step
       as a projection (see modula_postproject).
    4. Embedding regularize: `weight /= weight.norm(dim=1)` — always
       unit-row-normalizes, independent of `strength`; applied post-step.
    5. Mass-0 atoms (target_norm==0) have their update zeroed, matching
       `CompositeModule.normalize`'s `w *= 0` branch.

The LR schedule is applied by scale_by_neg_lr in the chain between the
EMA stage and the projection stage, so we hand the projection stage the
actual weight delta `-lr * schedule * update` and ask it to correct for
the regularize-on-weights semantics.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torchopt.alias.utils import _get_use_chain_flat, scale_by_neg_lr
from torchopt.combine import chain
from torchopt.typing import GradientTransformation, OptState, Params, ScalarOrSchedule, Updates


EPS = 1e-12


class ModulaState(NamedTuple):
    exp_avgs: list
    exp_avg_sqs: list
    us: list  # persistent power-iteration buffer per 2D Linear atom; None otherwise
    step: Tensor


class ModulaPostState(NamedTuple):
    # stateless; kept so torchopt's pytree tracking has something to hold
    step: Tensor


def _to_list(x: Params) -> list:
    return list(x.values()) if hasattr(x, "values") else list(x)


def _resolve_spec(
    x: Params,
    atom_spec: dict[str, tuple[str, float]],
) -> list[tuple[str, float]]:
    """Return a positional list of (kind, target_norm) aligned with `x`.

    When `x` is a dict (the common case from bergson's Trainer) we look
    up each parameter's spec entry by name, which is robust to reordering.
    When `x` arrives as a positional list (e.g. after torchopt.chain.flat
    flattens) we fall back to the insertion order of `atom_spec` — which
    matches `named_parameters()` order as emitted by
    `ModulaGPTForCausalLM.modula_optim_spec()`.
    """
    default = ("unknown", 0.0)
    if hasattr(x, "keys"):
        return [atom_spec.get(k, default) for k in x.keys()]
    values = list(atom_spec.values())
    n = len(x) if hasattr(x, "__len__") else None
    if n is not None and n != len(values):
        return values + [default] * max(0, n - len(values))
    return values


def _modula_ema_and_normalize(
    atom_spec: dict[str, tuple[str, float]],
    *,
    beta1: float,
    beta2: float,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
    """Stage 1: reference-matching Adam EMA + per-atom normalize.

    Produces unit-target-norm updates (pre-LR); the chain's scale_by_neg_lr
    multiplies by -lr afterwards.
    """

    def init_fn(params: Params) -> OptState:
        param_list = _to_list(params)
        spec = _resolve_spec(params, atom_spec)
        exp_avgs, exp_avg_sqs, us = [], [], []
        for i, p in enumerate(param_list):
            exp_avgs.append(
                torch.zeros_like(p, requires_grad=moment_requires_grad)
            )
            exp_avg_sqs.append(
                torch.zeros_like(p, requires_grad=moment_requires_grad)
            )
            kind = spec[i][0] if i < len(spec) else "unknown"
            if kind == "linear" and p.ndim == 2:
                us.append(
                    torch.randn(
                        p.shape[1], device=p.device, dtype=p.dtype,
                        requires_grad=False,
                    )
                )
            else:
                us.append(None)
        return ModulaState(
            exp_avgs=exp_avgs,
            exp_avg_sqs=exp_avg_sqs,
            us=us,
            step=torch.tensor(0, dtype=torch.long),
        )

    def update_fn(
        updates: Params,
        state: OptState,
        *,
        params: Params | None = None,
        inplace: bool = False,
    ) -> tuple[Updates, OptState]:
        del params, inplace  # unused in this stage
        update_list = _to_list(updates)
        spec = _resolve_spec(updates, atom_spec)

        # Reference train-gpt.py uses 0-indexed `step` in
        # `(1-beta)**(step/(step+1))`. state.step starts at 0 and is
        # incremented at the END of the call, so on the first call we
        # want step0 = 0 ⇒ coefficient = (1-beta)^0 = 1 ⇒ mom := grad.
        step0 = state.step
        step0_f = step0.to(dtype=torch.float32)
        exponent = step0_f / (step0_f + 1)
        coef1 = (1 - beta1) ** exponent
        coef2 = (1 - beta2) ** exponent

        new_exp_avgs, new_exp_avg_sqs, new_us, new_update_list = [], [], [], []

        for i, (update, exp_avg, exp_avg_sq, u) in enumerate(
            zip(update_list, state.exp_avgs, state.exp_avg_sqs, state.us)
        ):
            kind, target_norm = spec[i] if i < len(spec) else ("unknown", 0.0)

            if update is None:
                new_exp_avgs.append(exp_avg)
                new_exp_avg_sqs.append(exp_avg_sq)
                new_us.append(u)
                new_update_list.append(None)
                continue

            # Reference EMA: mom += coef * (grad - mom). No bias correction.
            # No `inplace` fast-path: bergson's trace-mode second backward
            # needs the EMA to keep a grad_fn, which in-place ops break.
            new_ea = exp_avg + coef1 * (update - exp_avg)
            new_eas = exp_avg_sq + coef2 * (update * update - exp_avg_sq)

            # Reference: `update = mom1 / mom2**0.5; zero_nans()`. For
            # single-backward training zero_nans is enough, but bergson's
            # second-backward VJP evaluates d(sqrt)/dx at x=0 (which is
            # infinite) for any zero-gradient row — e.g. unused embedding
            # rows — producing NaN scores. Add a tiny eps under the sqrt
            # so the derivative stays finite; values match to precision.
            adam_update = new_ea / (new_eas + EPS).sqrt()
            adam_update = torch.nan_to_num(adam_update, nan=0.0, posinf=0.0, neginf=0.0)

            if target_norm == 0.0:
                # Mass-0 subtree: `w *= 0` in CompositeModule.normalize.
                adam_update = torch.zeros_like(adam_update)
                new_us.append(u)
            elif kind == "linear" and adam_update.ndim == 2:
                # One power-iteration step. Matches _Linear.normalize:
                #   v = W u; v /= |v|; u = W^T v; W *= target_norm/|u|
                u_cur = u
                if u_cur is None or u_cur.shape[0] != adam_update.shape[1]:
                    u_cur = torch.randn(
                        adam_update.shape[1],
                        device=adam_update.device, dtype=adam_update.dtype,
                    )
                v = torch.mv(adam_update, u_cur)
                v = v / v.norm().clamp(min=EPS)
                new_u = torch.mv(adam_update.t(), v)
                u_norm = new_u.norm().clamp(min=EPS)
                adam_update = adam_update * (target_norm / u_norm)
                new_us.append(new_u)
            elif kind == "embedding" and adam_update.ndim == 2:
                # Per-row normalize. Matches _Embedding.normalize:
                #   weight *= target_norm / |weight rows|
                row_norms = adam_update.norm(dim=1, keepdim=True).clamp(min=EPS)
                adam_update = adam_update * (target_norm / row_norms)
                new_us.append(u)
            else:
                new_us.append(u)

            new_exp_avgs.append(new_ea)
            new_exp_avg_sqs.append(new_eas)
            new_update_list.append(adam_update)

        if hasattr(updates, "keys"):
            new_updates = dict(zip(updates.keys(), new_update_list))
        else:
            new_updates = new_update_list

        return new_updates, ModulaState(
            exp_avgs=new_exp_avgs,
            exp_avg_sqs=new_exp_avg_sqs,
            us=new_us,
            step=step0 + 1,
        )

    return GradientTransformation(init_fn, update_fn)


def _modula_postproject(
    atom_spec: dict[str, tuple[str, float]],
    lr_schedule: ScalarOrSchedule,
    weight_decay: float,
) -> GradientTransformation:
    """Stage 3: post-step projection matching `gpt.regularize(weights, ...)`.

    Runs AFTER scale_by_neg_lr, so `updates` is the actual weight delta
    (`-lr * target_norm * direction`). For each atom, compute what the
    weight would become after the delta, apply the atom's regularize to
    that hypothetical weight, and return the adjusted delta.

    - Linear atoms: weight *= (1 - lr_schedule * weight_decay)
    - Embedding atoms: weight /= row_norm (always unit row norm)
    - Mass-0 atoms: no-op (their delta is already zero from stage 1)
    """

    def init_fn(params: Params) -> OptState:
        return ModulaPostState(step=torch.tensor(0, dtype=torch.long))

    def update_fn(
        updates: Params,
        state: OptState,
        *,
        params: Params | None = None,
        inplace: bool = False,
    ) -> tuple[Updates, OptState]:
        assert params is not None, "modula_postproject needs params to project"
        del inplace
        param_list = _to_list(params)
        update_list = _to_list(updates)
        spec = _resolve_spec(updates, atom_spec)

        step = state.step + 1
        if callable(lr_schedule):
            lr_here = float(lr_schedule(step.item() - 1))
        else:
            lr_here = float(lr_schedule)
        linear_shrink = 1.0 - lr_here * weight_decay

        new_update_list = []
        for i, (delta, param) in enumerate(zip(update_list, param_list)):
            if delta is None:
                new_update_list.append(None)
                continue
            kind, target_norm = spec[i] if i < len(spec) else ("unknown", 0.0)

            if target_norm == 0.0 or param.ndim != 2:
                new_update_list.append(delta)
                continue

            post_update = param.detach() + delta

            if kind == "linear":
                desired = post_update * linear_shrink
            elif kind == "embedding":
                row_norms = post_update.norm(dim=1, keepdim=True).clamp(min=EPS)
                desired = post_update / row_norms
            else:
                new_update_list.append(delta)
                continue

            new_update_list.append(desired - param.detach())

        if hasattr(updates, "keys"):
            new_updates = dict(zip(updates.keys(), new_update_list))
        else:
            new_updates = new_update_list

        return new_updates, ModulaPostState(step=step)

    return GradientTransformation(init_fn, update_fn)


def modula_adamw(
    lr: ScalarOrSchedule,
    atom_spec: dict[str, tuple[str, float]],
    beta1: float = 0.9,
    beta2: float = 0.99,
    eps: float = 0.0,         # kept for bergson config symmetry; ignored
    eps_root: float = 0.0,    # ditto
    weight_decay: float = 0.0,
    *,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
    """Modula-faithful functional optimizer.

    Args:
        lr: learning rate (scalar or schedule).
        atom_spec: dict keyed by parameter name (as in
            `model.named_parameters()`), mapping to
            `(atom_kind, target_norm_scale)`. Produced by
            `ModulaGPTForCausalLM.modula_optim_spec()`.
            atom_kind ∈ {"linear", "embedding", "unknown"};
            target_norm=0 marks a mass-0 atom (update is zeroed).
            Any parameter missing from the dict is treated as
            ("unknown", 0.0), i.e. its update passes through the
            Adam EMA but is then zeroed at the normalize stage —
            the caller should populate the dict for every trainable
            parameter.
        beta1, beta2: Adam EMA betas. Reference train-gpt.py uses
            (0.9, 0.99); these are the defaults here. (bergson's config
            default is (0.95, 0.975); pass adam_beta1/adam_beta2 in the
            YAML to override.)
        weight_decay: coefficient for the Linear atom's
            `weight *= 1 - lr_schedule * weight_decay` regularize.
            Embedding regularize is unconditional (always unit-row-norm).
        eps, eps_root: ignored. The reference relies on `zero_nans()`.
    """
    del eps, eps_root
    if not (callable(lr) or lr >= 0.0):
        raise ValueError(f"Invalid learning rate: {lr}")

    chain_fn = chain
    scale_by_neg_lr_fn = scale_by_neg_lr
    if _get_use_chain_flat():
        chain_fn = chain_fn.flat  # type: ignore[attr-defined]
        scale_by_neg_lr_fn = scale_by_neg_lr_fn.flat  # type: ignore[attr-defined]

    return chain_fn(
        _modula_ema_and_normalize(
            atom_spec,
            beta1=beta1,
            beta2=beta2,
            moment_requires_grad=moment_requires_grad,
        ),
        scale_by_neg_lr_fn(lr),
        _modula_postproject(atom_spec, lr_schedule=lr, weight_decay=weight_decay),
    )
