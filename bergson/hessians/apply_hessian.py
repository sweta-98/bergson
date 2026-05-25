import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from simple_parsing import ArgumentParser
from torch import Tensor

from bergson.collector.collector import create_projection_matrix
from bergson.data import create_index, load_gradients
from bergson.distributed import init_dist
from bergson.hessians.eigenvectors import (
    _compute_full_matrix,
    fair_distribute_by_cost,
)
from bergson.hessians.sharded_computation import ShardedMul
from bergson.utils.logger import get_logger
from bergson.utils.utils import get_device


@dataclass
class EkfacConfig:
    hessian_method_path: str
    gradient_path: str
    run_path: str
    ev_correction: bool
    """If True, use the corrected eigenvalues, this requires
    `hessian_method_path` to have been created with
    `HessianConfig.ev_correction=True`."""
    debug: bool = False
    lambda_damp_factor: float = 0.1
    projection_dim: int = 0
    projection_type: Literal["normal", "rademacher"] = "rademacher"


SIDE_TO_COV = {"left": "gradient", "right": "activation"}


def build_kfac_projections(
    hessian_method_path: str,
    projection_dim: int,
    projection_type: Literal["normal", "rademacher"],
    lambda_damp_factor: float,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Build and save the precondition+sketch matrices ``M = R · cov^{-1/2}``.

    For each module and each side, this composes the random projection ``R``
    with the damped inverse square-root of the K-FAC factor:

        M = R · Q · diag((E + λ·mean(E))^{-1/2}) · Qᵀ              [p, d]

    The result is saved to ``hessian_method_path/projection_{side}_sharded/``.
    Loaded later by the gradient collector — when present, gradients come
    out of collection already preconditioned and sketched in one matmul, so
    the apply step and the score step both share the same projection.

    Layers are distributed across ranks via the same fair-by-cost scheme as
    the eigendecomposition; each rank writes its own shard.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    side_dims: dict[str, dict[str, int]] = {"left": {}, "right": {}}
    for side, cov in SIDE_TO_COV.items():
        with safe_open(
            os.path.join(
                hessian_method_path, f"eigen_{cov}_sharded/shard_0.safetensors"
            ),
            framework="pt",
        ) as f:
            for name in f.keys():
                side_dims[side][name] = f.get_tensor(name).shape[-1]

    names = list(side_dims["left"].keys())

    per_layer_dim = {n: max(side_dims["left"][n], side_dims["right"][n]) for n in names}
    my_names = fair_distribute_by_cost(per_layer_dim, world_size)[rank]

    out_dirs = {
        side: os.path.join(hessian_method_path, f"projection_{side}_sharded")
        for side in ("left", "right")
    }
    if rank == 0:
        for d in out_dirs.values():
            os.makedirs(d, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    saved: dict[str, dict[str, Tensor]] = {"left": {}, "right": {}}
    for name in my_names:
        for side, cov in SIDE_TO_COV.items():
            d = side_dims[side][name]
            Q = _compute_full_matrix(
                name=name,
                shard_path=os.path.join(hessian_method_path, f"eigen_{cov}_sharded"),
                rank=rank,
                world_size=world_size,
            ).to(device=device, dtype=torch.float32)
            E = _compute_full_matrix(
                name=name,
                shard_path=os.path.join(hessian_method_path, f"eigval_{cov}_sharded"),
                rank=rank,
                world_size=world_size,
            ).to(device=device, dtype=torch.float32)

            damp = lambda_damp_factor * E.mean()
            D = (E + damp).clamp_min(torch.finfo(torch.float32).tiny).rsqrt()  # [d]

            R = create_projection_matrix(
                f"{name}/{side}",
                projection_dim,
                d,
                torch.float32,
                device,
                projection_type,
            )
            # create_projection_matrix returns unit-norm rows, so E[RᵀR] =
            # (p/d)·I. An unbiased sketch needs E[RᵀR] = I, so rescale by
            # √(d/p); otherwise each side carries a p/d factor and the score
            # picks up a per-layer p²/(d_left·d_right) reweighting that
            # corrupts ranking relative to the exact (projection_dim=0) path.
            R = R * (d / projection_dim) ** 0.5

            M = R @ (Q * D) @ Q.T
            saved[side][name] = M.to(dtype=dtype).cpu().contiguous()

    for side in ("left", "right"):
        save_file(
            saved[side],
            os.path.join(out_dirs[side], f"shard_{rank}.safetensors"),
        )

    get_logger().info(
        f"Saved M_left/M_right to {out_dirs['left']} and {out_dirs['right']}"
    )

    if dist.is_initialized():
        dist.barrier()


def load_kfac_projections(
    hessian_method_path: str | os.PathLike,
    cache: dict,
    target_names,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Populate ``cache`` with the per-layer M matrices saved by
    ``build_kfac_projections``. ``cache`` is the projection cache used by
    ``HookCollectorBase.projection()`` (keyed by ``(name, side, device)``);
    a cache hit short-circuits random projection at gradient collection.

    ``dtype`` matches what the collector will request at hook time (the
    model's gradient dtype). Saved M lives in fp32 on disk for precision
    and is cast on the way in.
    """
    targets = set(target_names)
    for side in ("left", "right"):
        side_dir = Path(hessian_method_path) / f"projection_{side}_sharded"
        if not side_dir.exists():
            return
        for shard_file in sorted(side_dir.glob("shard_*.safetensors")):
            with safe_open(str(shard_file), framework="pt", device=str(device)) as f:
                for name in f.keys():
                    if name in targets:
                        cache[(name, side, device)] = f.get_tensor(name).to(dtype=dtype)


class EkfacApplicator:
    def __init__(self, cfg: EkfacConfig, apply_fn=None):
        self.cfg = cfg
        self.path = cfg.hessian_method_path
        self.gradient_path = cfg.gradient_path
        self.apply_fn = apply_fn

        self.logger = get_logger(
            "EkfacApplicator", level="DEBUG" if cfg.debug else "INFO"
        )

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = get_device(self.rank)

        self.sharded_computer = ShardedMul()

    def compute_ivhp_sharded(self):
        if self.cfg.projection_dim > 0:
            if self.cfg.ev_correction:
                raise ValueError(
                    "K-FAC + random projection (compression) is incompatible "
                    "with `ev_correction=True`: eigenvalue correction acts on "
                    "the joint S⊗A spectrum and cannot be folded into a per-side "
                    "inverse-square-root. Use a Kronecker-factored method "
                    "(kfac, shampoo) without ev_correction."
                )
            return self._apply_compressed()
        return self._apply_legacy()

    def _apply_compressed(self):
        """Apply M_left · G_q · M_rightᵀ to the saved query gradients.

        Assumes step 3.5 (``build_kfac_projections``) has already saved M
        under ``hessian_method_path/projection_{side}_sharded/``. Output
        shape per layer is [N, p, p].
        """
        p = self.cfg.projection_dim

        M_left: dict[str, Tensor] = {}
        M_right: dict[str, Tensor] = {}
        for side, store in (("left", M_left), ("right", M_right)):
            side_dir = os.path.join(self.path, f"projection_{side}_sharded")
            for shard_file in sorted(Path(side_dir).glob("shard_*.safetensors")):
                shard = load_file(str(shard_file), device=str(self.device))
                for k, v in shard.items():
                    store[k] = v.to(dtype=torch.float32)

        mmap = load_gradients(self.gradient_path)
        with open(os.path.join(self.gradient_path, "info.json")) as f:
            info = json.load(f)

        grad_sizes = {name: p * p for name in M_left}
        grad_buffer = create_index(
            Path(self.cfg.run_path),
            num_grads=info["num_grads"],
            grad_sizes=grad_sizes,
            dtype=np.float32,
        )

        self.logger.info(
            f"Loaded gradients for {len(mmap)} queries and applying M·G·Mᵀ..."
        )

        for name, M_l in M_left.items():
            M_r = M_right[name]
            d_S, d_A = M_l.shape[1], M_r.shape[1]
            G = (
                torch.from_numpy(mmap[name][:])
                .to(device=self.device, dtype=torch.float32)
                .view(-1, d_S, d_A)
            )
            # ĝ_q = M_left · G · M_rightᵀ        [N, p, p]
            sketched = torch.einsum("ps,nsa,ra->npr", M_l, G, M_r)
            grad_buffer[name][:] = (
                sketched.to(device="cpu", non_blocking=True).flatten(1).numpy()
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        grad_buffer.flush()
        self.logger.info(f"Saved sketched IVHP gradients to {self.cfg.run_path}")

    def _apply_legacy(self):
        """Full-rank IVHP via the eigenbasis rotate-divide-rotate path."""
        eigen_a = load_file(
            self.path + f"/eigen_activation_sharded/shard_{self.rank}.safetensors",
            device=self.device,
        )
        eigen_g = load_file(
            self.path + f"/eigen_gradient_sharded/shard_{self.rank}.safetensors",
            device=self.device,
        )
        lambda_dir = (
            "eigenvalue_correction_sharded"
            if self.cfg.ev_correction
            else "eigenvalue_sharded"
        )
        lambda_factor = load_file(
            self.path + f"/{lambda_dir}/shard_{self.rank}.safetensors",
            device=self.device,
        )

        for k, v in lambda_factor.items():
            eigen_a[k] = eigen_a[k].to(dtype=torch.float32)
            eigen_g[k] = eigen_g[k].to(dtype=torch.float32)
            lambda_factor[k] = v.to(dtype=torch.float32)

        grad_sizes = {
            name: eigen_g[name].shape[1] * eigen_a[name].shape[1] for name in eigen_a
        }

        mmap = load_gradients(self.gradient_path)
        with open(os.path.join(self.gradient_path, "info.json")) as f:
            info = json.load(f)

        grad_buffer = create_index(
            Path(self.cfg.run_path),
            num_grads=info["num_grads"],
            grad_sizes=grad_sizes,
            dtype=np.float32,
        )

        self.logger.info(
            f"Loaded gradients for {len(mmap)} queries and computing IVHP..."
        )

        # Forward rotation into eigenbasis: Q_S^T @ G @ Q_A
        transformed_gradients: dict[str, Tensor] = {}
        for k, v in eigen_a.items():
            gradients_noi = torch.from_numpy(mmap[k][:]).to(
                device=self.device, dtype=torch.float32
            )
            gradients_noi = gradients_noi.view(
                -1, eigen_g[k].shape[1], eigen_a[k].shape[1]
            )
            transformed_gradients[k] = self.sharded_computer._matmul(
                vector_nsa=gradients_noi, matrix_cb=v
            )

        self.logger.debug("Finished G @ Q_A")

        for k, v in eigen_g.items():
            transformed_gradients[k] = self.sharded_computer._matmul(
                vector_nsa=transformed_gradients[k].transpose(-2, -1), matrix_cb=v
            ).transpose(-2, -1)

        self.logger.debug("Finished G' = Q_S^T @ G @ Q_A")

        # Apply eigenvalue function in eigenbasis (default = damped inverse).
        for k, v in lambda_factor.items():
            if self.apply_fn is None:
                self.sharded_computer._hadamard(
                    matrix_noi=transformed_gradients[k],
                    lambda_ci=v,
                    lambda_damp_factor=self.cfg.lambda_damp_factor,
                )
            else:
                self.sharded_computer._apply_eigfn(
                    matrix_noi=transformed_gradients[k],
                    lambda_ci=v,
                    fn=self.apply_fn,
                )

        self.logger.debug("Finished G' / lambda")
        del lambda_factor
        gc.collect()

        # Rotate back to parameter space: Q_S @ G' @ Q_A^T
        for k, v in eigen_g.items():
            transformed_gradients[k] = self.sharded_computer._transpose_matmul(
                vector_nsa=transformed_gradients[k].transpose(-2, -1), matrix_cb=v
            ).transpose(-2, -1)

        self.logger.debug("Finished Q_S @ G'")
        del eigen_g
        gc.collect()

        for k, v in eigen_a.items():
            transformed_gradients[k] = self.sharded_computer._transpose_matmul(
                vector_nsa=transformed_gradients[k], matrix_cb=v
            )

        self.logger.debug("Finished H^{-1} G = Q_S @ (G' / lambda) @ Q_A^T")
        del eigen_a
        gc.collect()

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        for k, v in transformed_gradients.items():
            grad_buffer[k][:] = v.to(device="cpu", non_blocking=True).flatten(1).numpy()

        grad_buffer.flush()

        self.logger.info(f"Saved IVHP gradients to {self.cfg.run_path}")


def apply_worker(
    rank: int,  # global
    local_rank: int,  # local
    world_size: int,
    cfg: EkfacConfig,
):
    """Worker function for distributed IVHP computation."""
    init_dist(rank, local_rank, world_size)

    applicator = EkfacApplicator(cfg)
    applicator.compute_ivhp_sharded()


def build_projections_worker(
    rank: int,  # global
    local_rank: int,  # local
    world_size: int,
    cfg: EkfacConfig,
):
    """Worker for Step 3.5: build ``M = R · cov^{-1/2}`` and save to disk."""
    init_dist(rank, local_rank, world_size)

    build_kfac_projections(
        cfg.hessian_method_path,
        projection_dim=cfg.projection_dim,
        projection_type=cfg.projection_type,
        lambda_damp_factor=cfg.lambda_damp_factor,
        dtype=torch.float32,
        device=get_device(rank),
    )


if __name__ == "__main__":
    from bergson.config import DistributedConfig
    from bergson.distributed import launch_distributed_run

    parser = ArgumentParser()
    parser.add_arguments(EkfacConfig, dest="cfg")
    args = parser.parse_args()

    launch_distributed_run(
        "apply_hessian",
        apply_worker,
        [args.cfg],
        DistributedConfig(),
    )
