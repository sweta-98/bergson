import gc
import json
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from simple_parsing import ArgumentParser
from torch import Tensor

from bergson.data import create_index, load_gradients
from bergson.hessians.sharded_computation import ShardedMul
from bergson.utils.logger import get_logger
from bergson.utils.utils import get_device, get_device_index


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


class EkfacApplicator:
    def __init__(self, cfg: EkfacConfig):
        self.cfg = cfg
        self.path = cfg.hessian_method_path
        self.gradient_path = cfg.gradient_path

        self.logger = get_logger(
            "EkfacApplicator", level="DEBUG" if cfg.debug else "INFO"
        )

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = get_device(self.rank)

        self.sharded_computer = ShardedMul()

    def compute_ivhp_sharded(self):
        if os.path.isdir(
            os.path.join(self.path, "whitening_projection_activation_sharded")
        ):
            if self.cfg.ev_correction:
                raise ValueError(
                    "ev_correction=True is incompatible with precomputed "
                    "whitening-projection matrices (the EK-FAC eigenvalue "
                    "correction does not decompose into per-side row/column ops "
                    "and so was disallowed at hessian-build time)."
                )
            return self._compute_ivhp_with_whitening_projection()

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

        # Divide by damped eigenvalues in eigenbasis
        for k, v in lambda_factor.items():
            self.sharded_computer._hadamard(
                matrix_noi=transformed_gradients[k],
                lambda_ci=v,
                lambda_damp_factor=self.cfg.lambda_damp_factor,
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

        torch.cuda.synchronize()
        for k, v in transformed_gradients.items():
            grad_buffer[k][:] = v.to(device="cpu", non_blocking=True).flatten(1).numpy()

        grad_buffer.flush()

        self.logger.info(f"Saved IVHP gradients to {self.cfg.run_path}")

    def _compute_ivhp_with_whitening_projection(self):
        """IVHP via precomputed `M = (cov + αI)^{-1} P^T` matrices.

        For each layer, compute ``M_S^T G M_A`` of shape ``[N, p, p]`` in two
        sharded matmuls — the per-side damped curvature inverse and the
        random projection are both already folded into the ``M`` matrices
        (built by ``compute_whitening_projection_matrices`` at hessian time),
        so no eigenbasis rotation or eigenvalue divide is needed here.
        """
        M_a = load_file(
            self.path
            + f"/whitening_projection_activation_sharded/shard_{self.rank}.safetensors",
            device=self.device,
        )
        M_g = load_file(
            self.path
            + f"/whitening_projection_gradient_sharded/shard_{self.rank}.safetensors",
            device=self.device,
        )

        for k in M_a:
            M_a[k] = M_a[k].to(dtype=torch.float32)
            M_g[k] = M_g[k].to(dtype=torch.float32)

        # Per-layer output is [p_S, p_A] flattened; M shards are [d/W, p].
        grad_sizes = {name: M_g[name].shape[1] * M_a[name].shape[1] for name in M_a}

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
            f"Loaded gradients for {len(mmap)} queries and computing IVHP "
            "via whitening-projection matrices..."
        )

        transformed_gradients: dict[str, Tensor] = {}
        for k, M_a_shard in M_a.items():
            M_g_shard = M_g[k]

            # Recover full per-layer dims from shard shapes.
            d_S = M_g_shard.shape[0] * self.world_size
            d_A = M_a_shard.shape[0] * self.world_size

            gradients_noi = torch.from_numpy(mmap[k][:]).to(
                device=self.device, dtype=torch.float32
            )
            gradients_noi = gradients_noi.view(-1, d_S, d_A)

            # Step 1: G @ M_A : [N, d_S, d_A] @ [d_A, p] -> [N, d_S, p]
            intermediate = self.sharded_computer._matmul(
                vector_nsa=gradients_noi, matrix_cb=M_a_shard
            )

            # Step 2: M_S^T @ (G @ M_A) -> [N, p_S, p_A].
            # Computed as ((G @ M_A)^T @ M_S)^T to fit the row-sharded M_g
            # shape via _matmul.
            intermediate = self.sharded_computer._matmul(
                vector_nsa=intermediate.transpose(-2, -1), matrix_cb=M_g_shard
            ).transpose(-2, -1)

            transformed_gradients[k] = intermediate

        del M_a, M_g
        gc.collect()

        self.logger.debug("Finished M_S^T @ G @ M_A")

        torch.cuda.synchronize()
        for k, v in transformed_gradients.items():
            grad_buffer[k][:] = v.to(device="cpu", non_blocking=True).flatten(1).numpy()

        grad_buffer.flush()

        self.logger.info(f"Saved IVHP gradients to {self.cfg.run_path}")


def apply_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    cfg: EkfacConfig,
):
    """Worker function for distributed IVHP computation."""
    if torch.cuda.is_available():
        torch.cuda.set_device(get_device_index(local_rank))

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(get_device(local_rank)),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )

    applicator = EkfacApplicator(cfg)
    applicator.compute_ivhp_sharded()


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
