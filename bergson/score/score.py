import psutil
import pynvml
import json
import os
import faulthandler
faulthandler.enable()
import random
import psutil
import shutil
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Literal, cast

import torch
import torch.distributed as dist
from datasets import Dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from bergson.collection import collect_gradients
from bergson.config import IndexConfig, ScoreConfig
from bergson.data import allocate_batches, load_gradient_dataset, load_gradients
from bergson.distributed import launch_distributed_run
from bergson.gradients import GradientProcessor
from bergson.score.process_preconditioners import mix_and_save_processors
from bergson.score.scorer import Scorer
from bergson.utils.utils import assert_type
from bergson.utils.worker_utils import (
    create_processor,
    setup_data_pipeline,
    setup_model_and_peft,
)


def preprocess_grads(
    grad_ds: Dataset,
    grad_column_names: list[str],
    unit_normalize: bool,
    batch_size: int,
    device: torch.device,
    accumulate_grads: Literal["mean", "sum", "none"] = "none",
    normalize_accumulated_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """Preprocess the gradients in the dataset. Returns a dictionary
    of preprocessed gradients with shape [1, grad_dim]. Preprocessing
    includes some combination of unit normalization, accumulation,
    accumulated gradient normalization, and dtype conversion."""

    # Short-circuit if possible
    if accumulate_grads == "none" and not unit_normalize:
        return {
            column_name: grad_ds[:][column_name].to(device=device)
            for column_name in grad_column_names
        }

    # Get sum and sum of squares of the gradients
    acc = {
        column_name: torch.zeros_like(
            grad_ds[0][column_name], device=device, dtype=torch.float32
        )
        for column_name in grad_column_names
    }
    ss_acc = torch.tensor(0.0, device=device, dtype=torch.float32)
    if not unit_normalize:
        ss_acc.fill_(1.0)

    def sum_(cols):
        nonlocal ss_acc

        for column_name in grad_column_names:
            x = cols[column_name].to(device=device, dtype=torch.float32)
            acc[column_name].add_(x.sum(0))

            if unit_normalize:
                # To normalize the mean gradient we can divide by the sum of
                # squares of every gradient element in the dataset
                ss_acc += x.pow(2).sum()

    grad_ds.map(
        sum_,
        batched=True,
        batch_size=batch_size,
    )

    ss_acc = ss_acc.sqrt()
    assert ss_acc > 0, "Sum of squares of entire dataset is zero"

    # Process the gradient dataset
    if accumulate_grads == "mean":
        grads = {
            column_name: (acc[column_name] / ss_acc / len(grad_ds)).unsqueeze(0)
            for column_name in grad_column_names
        }
    elif accumulate_grads == "sum":
        grads = {
            column_name: (acc[column_name] / ss_acc).unsqueeze(0)
            for column_name in grad_column_names
        }
    elif accumulate_grads == "none":
        grads = {
            column_name: grad_ds[:][column_name].to(device=device)
            for column_name in grad_column_names
        }
        if unit_normalize:
            norms = torch.cat(list(grads.values()), dim=1).norm(dim=1, keepdim=True)
            grads = {k: v / norms for k, v in grads.items()}
    else:
        raise ValueError(f"Invalid accumulate_grads: {accumulate_grads}")

    # Normalize the accumulated gradient
    if normalize_accumulated_grad:
        grad_norm = torch.cat(
            [grads[column_name].flatten() for column_name in grad_column_names], dim=0
        ).norm()
        for column_name in grad_column_names:
            grads[column_name] /= grad_norm

    return grads


def precondition_ds(
    query_ds: Dataset,
    score_cfg: ScoreConfig,
    target_modules: list[str],
    device: torch.device,
    grad_sizes: dict[str, int],
):
    """Precondition the dataset with the query and index preconditioners."""
    rank = dist.get_rank() if dist.is_initialized() else 0

    query_ds = query_ds.with_format(
        "torch", columns=target_modules, output_all_columns=True
    )

    use_q = score_cfg.query_preconditioner_path is not None
    use_i = score_cfg.index_preconditioner_path is not None

    if use_q or use_i:
        # print("dsfsd")
        if score_cfg.mixed_preconditioner_path is None:
            # print("Mixing and saving processors...")
            if use_q and not use_i:
                score_cfg.mixed_preconditioner_path = (
                    score_cfg.query_preconditioner_path
                )
            elif not use_q and use_i:
                score_cfg.mixed_preconditioner_path = (
                    score_cfg.index_preconditioner_path
                )
            else:
                mix_and_save_processors(
                    score_cfg.query_preconditioner_path,
                    score_cfg.index_preconditioner_path,
                    score_cfg.mixing_coefficient,
                    Path(score_cfg.query_path),
                    target_modules,
                    device,
                    grad_sizes,
                )
                score_cfg.mixed_preconditioner_path = score_cfg.query_path
                # print("Mixed processor saved to disk")

        # print("Making query path")
        query_path = Path(score_cfg.query_path)
        query_path.mkdir(parents=True, exist_ok=True)
        preconditioned_path = str(query_path / "preconditioned")
        # print("made query path")

        assert score_cfg.mixed_preconditioner_path is not None
        mixed_processor_path = Path(score_cfg.mixed_preconditioner_path)

        # Validate that all target_modules exist in grad_sizes before processing
        missing_modules = [name for name in target_modules if name not in grad_sizes]
        if missing_modules:
            raise KeyError(
                f"Modules not found in grad_sizes: {missing_modules}. "
                f"Available modules: {list(grad_sizes.keys())[:10]}..."
            )

        if rank == 0:
            processed_data = {}
            for name in target_modules:
                print(f"Rank 0 processing module {name}", flush=True)
                module_data = query_ds[:][name]
                print("1", flush=True)
                assert isinstance(module_data, torch.Tensor)
                print("2", flush=True)

                processor = GradientProcessor.load(
                    mixed_processor_path, map_location="cpu", module_names=[name]
                )
                print("2.1 asserting", flush=True)
                assert name in processor.preconditioners, f"Module {name} not found in preconditioners"
                assert name in processor.preconditioners_eigen, f"Module {name} not found in preconditioners_eigen"
                print("3", flush=True)
                prec = processor.preconditioners[name]
                print("3.1", flush=True)
                eigval, eigvec = processor.preconditioners_eigen[name]
                print("3.2", flush=True)
                eigval, eigvec = eigval.to(device), eigvec.to(device)
                print("4", flush=True)

                # Compute H_inv
                h_inv = (eigvec * (1.0 / eigval) @ eigvec.mT).to(dtype=prec.dtype)
                print("5", flush=True)

                # 4. Apply Transformation
                # module_data is [1, dim], h_inv is [dim, dim] -> Result is [1, dim]
                # Use torch.cuda.synchronize() to ensure GPU operations complete before .cpu()
                torch.cuda.synchronize(device)
                print("6", flush=True)
                module_data_gpu = module_data.to(device)
                result_gpu = module_data_gpu @ h_inv
                print("7", flush=True)
                processed_tensor = result_gpu.cpu()  # type: ignore
                print("8", flush=True)
                torch.cuda.synchronize(device)
                print("9", flush=True)

                processed_data[name] = processed_tensor

                del processor, h_inv, eigval, eigvec, module_data
                print("10", flush=True)
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                gc.collect()
                print("11", flush=True)

                # Log memory usage
                print("Memory usage:", torch.cuda.memory_allocated(device) / (1024 ** 3), "GB", flush=True)
                # Log CPU memory usage
                print("CPU memory usage:", psutil.cpu_percent(), flush=True)

            print("Constructing final preconditioned dataset...", flush=True)
            
            ds = Dataset.from_dict(processed_data)

            os.makedirs(Path(preconditioned_path).parent, exist_ok=True)
            ds.save_to_disk(preconditioned_path, num_proc=1)

        if dist.is_initialized():
            rank = dist.get_rank()
            print(
                f"[precondition_ds] Rank {rank} dist.barrier())",
                flush=True,
            )
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            dist.barrier(device_ids=[local_rank])
            # Log IMMEDIATELY after barrier with print (no file I/O)
            print(
                f"[precondition_ds] Rank {rank} passed barrier...",
                flush=True,
            )

        print("All loading preconditioned dataset from disk...", rank)
        query_ds = assert_type(Dataset, load_from_disk(preconditioned_path, keep_in_memory=False))

    return query_ds.with_format("torch", columns=score_cfg.modules)


def get_query_ds(score_cfg: ScoreConfig):
    """
    Load and preprocess the query dataset to get the query gradients. Preconditioners
    may be mixed as described in https://arxiv.org/html/2410.17413v1#S3.
    """
    print(f"[get_query_ds] Loading query dataset from {score_cfg.query_path}...")
    # Collect the query gradients if they don't exist
    query_path = Path(score_cfg.query_path)
    if not query_path.exists():
        raise FileNotFoundError(
            f"Query dataset not found at {score_cfg.query_path}. "
            "Please build a query dataset index first."
        )

    print(f"[get_query_ds] Reading info.json...")
    # Load the query dataset
    with open(query_path / "info.json", "r") as f:
        target_modules = json.load(f)["dtype"]["names"]
    print(f"[get_query_ds] Found {len(target_modules)} target modules")

    if not score_cfg.modules:
        score_cfg.modules = target_modules

    print(f"[get_query_ds] Loading gradient dataset (structured=True)...")
    try:
        query_ds = load_gradient_dataset(Path(score_cfg.query_path), structured=True)
        print(f"[get_query_ds] Gradient dataset loaded, length={len(query_ds)}")
    except ValueError as e:
        if "integer won't fit into a C int" not in str(e):
            raise e

        print(
            "Query gradients are too large to load with structure. "
            "Attempting to load without structure..."
        )

        mmap = load_gradients(Path(score_cfg.query_path), structured=False)

        # Convert unstructured gradients to a dictionary of module-wise tensors
        with open(query_path / "info.json", "r") as f:
            metadata = json.load(f)
            grad_sizes = metadata["grad_sizes"]

        sizes = torch.tensor(list(grad_sizes.values()))
        module_offsets = torch.tensor([0] + torch.cumsum(sizes, dim=0).tolist())

        query_ds = Dataset.from_dict(
            {
                name: mmap[:, module_offsets[i] : module_offsets[i + 1]].copy()
                for i, name in enumerate(grad_sizes.keys())
                if name in target_modules
            }
        )

    return query_ds.with_format("torch", columns=target_modules)


def load_grad_sizes(
    score_cfg: ScoreConfig,
) -> dict[str, int]:
    """Load the preconditioner metadata from the score configuration."""
    # Grad sizes are the flattened lengths of the gradient vectors
    with open(Path(score_cfg.query_path) / "info.json", "r") as f:
        return json.load(f)["grad_sizes"]


def score_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    index_cfg: IndexConfig,
    score_cfg: ScoreConfig,
    # ds: Dataset | IterableDataset,
):
    """
    Score worker executed per rank to produce and score gradients against a query.

    Parameters
    ----------
    rank : int
        Distributed rank / GPU ID for this worker.
    world_size : int
        Total number of workers participating in the run.
    index_cfg : IndexConfig
        Specifies the model, tokenizer, PEFT adapters, and other settings.
    score_cfg : ScoreConfig
        Score configuration specifying query path, target modules, and scoring
        method (mean/nearest/individual).
    ds : Dataset | IterableDataset
        The entire dataset to be indexed. A subset is assigned to each worker.
    query_grads : dict[str, torch.Tensor]
        Preprocessed query gradient tensors (often [1, grad_dim]) keyed by module name.
    """

    import sys

    print(
        f"[score_worker] Rank {rank} starting, local_rank={local_rank}, world_size={world_size}",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()

    ds = load_from_disk("tmp_score")
    print(f"[score_worker] Rank {rank} dataset loaded, length={len(ds)}")

    torch.cuda.set_device(local_rank)
    local_device = torch.device(f"cuda:{local_rank}")

    # These should be set by the main process
    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        print(
            f"[score_worker] Rank {rank} initializing process group at {addr}:{port}..."
        )
        try:
            dist.init_process_group(
                "nccl",
                init_method=f"tcp://{addr}:{port}",
                device_id=torch.device(f"cuda:{local_rank}"),
                rank=rank,
                timeout=timedelta(hours=1),
                world_size=world_size,
            )
            print(f"[score_worker] Rank {rank} process group initialized")
            # Add barrier to ensure all workers are synchronized before loading dataset
            dist.barrier()
            # Stagger dataset loading to avoid file lock conflicts
            # Each rank waits a small random amount to avoid simultaneous file access
            time.sleep(rank * 0.1 + random.uniform(0, 0.05))
        except Exception as e:
            print(
                f"[score_worker] Rank {rank} process group initialization failed: {e}",
                flush=True,
            )
            raise

    grad_sizes = load_grad_sizes(score_cfg)

    # Load query_ds inside the worker to avoid serialization issues
    print(f"[score_worker] Rank {rank} loading query dataset...", flush=True)
    query_ds = get_query_ds(score_cfg)
    print(
        f"[score_worker] Rank {rank} query dataset loaded, starting precondition_ds...",
        flush=True,
    )
    query_ds = precondition_ds(
        query_ds,
        score_cfg,
        score_cfg.modules,
        local_device,
        grad_sizes,
    )
    print(f"[score_worker] Rank {rank} precondition_ds completed", flush=True)
    query_grads = preprocess_grads(
        query_ds,
        score_cfg.modules,
        score_cfg.unit_normalize,
        score_cfg.batch_size,
        local_device,
        accumulate_grads="mean" if score_cfg.score == "mean" else "none",
        normalize_accumulated_grad=score_cfg.score == "mean",
    )
    print("preproc done", flush=True)

    model, target_modules = setup_model_and_peft(index_cfg, local_rank)
    model = cast(PreTrainedModel, model)
    grads_dtype = torch.float32 if model.dtype == torch.float32 else torch.float16
    processor = create_processor(index_cfg, local_rank, rank)
    print("Created processor", flush=True)

    attention_cfgs = {
        module: index_cfg.attention for module in index_cfg.split_attention_modules
    }

    print("A", flush=True)

    kwargs = {
        "model": model,
        "data": ds,
        "processor": processor,
        "cfg": index_cfg,
        "target_modules": target_modules,
        "attention_cfgs": attention_cfgs,
    }
    print("B", flush=True)

    if isinstance(ds, Dataset):
        print("ds is a Dataset", flush=True)
        kwargs["batches"] = allocate_batches(ds["length"], index_cfg.token_batch_size)
        print("scorer", flush=True)
        kwargs["scorer"] = Scorer(
            index_cfg.partial_run_path,
            len(ds),
            query_grads,
            score_cfg,
            device=local_device,
            dtype=grads_dtype,
        )
        print("Scorer", flush=True)

        collect_gradients(**kwargs)
    else:
        # Convert each shard to a Dataset then map over its gradients
        buf, shard_id = [], 0

        def flush(kwargs):
            nonlocal buf, shard_id
            if not buf:
                return
            ds_shard = assert_type(Dataset, Dataset.from_list(buf))
            batches = allocate_batches(
                ds_shard["length"][:], index_cfg.token_batch_size
            )
            kwargs["ds"] = ds_shard
            kwargs["batches"] = batches

            kwargs["scorer"] = Scorer(
                index_cfg.partial_run_path / f"shard-{shard_id:05d}",
                len(ds_shard),
                query_grads,
                score_cfg,
                torch.device(f"cuda:{rank}"),
                model.dtype if model.dtype != "auto" else torch.float32,
            )

            collect_gradients(**kwargs)

            buf.clear()
            shard_id += 1

        for ex in tqdm(ds, desc="Collecting gradients"):
            buf.append(ex)
            if len(buf) == index_cfg.stream_shard_size:
                flush(kwargs=kwargs)

        flush(kwargs=kwargs)  # Final flush
        processor.save(index_cfg.partial_run_path, rank)


def score_dataset(
    index_cfg: IndexConfig,
    score_cfg: ScoreConfig,
):
    """
    Score a dataset against an existing gradient index.

    Parameters
    ----------
    index_cfg : IndexConfig
        Specifies the run path, dataset, model, tokenizer, PEFT adapters,
        and other gradient collection settings.
    score_cfg : ScoreConfig
        Specifies the query path, target modules, and scoring method
        (mean/nearest/individual).
    """
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    start_rank = int(os.environ.get("START_RANK", 0))
    actual_rank = start_rank + rank

    print(
        f"[score_dataset] Rank {actual_rank} (start_rank={start_rank}, local_rank={rank}) starting"
    )

    if actual_rank == 0:
        index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
        with (index_cfg.partial_run_path / "index_config.json").open("w") as f:
            json.dump(asdict(index_cfg), f, indent=2)
        print("dumping json")
        with (index_cfg.partial_run_path / "score_config.json").open("w") as f:
            json.dump(asdict(score_cfg), f, indent=2)

    print(f"[score_dataset] Rank {actual_rank} loading dataset...")
    ds = setup_data_pipeline(index_cfg)
    ds.save_to_disk("tmp_score")
    print(
        f"[score_dataset] Rank {actual_rank} dataset loaded, loading query dataset..."
    )

    # Don't load query_ds here - load it inside the worker to avoid serialization issues
    # query_ds = get_query_ds(score_cfg)
    # print(f"[score_dataset] Rank {actual_rank} query dataset loaded, launching distributed run...")

    print(
        f"[score_dataset] Rank {actual_rank} launching distributed run (query_ds will be loaded in worker)..."
    )
    launch_distributed_run("score", score_worker, [index_cfg, score_cfg])

    if rank == 0:
        shutil.move(index_cfg.partial_run_path, index_cfg.run_path)
