#!/usr/bin/env python3
"""Training entry point for latent reasoning model.

Usage:
    # Single GPU:
    python scripts/train.py --config configs/base.yaml

    # Multi-GPU (e.g., 4 GPUs):
    torchrun --nproc_per_node=4 scripts/train.py --config configs/base.yaml

    # TPU (e.g., v4-8 with 4 chips):
    python scripts/train.py --config configs/base.yaml distributed.backend=xla

    # With overrides:
    torchrun --nproc_per_node=4 scripts/train.py --config configs/base.yaml training.learning_rate=1e-5
"""

import argparse
import os
import shutil
import sys

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import AutoTokenizer

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import BigMathDataset
from src.data.collator import LatentReasoningCollator
from src.model.latent_gemma import LatentReasoningModel
from src.eval.evaluator import Evaluator
from src.training.trainer import LatentReasoningTrainer


def _init_cuda_distributed():
    """Initialize CUDA/NCCL distributed process group."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank


def _init_xla_distributed():
    """Initialize XLA/TPU distributed process group.

    Called inside each process spawned by xmp.spawn.
    """
    import torch_xla.distributed.xla_backend  # noqa: F401 — registers the 'xla' backend
    import torch_xla.runtime as xr

    if not dist.is_initialized():
        dist.init_process_group(backend="xla", init_method="xla://")

    rank = xr.global_ordinal()
    return rank


def _get_device(backend: str, rank: int):
    """Get the appropriate device for the backend."""
    if backend == "xla":
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    return torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")


def _parse_config():
    """Parse CLI args and build OmegaConf config."""
    parser = argparse.ArgumentParser(description="Train latent reasoning model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args, overrides = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    if overrides:
        cli_conf = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, cli_conf)
    return config


def _train_fn(rank_unused, config):
    """Training function run by each process.

    For XLA this is called via xmp.spawn (rank_unused is the index arg
    injected by spawn — we ignore it and use xm.get_ordinal() instead).
    For CUDA it is called directly from main().
    """
    backend = getattr(config.distributed, "backend", "cuda")

    # Detect and init distributed environment
    if backend == "xla":
        rank = _init_xla_distributed()
        is_distributed = True
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # CUDA: torchrun sets RANK/WORLD_SIZE
        rank = _init_cuda_distributed()
        is_distributed = True
    else:
        rank = 0
        is_distributed = False

    if rank == 0:
        print(f"Backend: {backend}")
        print(f"Config:\n{OmegaConf.to_yaml(config)}")

    # --- Resolve local paths from scratch ---
    scratch_dir = config.paths.scratch_dir if hasattr(config, "paths") else None
    model_path = config.model.name
    dataset_path = None
    eval_data_dir = None

    if scratch_dir:
        local_model = os.path.join(
            scratch_dir, "models", config.model.name.split("/")[-1]
        )
        if os.path.isdir(local_model):
            model_path = local_model

        local_dataset = os.path.join(
            scratch_dir, "datasets", config.data.dataset.split("/")[-1]
        )
        if os.path.isdir(local_dataset):
            dataset_path = local_dataset

        eval_data_dir = os.path.join(scratch_dir, "datasets")

    # Copy training dataset to $TMPDIR for fast I/O during training
    tmp_dir = os.environ.get("TMPDIR")
    if tmp_dir and dataset_path:
        tmp_dataset = os.path.join(
            tmp_dir, "datasets", config.data.dataset.split("/")[-1]
        )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0 and not os.path.isdir(tmp_dataset):
            if rank == 0:
                print(f"Copying dataset to TMPDIR: {tmp_dataset}")
            shutil.copytree(dataset_path, tmp_dataset)
        if is_distributed:
            dist.barrier()
        dataset_path = tmp_dataset

    # Point model loading at the resolved path
    config.model.name = model_path

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset and collator
    if rank == 0:
        print("Loading dataset...")
    train_dataset = BigMathDataset(
        tokenizer=tokenizer,
        max_question_tokens=config.data.max_question_tokens,
        max_answer_tokens=config.data.max_answer_tokens,
        data_dir=dataset_path,
    )
    collator = LatentReasoningCollator(tokenizer=tokenizer)

    # Build model
    if rank == 0:
        print("Loading model...")
    model = LatentReasoningModel(config)

    # Build evaluator
    device = _get_device(backend, rank)
    evaluator = Evaluator(config, tokenizer, device, eval_data_dir=eval_data_dir)

    # Build trainer and train
    trainer = LatentReasoningTrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        collator=collator,
        evaluator=evaluator,
        tokenizer=tokenizer,
    )

    if rank == 0:
        print("Starting training...")
    trainer.train()

    # Cleanup
    if is_distributed:
        dist.destroy_process_group()


def main():
    config = _parse_config()
    backend = getattr(config.distributed, "backend", "cuda")

    if backend == "xla":
        import torch_xla.distributed.xla_multiprocessing as xmp
        # xmp.spawn launches one process per TPU core; nprocs=None auto-detects
        xmp.spawn(_train_fn, args=(config,), nprocs=None)
    else:
        _train_fn(0, config)


if __name__ == "__main__":
    main()
