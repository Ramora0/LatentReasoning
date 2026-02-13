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
    """Initialize XLA/TPU distributed process group."""
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_backend  # noqa: F401 — registers the 'xla' backend

    # XLA dist init: use xla backend
    if not dist.is_initialized():
        dist.init_process_group(backend="xla")

    rank = xm.get_ordinal()
    return rank


def _get_device(backend: str, rank: int):
    """Get the appropriate device for the backend."""
    if backend == "xla":
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    return torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Train latent reasoning model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args, overrides = parser.parse_known_args()

    # Load config with CLI overrides
    config = OmegaConf.load(args.config)
    if overrides:
        cli_conf = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, cli_conf)

    backend = getattr(config.distributed, "backend", "cuda")

    # Detect and init distributed environment
    if backend == "xla":
        # XLA: always init (xla_spawn or torchrun sets env vars)
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
    )
    collator = LatentReasoningCollator(tokenizer=tokenizer)

    # Build model
    if rank == 0:
        print("Loading model...")
    model = LatentReasoningModel(config)

    # Build evaluator
    device = _get_device(backend, rank)
    evaluator = Evaluator(config, tokenizer, device)

    # Build trainer and train
    trainer = LatentReasoningTrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        collator=collator,
        evaluator=evaluator,
    )

    if rank == 0:
        print("Starting training...")
    trainer.train()

    # Cleanup
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
