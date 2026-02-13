#!/usr/bin/env python3
"""Evaluation entry point for latent reasoning model.

Usage:
    python scripts/evaluate.py --config configs/base.yaml --checkpoint checkpoints/step_60000/checkpoint.pt
    python scripts/evaluate.py --config configs/base.yaml --checkpoint checkpoints/step_60000/checkpoint.pt --K 4 --p 0.0

    # TPU:
    python scripts/evaluate.py --config configs/base.yaml --checkpoint checkpoints/step_60000/checkpoint.pt distributed.backend=xla
"""

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.latent_gemma import LatentReasoningModel
from src.eval.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate latent reasoning model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--K", type=int, default=None, help="Override K (latent iterations)")
    parser.add_argument("--p", type=float, default=None, help="Override p (interpolation weight)")
    parser.add_argument("--benchmarks", nargs="+", default=None, help="Override benchmarks to run")
    args, overrides = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    if overrides:
        cli_conf = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, cli_conf)

    if args.benchmarks:
        config.eval.benchmarks = args.benchmarks

    backend = getattr(config.distributed, "backend", "cuda")
    if backend == "xla":
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    model = LatentReasoningModel(config)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Get K and p
    K = args.K if args.K is not None else ckpt.get("K", config.latent.initial_K)
    p = args.p if args.p is not None else ckpt.get("p", 0.0)

    print(f"Evaluating with K={K}, p={p}")

    # Run evaluation
    evaluator = Evaluator(config, tokenizer, device)
    results = evaluator.evaluate(model, K=K, p=p)

    print("\nResults:")
    for benchmark, accuracy in results.items():
        print(f"  {benchmark}: {accuracy:.4f} ({accuracy*100:.1f}%)")


if __name__ == "__main__":
    main()
