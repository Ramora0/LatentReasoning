#!/usr/bin/env python3
"""Download all large files to scratch storage.

Run this once before training to avoid repeated downloads:
    HF_TOKEN=hf_xxx python scripts/download.py
    HF_TOKEN=hf_xxx python scripts/download.py --scratch-dir /custom/scratch/path
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SCRATCH_BASE = "/fs/scratch/PAS2836/lees_stuff"


def download_model(model_name, scratch_dir):
    """Download a HuggingFace model via snapshot_download."""
    from huggingface_hub import snapshot_download

    local_dir = os.path.join(scratch_dir, "models", model_name.split("/")[-1])
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"  Model already exists at {local_dir}, skipping.")
        return local_dir

    print(f"  Downloading model '{model_name}' -> {local_dir}")
    snapshot_download(repo_id=model_name, local_dir=local_dir)
    print(f"  Done.")
    return local_dir


def download_dataset(dataset_name, scratch_dir, config_name=None):
    """Download a HuggingFace dataset and save to disk in Arrow format."""
    from datasets import load_dataset

    safe_name = dataset_name.split("/")[-1]
    local_dir = os.path.join(scratch_dir, "datasets", safe_name)
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"  Dataset already exists at {local_dir}, skipping.")
        return local_dir

    print(f"  Downloading dataset '{dataset_name}' -> {local_dir}")
    if config_name:
        ds = load_dataset(dataset_name, config_name)
    else:
        ds = load_dataset(dataset_name)
    ds.save_to_disk(local_dir)
    print(f"  Done.")
    return local_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download all large files (model + datasets) to scratch storage"
    )
    parser.add_argument(
        "--scratch-dir",
        type=str,
        default=SCRATCH_BASE,
        help=f"Base scratch directory (default: {SCRATCH_BASE})",
    )
    args = parser.parse_args()

    scratch_dir = args.scratch_dir
    os.makedirs(os.path.join(scratch_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(scratch_dir, "datasets"), exist_ok=True)

    # Authenticate with HuggingFace if token is set
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("Authenticated with HF_TOKEN")
    else:
        print("Warning: HF_TOKEN not set, gated models/datasets may fail")

    print(f"Scratch directory: {scratch_dir}\n")

    # 1. Base model
    print("[1/4] Model: MCES10/maths-problems-gemma-2-2b-it")
    download_model("MCES10/maths-problems-gemma-2-2b-it", scratch_dir)

    # 2. Training dataset
    print("[2/4] Training data: SynthLabsAI/Big-Math-RL-Verified")
    download_dataset("SynthLabsAI/Big-Math-RL-Verified", scratch_dir)

    # 3. GSM8K eval dataset
    print("[3/4] Eval data: gsm8k")
    download_dataset("gsm8k", scratch_dir, config_name="main")

    # 4. MATH eval dataset
    print("[4/4] Eval data: hendrycks/competition_math")
    download_dataset("hendrycks/competition_math", scratch_dir)

    print(f"\nAll downloads complete.")
    print(f"  Models:   {os.path.join(scratch_dir, 'models')}")
    print(f"  Datasets: {os.path.join(scratch_dir, 'datasets')}")


if __name__ == "__main__":
    main()
