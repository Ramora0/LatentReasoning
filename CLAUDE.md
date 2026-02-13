# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Latent Reasoning trains a pretrained Gemma 2 2B LLM to reason in continuous latent space instead of discrete token space. The core idea: instead of chain-of-thought token generation, the transformer runs K recurrent latent iterations on hidden states, then decodes the final answer. Training uses standard cross-entropy on decoded answer tokens — no RL, no supervision on latent states.

Base model: `MCES10/maths-problems-gemma-2-2b-it` (Gemma 2 2B with tied embedding/unembedding, already SFT'd for math).

## Commands

```bash
# Install
pip install -e .          # core deps
pip install -e ".[dev]"   # + pytest, pytest-timeout
pip install -e ".[tpu]"   # + torch_xla

# Train
python scripts/train.py --config configs/base.yaml                           # single GPU
torchrun --nproc_per_node=4 scripts/train.py --config configs/base.yaml      # multi-GPU FSDP
python scripts/train.py --config configs/base.yaml distributed.backend=xla   # TPU

# Config overrides via CLI dotlist
torchrun --nproc_per_node=4 scripts/train.py --config configs/base.yaml training.learning_rate=1e-5

# Evaluate
python scripts/evaluate.py --config configs/base.yaml --checkpoint checkpoints/step_60000/checkpoint.pt
python scripts/evaluate.py --config configs/base.yaml --checkpoint checkpoints/step_60000/checkpoint.pt --K 4 --p 0.0

# Tests
pytest tests/                    # all tests (GPU tests skip if no CUDA)
pytest tests/test_model.py -k "TestBridgeLayer"         # bridge tests (no GPU needed)
pytest tests/test_model.py -k "TestCurriculumScheduler"  # curriculum tests (no GPU needed)
pytest tests/test_model.py -k "TestLatentReasoningModel"  # full model tests (needs GPU)
```

## Architecture: Three-Phase Forward Pass

The model (`src/model/latent_gemma.py:LatentReasoningModel`) wraps the base Gemma 2 model with a three-phase forward pass:

1. **Phase 1 — Encode** (`phase1_encode`): Embeds question tokens via the frozen embedding matrix (scaled by √d_model).
2. **Phase 2 — Latent Steps** (`phase2_latent_steps`): Runs K recurrent iterations through the transformer. Each step uses annealed interpolation: `input = token_emb * p + latent_state * (1-p)`, where the token_emb path is **detached** (no gradient). Gradients flow only through the latent path. Uses `torch.utils.checkpoint` per step.
3. **Phase 3 — Decode** (`phase3_decode`): Bridge layer maps latent states back toward token manifold, then runs teacher-forced decoding through the **frozen** transformer to produce answer logits. Loss is standard cross-entropy on answer tokens.

### Trainable vs Frozen

| Component | Status |
|-----------|--------|
| Embedding matrix | Frozen always |
| lm_head (unembedding) | Frozen always |
| Transformer weights | Trainable during Phase 2 latent steps, frozen during Phase 3 decode (but gradients flow through activations) |
| Bridge layer (`src/model/bridge.py`) | Trainable (LayerNorm → Linear, initialized near-identity) |

### Curriculum (`src/training/curriculum.py`)

Two schedules drive training:
- **p annealing**: Linearly interpolates from p_start (0.9) → p_end (0.0) over p_anneal_steps. At p=0.9 the model operates near-standard decoding; at p=0.0 it's pure latent reasoning.
- **K scheduling**: Stepwise increase in latent iterations (e.g., K=2 → K=4 → K=8 at configured step thresholds).

### Key Design Detail: Gemma 2 Normalizer

Gemma 2 internally multiplies `inputs_embeds` by √d_model. The model pre-divides by this normalizer in `_run_transformer` to cancel the scaling, and pre-multiplies raw embeddings by the normalizer in `phase1_encode`. This is critical — incorrect scaling breaks the latent loop.

## Code Layout

- `src/model/` — Model definition (`LatentReasoningModel`) and bridge layer
- `src/training/` — Trainer (FSDP for both CUDA and XLA/TPU) and curriculum scheduler
- `src/data/` — `BigMathDataset` (loads `SynthLabsAI/Big-Math-RL-Verified`) and collator (pads question/answer independently)
- `src/eval/` — Evaluator (GSM8K, MATH benchmarks) and answer extraction/comparison metrics
- `scripts/` — Entry points for training and evaluation
- `configs/base.yaml` — Default config (OmegaConf); all settings controlled here
- `plan.md` — Full design document with rationale, risks, and implementation plan

## Config

All configuration is via OmegaConf YAML (`configs/base.yaml`) with CLI dotlist overrides. Key sections: `model`, `latent` (K/p schedules), `training`, `distributed` (backend: "cuda" or "xla", FSDP settings), `data`, `eval`, `logging`, `checkpointing`.

## Backend Support

The trainer and model support both CUDA/NCCL and XLA/TPU backends. Backend is selected via `distributed.backend` in config. Key differences:
- XLA uses `eager` attention implementation; CUDA uses `sdpa`
- XLA uses `XlaFullyShardedDataParallel`; CUDA uses `torch.distributed.fsdp`
- XLA uses `MpDeviceLoader` for async host-to-device transfer
- XLA checkpointing uses `xm.save`

## Dataset

Training: `SynthLabsAI/Big-Math-RL-Verified` (~250K math problems with verified answers). Only question text and final numerical answer are used — no chain-of-thought needed. Question and answer are tokenized and padded separately by the collator.
