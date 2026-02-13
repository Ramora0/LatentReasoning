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
pytest tests/test_model.py -k "TestCurriculumScheduler"  # curriculum tests (no GPU needed)
pytest tests/test_model.py -k "TestCurriculumScheduler"  # curriculum tests (no GPU needed)
pytest tests/test_model.py -k "TestLatentReasoningModel"  # full model tests (needs GPU)
```

## Architecture: Three-Phase Forward Pass

The model (`src/model/latent_gemma.py:LatentReasoningModel`) wraps the base Gemma 2 model with a three-phase forward pass:

1. **Phase 1 — Encode** (`phase1_encode`): Embeds question tokens via the embedding matrix (scaled by √d_model). Questions are **left-padded** so real tokens get contiguous RoPE positions.
2. **Phase 2 — Latent Steps** (`phase2_latent_steps`): **Autoregressive appending** — each of K steps runs the transformer, takes the last position's hidden state, blends it with a detached token embedding (argmax → re-embed), and appends the result. Output grows from `[B, q_len, d_model]` to `[B, q_len + K, d_model]`. Gradients flow only through the latent path (the `(1-p)` branch). Uses `torch.utils.checkpoint` per step.
3. **`<answer>` token** — A learned embedding (`answer_token_emb`) is appended after the K thought vectors to signal the transition to answer generation.
4. **Phase 3 — Decode** (`phase3_decode`): The prefix (question + thoughts + `<answer>`) is concatenated with teacher-forced answer embeddings. After the transformer pass, answer-position hidden states are projected directly through the frozen lm_head to produce logits. Loss is standard cross-entropy on answer tokens.

### Trainable vs Frozen

| Component | Status |
|-----------|--------|
| Embedding matrix | Frozen always |
| lm_head (unembedding) | Frozen always (tied to embedding) |
| Transformer weights | Trainable |
| `answer_token_emb` | Trainable (learned transition marker) |

### Curriculum (`src/training/curriculum.py`)

K (number of latent iterations) is fixed at 8 for all training (set via `latent.K` in config). The only schedule is **p annealing**: linearly interpolates from p_start (0.9) → p_end (0.0) over p_anneal_steps. At p=0.9 the model operates near-standard autoregressive generation; at p=0.0 it appends continuous "thought vectors."

### Key Design Detail: Gemma 2 Normalizer

Gemma 2 internally multiplies `inputs_embeds` by √d_model. The model pre-divides by this normalizer in `_run_transformer` to cancel the scaling, and pre-multiplies raw embeddings by the normalizer in `phase1_encode`. This is critical — incorrect scaling breaks the latent loop.

### Key Design Detail: Left-Padded Questions

Questions are left-padded in the collator (`src/data/collator.py`). HuggingFace Gemma 2 computes `position_ids` from `attention_mask`, so left-padded `mask = [0, 0, 1, 1, 1]` yields contiguous positions `[0, 1, 2]` for real tokens. This eliminates RoPE position gaps between question tokens and appended thought vectors.

## Code Layout

- `src/model/` — Model definition (`LatentReasoningModel`)
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

Training: `SynthLabsAI/Big-Math-RL-Verified` (~250K math problems with verified answers). Only question text and final numerical answer are used — no chain-of-thought needed. Question and answer are tokenized and padded separately by the collator (questions left-padded, answers right-padded).
