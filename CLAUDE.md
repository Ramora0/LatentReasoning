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

# TPU manual install (torch 2.9.0 CPU + torch_xla 2.9.0)
pip uninstall -y torch torch_xla && \
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu && \
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' -f https://storage.googleapis.com/libtpu-wheels/index.html

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
2. **Phase 2 — Latent Steps** (`phase2_latent_steps`): **Training**: K steps run under `torch.no_grad()` with KV cache (`_generate_thoughts_no_grad`), producing K blended thought vectors. These are detached and given `requires_grad=True` as stitching inputs. **Inference**: same autoregressive loop but returns KV cache for decoding. Output grows from `[B, q_len, d_model]` to `[B, q_len + K, d_model]`.
3. **`<answer>` token** — A learned embedding (`answer_token_emb`) is appended after the K thought vectors to signal the transition to answer generation.
4. **Phase 3 — Decode** (`phase3_decode`): The prefix (question + thoughts + `<answer>`) is concatenated with teacher-forced answer embeddings. Single full-sequence transformer pass with gradients. Answer-position hidden states are projected through the frozen lm_head to produce logits. Loss is standard cross-entropy on answer tokens. During training, also extracts `thought_outputs` (hidden states at thought positions) for gradient stitching.

### Gradient Stitching (Training)

Instead of backpropagating through K sequential transformer calls, gradient stitching decouples thought generation from gradient computation:

1. **Forward**: Phase 2 generates thoughts without grad. Phase 3 runs the full sequence `[q, <thinking>, t_0..t_{K-1}, <answer>, ans...]` through the transformer with gradients.
2. **`loss.backward(retain_graph=True)`**: Computes gradients on `thought_inputs` (leaf tensors at thought positions).
3. **D stitching iterations**: Extract gradient `g[k]` from each thought input, compute `L_stitch = Σ g[k] · thought_outputs[k]`, call `L_stitch.backward()`. This propagates gradients across thought boundaries through the Phase 3 computation graph.

`stitching_depth` (D) defaults to K (exact gradients). D=1 is likely sufficient since Phase 3's attention already creates cross-boundary dependencies. No gradient checkpointing is used — all activations are stored for the backward passes.

### Trainable vs Frozen

| Component | Status |
|-----------|--------|
| Embedding matrix | Frozen always |
| lm_head (unembedding) | Frozen always (tied to embedding) |
| Transformer weights | Trainable |
| `thinking_token_emb` | Trainable (learned transition marker) |
| `answer_token_emb` | Trainable (learned transition marker) |

### Curriculum (`src/training/curriculum.py`)

K (number of latent iterations) is fixed at 8 for all training (set via `latent.K` in config). The only schedule is **p annealing**: linearly interpolates from p_start (0.9) → p_end (0.0) over `p_anneal_ratio` (default 0.8) of total optimizer steps. At p=0.9 the model operates near-standard autoregressive generation; at p=0.0 it appends continuous "thought vectors."

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

All configuration is via OmegaConf YAML (`configs/base.yaml`) with CLI dotlist overrides. Key sections: `model`, `latent` (K/p schedules/stitching_depth), `training`, `distributed` (backend: "cuda" or "xla", FSDP settings), `data`, `eval`, `logging`, `checkpointing`.

## Backend Support

The trainer and model support both CUDA/NCCL and XLA/TPU backends. Backend is selected via `distributed.backend` in config. Key differences:
- XLA uses `eager` attention implementation; CUDA uses `sdpa`
- XLA uses `XlaFullyShardedDataParallel`; CUDA uses `torch.distributed.fsdp`
- XLA uses `MpDeviceLoader` for async host-to-device transfer
- XLA checkpointing uses `xm.save`

## Dataset

Training: `SynthLabsAI/Big-Math-RL-Verified` (~250K math problems with verified answers). Only question text and final numerical answer are used — no chain-of-thought needed. Question and answer are tokenized and padded separately by the collator (questions left-padded, answers right-padded).
