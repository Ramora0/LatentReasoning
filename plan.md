# Latent Reasoning Training Plan — Gemma 2 2B

## Core Thesis

Train a pretrained LLM to reason entirely in continuous latent space rather than discrete token space, showing that end-to-end differentiable latent reasoning provides superior training signal to RL-based chain-of-thought approaches on math problems.

## Architecture

### Base Model
- **MCES10/maths-problems-gemma-2-2b-it** — a Gemma 2 2B checkpoint already SFT'd for math reasoning
- 26 transformer layers, **tied embedding/unembedding matrices** (critical: tied weights ensure the model's output latent space is very similar to its input space, making latent recurrence coherent)

### Three-Phase Forward Pass

**Phase 1: Encode (discrete → latent)**
1. Tokenize the input math question using standard Gemma tokenizer
2. Embed tokens using the original embedding matrix (frozen)
3. Run one standard forward pass through the full transformer to produce contextualized hidden states
4. Take the final hidden state as the initial latent representation

**Phase 2: Latent Computation (pure latent space)**
1. The transformer operates as a recurrent computation engine: hidden state → transformer → new hidden state
2. Each "latent step" feeds the previous step's hidden state (shape: `[batch, seq_len, d_model]`) back as input to the transformer
3. Run K latent steps (K is a hyperparameter, curriculum-scheduled)
4. No tokenization, no discrete bottleneck — the model computes freely in continuous d_model space
5. The latent states are whatever the model currently produces — they are NOT replaced or supervised directly

**Phase 3: Decode (latent → discrete)**
1. A learned **bridge layer** (LayerNorm → Linear, or a small MLP) maps the final latent hidden state back toward the token embedding manifold
2. Re-attach the original unembedding matrix (frozen) and run standard autoregressive decoding to produce the answer
3. The bridge is the only new learned component connecting latent computation to discrete decoding

### Bridge Layer Specification
- Input: final latent hidden state from Phase 2 (`d_model`-dimensional)
- Architecture: `LayerNorm → Linear(d_model, d_model)` (start simple, can add an MLP if needed)
- Output: a vector in d_model space that the frozen decoder can read from
- Consider initializing close to identity so initial behavior is pass-through

### Trainable vs Frozen Parameters
| Component | During Latent Steps (Phase 2) | During Decode (Phase 3) |
|-----------|-------------------------------|------------------------|
| Embedding matrix | Not used | Not used |
| Transformer weights | **Trainable** (or LoRA) | **Frozen** |
| Bridge layer | N/A | **Trainable** |
| Unembedding matrix | Not used | **Frozen** |

Freeze the decoder-phase transformer and unembedding to preserve the model's ability to produce coherent tokens. Only train the latent-phase transformer and the bridge.

### Distribution Shift Mitigation: Annealed Interpolation

The core challenge is that pretrained transformer layers expect token-embedding-like inputs, but latent states have different statistics. We handle this with a smooth annealing variable **p**.

At each latent step, the input to the transformer is:
```
input = token_embedding * p + latent_state * (1 - p)
```

Where:
- `token_embedding` = the standard discrete token embedding of the output from the previous step (via argmax → embed lookup). **This is detached from the gradient graph** (no gradients flow through the token path).
- `latent_state` = the raw hidden state output from the previous latent step. **Gradients flow through this path.**
- `p` = annealing variable, starts at **0.9** and is annealed down toward **0.0** over training

Behavior:
- **p = 1.0**: Standard tokenized autoregressive decoding (pure discrete). No latent reasoning.
- **p = 0.9** (start of training): Almost standard decoding with a small latent residual. The model can train stably because inputs look almost like normal token embeddings.
- **p = 0.0** (end of training): Pure latent reasoning. The discrete token path is fully removed.

The annealing schedule should be gradual. Suggested: linear decay from 0.9 to 0.0 over the course of training. Monitor loss stability — if loss spikes when p drops, slow the schedule.

This is elegant because:
- Early training: the model learns math reasoning in a near-standard regime
- Mid training: the model gradually learns to use the latent channel for additional information
- Late training: the model operates purely in latent space, and the weights have smoothly adapted

## Training

### Loss Function
- Standard **cross-entropy** on the decoded answer tokens, teacher-forced against the ground truth answer
- This is the ONLY loss

### Gradient Flow
The entire forward pass is end-to-end differentiable:
```
CE Loss on answer tokens
    ↓ backprop
Decoder (frozen, but gradients flow through)
    ↓
Bridge layer (trainable)
    ↓
Latent step K — input was: token_emb * p + latent * (1-p)
    ↓              gradients flow only through the latent term
Latent step K-1
    ↓
...
    ↓
Latent step 1
    ↓
Initial encoded representation
```

The model always sees its own (possibly wrong) latent states during training. Ground truth is only injected at the decoder via teacher forcing. The loss gradient shapes the latent computation over time — identical to how any encoder learns without direct supervision on its hidden states.

### Why This Is Superior to RL
- RL gives a scalar reward (right/wrong) after the full rollout — one bit of signal
- This gives a **per-token gradient through every latent step** — dense, directional signal
- RL requires variance reduction tricks (baselines, PPO clipping, etc.) — this is just standard backprop
- The latent states get precise "which direction to move" information, not "try something different"

### Curriculum Strategy
1. **K scheduling**: Start with small K (e.g. 2-4), increase as training progresses and p decreases
2. **p annealing**: Start at 0.9, linearly anneal toward 0.0. This is the primary curriculum mechanism — it smoothly transitions from near-standard decoding to pure latent reasoning.
3. Optionally anneal both simultaneously: as p decreases (model relies more on latent states), increase K (model gets more latent compute).

### Practical Training Details
- **Gradient checkpointing**: Mandatory. At K=8, backpropagating through 208 effective layers (8 × 26).
- **Gradient clipping**: Clip global norm to ~1.0. Monitor gradient norms at the earliest latent steps to verify signal isn't vanishing.
- **Optimizer**: AdamW, learning rate ~1e-5 to 5e-5, cosine schedule with warmup
- **Batch size**: As large as memory allows given the BPTT depth. Gradient accumulation likely needed.
- **Mixed precision**: bf16 for forward pass, fp32 for gradient accumulation (important for long backprop chains)

## Dataset

### Big-Math
- Source: `SynthLabsAI/Big-Math-RL-Verified` on HuggingFace
- ~250K high-quality math problems with verified numerical answers
- Use the full dataset, no filtering
- All answers are uniquely verifiable — perfect for cross-entropy on decoded answers
- Apache licensed

### What You Need From Each Problem
- The question text (for embedding in Phase 1)
- The final numerical answer (for teacher-forcing in Phase 3)
- You do NOT need chain-of-thought solutions — reasoning happens in latent space

## Evaluation

### Primary Metrics
- **GSM8K test set accuracy**: Standard grade-school math benchmark
- **MATH test set accuracy**: Competition-level math
- **Accuracy vs. K curve**: How does performance scale with number of latent steps? This is the key result showing the model is actually using latent computation.
- **Accuracy vs. p curve**: Track performance at different annealing stages to show the model progressively learns to reason in latent space

## Key Risks

### Vanishing Gradients Through Deep BPTT
- At high K, gradient signal may not reach early latent steps
- **Mitigation**: Gradient clipping, gradient norm monitoring, start with small K

### Bridge Failure Mode
- If latent states drift too far from any decodable representation, the bridge can't recover
- **Mitigation**: The annealing approach inherently prevents this — states are always a blend of token-like and latent until late training, so the bridge has time to adapt. Consider an optional anchor loss if training is unstable.

### p Annealing Too Fast
- If p drops faster than the model can adapt, loss will spike and training may destabilize
- **Mitigation**: Monitor loss as a function of p. Slow the annealing schedule if needed. Can use loss-gated annealing (only decrease p when loss is below a threshold).

## Implementation Order
1. **Data pipeline**: Download Big-Math, extract question/answer pairs
2. **Model surgery**: Load MCES10/maths-problems-gemma-2-2b-it, implement the three-phase forward pass with detachable embedding/unembedding
3. **Annealing mechanism**: Implement the `token_emb * p + latent * (1-p)` interpolation with p scheduling, ensuring token_emb path is detached from gradient graph
4. **Bridge layer**: Implement and initialize close to identity
5. **Training loop at p=0.9, small K**: Verify the pipeline trains and converges when close to standard decoding
6. **Annealing**: Run full training with p annealing from 0.9 → 0.0, monitor stability
7. **Evaluation**: Run GSM8K and MATH benchmarks, generate accuracy-vs-K and accuracy-vs-p curves

## Related Work
- **Coconut** (Meta, 2024): "Training Large Language Models to Reason in a Continuous Latent Space" — feeds hidden states back as input, but retrofits via fine-tuning with specific objectives. This work differs by training the latent computation from scratch with pure answer-level supervision and smooth annealing from discrete to continuous.
- **Adaptive Computation Time** (Graves, 2016): The halting mechanism for variable-depth computation — relevant if we later add dynamic K.
- **Universal Transformers** (Dehghani et al., 2019): Weight-sharing across layers for iterative refinement.