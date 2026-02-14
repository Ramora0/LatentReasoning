# Latent Reasoning: Gradient Stitching Plan

## Current Approach

Each latent thought step is a full forward pass through the model **with gradients enabled**. For K thought steps, this means K sequential forward passes, each building on the previous output, with the full computation graph retained across all steps. Backward pass traverses the entire chain.

This is correct but expensive: peak memory scales with K (all intermediate activations stored simultaneously), and the sequential dependency prevents any parallelism during forward.

## New Approach

Three phases, only one of which requires gradients.

### Phase 1: Generate thoughts (no_grad)

Run K sequential forward passes with `torch.no_grad()` and standard KV caching. Each step takes the previous output and generates the next latent thought token. This is identical to normal autoregressive generation — cheap, no activations stored, KV cache makes each step O(n) instead of O(n²).

**Output:** K detached latent thought vectors `[t_1, t_2, ..., t_K]`.

### Phase 2: Parallel forward (with grad)

Single forward pass over the full sequence `[question_tokens, t_1, t_2, ..., t_K, answer_tokens]`, with gradients enabled. The thought vectors are fed in as detached inputs — the model doesn't know they came from itself.

Because the inputs at thought positions are detached, the computation graph has no edges connecting `o_{k-1}` (the model's output at position k-1) to `t_k` (the input at position k). Every position's computation is an independent subgraph rooted at the input embeddings/vectors.

**Output:** Answer logits, plus live `o_k` nodes at every thought position.

### Phase 3: Backward with gradient stitching

**Step 1 — Standard backward:**

Compute answer loss (masked to answer tokens only). Call `loss.backward(retain_graph=True)`. This gives us:

- `∂L/∂(t_k)` for every thought position k (stored as `t_k.grad`)
- Weight gradients from the direct path (depth-0: how W processes fixed thoughts into the answer)

**Step 2 — Iterative stitching:**

For d = 1 to D (where D ≤ K, but in practice 3-5 is likely sufficient):

1. Detach current thought gradients: `g_k = t_k.grad.detach()` for all k
2. Construct stitching loss: `L_stitch = Σ_k (g_k * o_{k-1}).sum()`
3. Call `L_stitch.backward(retain_graph=True)`
4. This accumulates corrected gradients into `W.grad` and updates `t_k.grad` to reflect depth-(d) credit assignment

Each iteration pushes gradient signal one step deeper through the thought chain. Iteration 1 captures "changing W changes thought k, affecting the answer through thought k+1." Iteration 2 captures the further cascade through k+2, and so on.

**Start with D = K for exact gradients.** Ablate D downward once training is validated.

## Implementation Changes

### What to add

1. **Detach thought vectors after Phase 1.** They should be plain tensors with `requires_grad=True` but no grad_fn. This is probably already the case if Phase 1 uses no_grad, but verify — if they're slices of a larger tensor, clone and detach explicitly.

2. **Store output vectors at thought positions.** After Phase 2, extract `o_k` for k = 0 to K-1 from the model's output hidden states. These must remain live in the graph (do not detach them). In HuggingFace, pass `output_hidden_states=True` or hook into the final layer to capture the last hidden state at each thought position.

3. **Stitching loop.** After the standard backward:

```python
# g[k] = gradient the loss wants to push into thought position k's input
# o[k] = model's output at thought position k (live, in graph)
# These are offset by 1: g[k] should flow into o[k-1]

g = [t.grad.detach().clone() for t in thought_inputs]

for d in range(D):
    # Stitching loss: redirect each g[k] into o[k-1]
    L_stitch = sum(
        (g[k] * thought_outputs[k - 1]).sum()
        for k in range(1, K + 1)
    )
    L_stitch.backward(retain_graph=True)

    # Update g with corrected gradients for next depth
    g = [t.grad.detach().clone() for t in thought_inputs]
```

4. **retain_graph=True on all backward calls except the last.** The final stitching iteration (or the standard backward if D=0) can release the graph.

### What to remove

- Sequential forward passes with gradients during thought generation
- Any gradient checkpointing infrastructure for the thought chain
- Any custom KV cache management for gradient flow

### What to watch for

- **thought_inputs must have `requires_grad=True`** before Phase 2, otherwise PyTorch won't compute `∂L/∂(t_k)`.
- **thought_outputs must stay attached to the graph.** Don't accidentally detach them when extracting from model outputs. No `.detach()`, no `.data`, no numpy conversion.
- **Zero gradients on thought_inputs between stitching iterations** if you're reading `.grad` — or accumulate into a separate buffer. PyTorch accumulates by default, so `t_k.grad` after iteration d contains the sum of all depths 0 through d. If you want just depth-d for the next iteration's g, subtract or zero appropriately.
- **The first thought position.** `g[0]` wants to flow into the question encoding, not a previous thought. Handle this boundary — either let it flow into the question outputs (trains the model to encode questions better for latent reasoning) or skip it.

## Memory and Compute Profile

**Memory:** One parallel forward pass over (question_len + K + answer_len) tokens through Gemma 2 2B, retained for all backward passes. Independent of D. Comparable to standard training on a sequence of the same length.

**Compute:**
- K cheap forward passes (no_grad, with KV cache)
- 1 full forward pass with grad
- (1 + D) backward passes through the same graph

For D = K = 64, that's 65 backward passes, which is expensive in wall time. But each is the same cost, and memory doesn't grow. Start here for correctness, then ablate D down.

For D = 5, total backward cost is 6x a standard training step on the same sequence length. Forward cost is dominated by the K no_grad generation steps, which are cheap individually but sequential.

## Validation Strategy

1. **Gradient correctness check (small scale).** With K = 3-4, compare W.grad from the stitching approach (D = K) against the current sequential-with-grad approach. They should match to floating point tolerance. Use `torch.allclose` on a few batches.

2. **Stitching depth ablation.** Train with D = K, then D = K/2, K/4, 5, 3, 1, 0. Plot training loss curves. The hypothesis is that curves converge for small D, confirming gradient vanishing across thought steps.

3. **Gradient norm by depth.** Log the norm of the stitching loss gradient at each depth d. If it decays geometrically, that directly measures the vanishing rate and tells you exactly where to truncate.