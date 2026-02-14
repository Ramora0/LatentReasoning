import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

class LatentReasoningModel(nn.Module):
    """Three-phase latent reasoning model built on Gemma 2 2B.

    Phase 1 (Encode): Embed question tokens.
    Phase 2 (Latent): K autoregressive latent steps (generated no-grad for training).
    Phase 3 (Decode): Teacher-forced answer decoding with direct lm_head projection.

    During training, gradient stitching is used: thoughts are generated without
    gradients, then a single full-sequence forward pass provides the computation
    graph. The trainer performs D stitching iterations to propagate gradients
    across thought boundaries.
    """

    def __init__(self, config):
        super().__init__()

        # XLA/TPU works best with eager attention; CUDA benefits from SDPA
        backend = getattr(config.distributed, "backend", "cuda")
        self.use_xla = backend == "xla"
        attn_impl = "eager" if self.use_xla else "sdpa"

        # XLA FSDP requires fp32 params (it casts to compute_dtype internally)
        model_dtype = torch.float32 if backend == "xla" else torch.bfloat16

        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            torch_dtype=model_dtype,
            attn_implementation=attn_impl,
        )

        # On XLA, Gemma 2's HybridCache uses SlidingWindowCache which
        # slices keys[:, :, -(sw-1):, :]. XLA enforces strict bounds on
        # negative indices (unlike CUDA which silently clamps), so this
        # crashes when seq_len < sliding_window. We bypass HybridCache
        # entirely by passing a plain DynamicCache in _run_transformer.
        # This is safe because our sequences are always << 4096 so
        # sliding window vs global attention gives identical results.

        self.d_model = self.base_model.config.hidden_size  # 2304 for Gemma 2 2B
        self.normalizer = math.sqrt(self.d_model)

        # Freeze embedding and lm_head permanently (they share tied weights)
        self.base_model.model.embed_tokens.weight.requires_grad_(False)
        if hasattr(self.base_model, "lm_head"):
            for param in self.base_model.lm_head.parameters():
                param.requires_grad_(False)

        # All transformer layer params stay trainable (requires_grad=True by default)

        # Learned transition tokens initialized from semantic embeddings (scaled by
        # normalizer) so they start at the same magnitude as all other positions.
        _tokenizer = AutoTokenizer.from_pretrained(config.model.name)

        # <thinking> token inserted between question encoding and latent steps
        _thinking_id = _tokenizer.encode("thinking", add_special_tokens=False)[0]
        _thinking_emb = self.base_model.model.embed_tokens.weight[_thinking_id].detach().clone()
        self.thinking_token_emb = nn.Parameter(
            (_thinking_emb * self.normalizer).reshape(1, 1, self.d_model)
        )

        # <answer> token inserted between thought vectors and answer decoding
        _answer_id = _tokenizer.encode("answer", add_special_tokens=False)[0]
        _init_emb = self.base_model.model.embed_tokens.weight[_answer_id].detach().clone()
        self.answer_token_emb = nn.Parameter(
            (_init_emb * self.normalizer).reshape(1, 1, self.d_model)
        )

        # Check for final logit softcapping (Gemma 2 feature)
        self.final_logit_softcapping = getattr(
            self.base_model.config, "final_logit_softcapping", None
        )

    @property
    def embed_tokens(self):
        return self.base_model.model.embed_tokens

    @property
    def lm_head(self):
        return self.base_model.lm_head

    @property
    def transformer(self):
        return self.base_model.model

    def _run_transformer(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor,
                         past_key_values=None, use_cache=False):
        """Run inputs through the transformer, pre-dividing by normalizer.

        Gemma 2 internally multiplies inputs_embeds by sqrt(d_model). By pre-dividing,
        we cancel this scaling so our already-scaled embeddings pass through correctly.

        When use_cache=True, returns (hidden_states, past_key_values).
        Otherwise returns just hidden_states.
        """
        # On XLA, supply a plain DynamicCache to prevent HF from creating
        # a HybridCache (whose SlidingWindowCache crashes on XLA when
        # seq_len < sliding_window due to strict negative-index bounds).
        if use_cache and past_key_values is None and self.use_xla:
            past_key_values = DynamicCache()
        outputs = self.transformer(
            inputs_embeds=inputs_embeds / self.normalizer,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if use_cache:
            return outputs.last_hidden_state, outputs.past_key_values
        return outputs.last_hidden_state

    def phase1_encode(self, question_ids: torch.Tensor) -> torch.Tensor:
        """Phase 1: Embed question tokens (no transformer pass).

        Args:
            question_ids: [B, q_len] token IDs

        Returns:
            hidden_states: [B, q_len, d_model] scaled embeddings
        """
        embeds = self.embed_tokens(question_ids) * self.normalizer
        return embeds

    @torch.no_grad()
    def _generate_thoughts_no_grad(self, hidden_states, attention_mask, K, p):
        """Generate K thought vectors without gradients using KV cache.

        Used during training to cheaply produce thought vectors before the
        full-sequence forward pass with gradient stitching.

        Returns:
            list of K tensors, each [B, 1, d_model]
        """
        B = hidden_states.shape[0]
        device = hidden_states.device
        thoughts = []

        # Step 0: full pass, build KV cache
        t0 = time.perf_counter()
        print(f"  [phase2-nograd] step 0/{K}: full transformer pass (seq_len={hidden_states.shape[1]})...", flush=True)
        hidden_out, kv_cache = self._run_transformer(
            hidden_states, attention_mask, use_cache=True,
        )
        last_hidden = hidden_out[:, -1:, :]
        del hidden_out

        logits = self.lm_head(last_hidden)
        token_ids = logits.argmax(dim=-1)
        token_emb = self.embed_tokens(token_ids) * self.normalizer
        blended = token_emb * p + last_hidden * (1 - p)
        thoughts.append(blended)
        print(f"  [phase2-nograd] step 0/{K}: done ({time.perf_counter()-t0:.2f}s)", flush=True)

        attention_mask = torch.cat(
            [attention_mask, torch.ones(B, 1, dtype=attention_mask.dtype, device=device)],
            dim=1,
        )

        # Steps 1..K-1: single-token passes with KV cache
        for k in range(1, K):
            tk = time.perf_counter()
            hidden_out, kv_cache = self._run_transformer(
                blended, attention_mask, past_key_values=kv_cache, use_cache=True,
            )
            last_hidden = hidden_out[:, -1:, :]

            logits = self.lm_head(last_hidden)
            token_ids = logits.argmax(dim=-1)
            token_emb = self.embed_tokens(token_ids) * self.normalizer
            blended = token_emb * p + last_hidden * (1 - p)
            thoughts.append(blended)
            print(f"  [phase2-nograd] step {k}/{K}: done ({time.perf_counter()-tk:.2f}s)", flush=True)

            attention_mask = torch.cat(
                [attention_mask, torch.ones(B, 1, dtype=attention_mask.dtype, device=device)],
                dim=1,
            )

        return thoughts

    def _prepare_stitching_inputs(self, thoughts):
        """Clone and detach thought vectors, enabling gradient tracking.

        Each thought is detached from the no-grad generation graph and given
        requires_grad=True so the trainer can extract gradients for stitching.

        Returns:
            list of K tensors, each [B, 1, d_model] with requires_grad=True
        """
        return [t.detach().clone().requires_grad_(True) for t in thoughts]

    def _phase2_core(self, hidden_states, attention_mask, p, K):
        """Core phase 2 for inference: K autoregressive latent steps with KV cache.

        Disables HF gradient checkpointing internally so KV cache works.

        Returns:
            (thoughts, kv_cache): thoughts is [B, K, d_model]
        """
        B = hidden_states.shape[0]
        device = hidden_states.device
        thoughts = []

        # Step 0: full pass, build KV cache
        t0 = time.perf_counter()
        print(f"  [phase2] step 0/{K}: full transformer pass (seq_len={hidden_states.shape[1]})...", flush=True)
        hidden_out, kv_cache = self._run_transformer(
            hidden_states, attention_mask, use_cache=True,
        )
        last_hidden = hidden_out[:, -1:, :]
        del hidden_out

        with torch.no_grad():
            logits = self.lm_head(last_hidden)
            token_ids = logits.argmax(dim=-1)
            token_emb = self.embed_tokens(token_ids) * self.normalizer
        blended = token_emb * p + last_hidden * (1 - p)
        thoughts.append(blended)
        print(f"  [phase2] step 0/{K}: thought appended ({time.perf_counter()-t0:.2f}s)", flush=True)

        attention_mask = torch.cat(
            [attention_mask, torch.ones(B, 1, dtype=attention_mask.dtype, device=device)],
            dim=1,
        )

        # Steps 1..K-1: single-token passes with KV cache
        for k in range(1, K):
            tk = time.perf_counter()
            hidden_out, kv_cache = self._run_transformer(
                blended, attention_mask, past_key_values=kv_cache, use_cache=True,
            )
            last_hidden = hidden_out[:, -1:, :]

            with torch.no_grad():
                logits = self.lm_head(last_hidden)
                token_ids = logits.argmax(dim=-1)
                token_emb = self.embed_tokens(token_ids) * self.normalizer
            blended = token_emb * p + last_hidden * (1 - p)
            thoughts.append(blended)
            print(f"  [phase2] step {k}/{K}: thought appended ({time.perf_counter()-tk:.2f}s)", flush=True)

            attention_mask = torch.cat(
                [attention_mask, torch.ones(B, 1, dtype=attention_mask.dtype, device=device)],
                dim=1,
            )

        return torch.cat(thoughts, dim=1), kv_cache  # [B, K, d_model]

    def phase2_latent_steps(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        K: int,
        p: float,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        """Phase 2: K autoregressive latent steps.

        Training: thoughts are generated without gradients (fast, KV cache),
        then detached copies with requires_grad=True are concatenated into the
        sequence. Gradient stitching in the trainer propagates gradients across
        thought boundaries via the full-sequence Phase 3 forward pass.

        Inference: direct execution, KV cache returned for autoregressive decoding.

        Args:
            hidden_states: [B, q_len, d_model] from Phase 1
            attention_mask: [B, q_len]
            K: number of latent iterations
            p: interpolation weight (1.0 = pure token, 0.0 = pure latent)

        Returns:
            (hidden_states, extended_mask, kv_cache):
                hidden_states: [B, q_len + K, d_model]
                extended_mask: [B, q_len + K]
                kv_cache: past_key_values (None during training)
        """
        B = hidden_states.shape[0]
        device = hidden_states.device

        if self.training:
            # Generate thoughts without gradients (fast, KV cache)
            raw_thoughts = self._generate_thoughts_no_grad(hidden_states, attention_mask, K, p)

            # Prepare stitching inputs (detached, requires_grad=True)
            thought_inputs = self._prepare_stitching_inputs(raw_thoughts)
            self._thought_inputs = thought_inputs

            # Stack thoughts for concatenation
            thoughts = torch.cat(thought_inputs, dim=1)  # [B, K, d_model]
            kv_cache = None
        else:
            # Inference: direct execution with KV cache
            thoughts, kv_cache = self._phase2_core(
                hidden_states, attention_mask, p, K,
            )

        # Extend mask and concatenate
        extended_mask = torch.cat(
            [attention_mask, torch.ones(B, K, dtype=attention_mask.dtype, device=device)],
            dim=1,
        )
        hidden_states = torch.cat([hidden_states, thoughts], dim=1)

        return hidden_states, extended_mask, kv_cache

    def phase3_decode(
        self,
        latent_states: torch.Tensor,
        prefix_mask: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor,
        kv_cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase 3: Teacher-forced answer decoding.

        The prefix (question + thought vectors + <answer> token) provides context.
        Answer hidden states are projected directly to vocabulary via lm_head.

        When kv_cache is provided, only the <answer> token and answer embeddings are
        processed (the prefix is already in the cache). Otherwise falls back to
        processing the full concatenated sequence.

        Args:
            latent_states: [B, prefix_len, d_model] (question + K thoughts + <answer>)
            prefix_mask: [B, prefix_len]
            answer_ids: [B, a_len] answer token IDs
            answer_mask: [B, a_len]
            kv_cache: optional past_key_values from phase2

        Returns:
            (loss, logits) where loss is CE on answer positions
        """
        # Teacher-forced answer embeddings (shifted right: use all but last token as input)
        answer_embeds = self.embed_tokens(answer_ids[:, :-1]) * self.normalizer  # [B, a_len-1, d_model]

        if kv_cache is not None:
            # KV cache covers question + K thoughts (NOT the <answer> token).
            # We need to process: <answer> token + answer_embeds
            # The <answer> token is the last position in latent_states.
            answer_marker = latent_states[:, -1:, :]  # [B, 1, d_model]
            decode_input = torch.cat([answer_marker, answer_embeds], dim=1)

            # Full attention mask must cover cached + new positions
            decode_mask = torch.cat(
                [prefix_mask, answer_mask[:, :-1]], dim=1
            )

            hidden_out = self._run_transformer(
                decode_input, decode_mask, past_key_values=kv_cache, use_cache=False
            )

            # Output covers only the new tokens: [<answer>, ans_0, ans_1, ...]
            # Skip <answer> position, take answer positions
            answer_hidden = hidden_out[:, 1:, :]  # [B, a_len-1, d_model]
        else:
            # No cache: process full sequence (used during training for gradient stitching)
            decode_input = torch.cat([latent_states, answer_embeds], dim=1)
            decode_mask = torch.cat(
                [prefix_mask, answer_mask[:, :-1]], dim=1
            )
            hidden_out = self._run_transformer(decode_input, decode_mask)
            prefix_len = latent_states.shape[1]
            answer_hidden = hidden_out[:, prefix_len:, :]  # [B, a_len-1, d_model]

            # Extract thought outputs for gradient stitching (training only).
            # thought_outputs[k] is the hidden state at the position that "produced"
            # thought k: <thinking> output produces t_0, t_0 output produces t_1, etc.
            if self.training and hasattr(self, '_thought_output_start'):
                start = self._thought_output_start
                K = self._K_for_stitching
                self._thought_outputs = [
                    hidden_out[:, start + k : start + k + 1, :]
                    for k in range(K)
                ]

        # Project directly to vocabulary
        logits = self.lm_head(answer_hidden)  # [B, a_len-1, vocab]

        # Apply final logit softcapping if present (Gemma 2)
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)

        # Compute cross-entropy loss
        # Labels are answer_ids shifted left (predict next token)
        labels = answer_ids[:, 1:].clone()  # [B, a_len-1]
        # Set padding positions to -100
        label_mask = answer_mask[:, 1:]
        labels[label_mask == 0] = -100

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        return loss, logits

    def forward(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor,
        K: int,
        p: float,
    ) -> dict:
        """Full forward pass: encode -> latent steps -> <answer> -> decode.

        Returns:
            dict with 'loss', 'logits', and during training:
            'thought_inputs' (list of K leaf tensors with requires_grad=True)
            'thought_outputs' (list of K tensors live in the computation graph)
        """
        # Phase 1: Encode
        t_start = time.perf_counter()
        print(f"[forward] Phase 1: Encoding (q_len={question_ids.shape[1]}, batch={question_ids.shape[0]})...", flush=True)
        hidden_states = self.phase1_encode(question_ids)

        # Store original question length for thought output extraction in phase3
        q_len_orig = hidden_states.shape[1]

        # Insert <thinking> token before latent steps
        B = hidden_states.shape[0]
        device = hidden_states.device
        thinking_marker = self.thinking_token_emb.expand(B, -1, -1)  # [B, 1, d_model]
        hidden_states = torch.cat([hidden_states, thinking_marker], dim=1)
        question_mask = torch.cat(
            [question_mask, torch.ones(B, 1, dtype=question_mask.dtype, device=device)],
            dim=1,
        )
        print(f"[forward] Phase 1: done ({time.perf_counter()-t_start:.2f}s)", flush=True)

        # Store position info for thought output extraction in phase3.
        # In the full sequence [q_tokens, <thinking>, t_0..t_{K-1}, <answer>, ans...]:
        # - <thinking> is at position q_len_orig
        # - thought_outputs[0] = hidden_out at <thinking> position (produces t_0)
        # - thought_outputs[k] = hidden_out at t_{k-1} position (produces t_k)
        self._thought_output_start = q_len_orig
        self._K_for_stitching = K

        # Phase 2: Latent steps
        t_phase2 = time.perf_counter()
        print(f"[forward] Phase 2: Latent steps (K={K}, p={p:.3f})...", flush=True)
        hidden_states, extended_mask, _ = self.phase2_latent_steps(
            hidden_states, question_mask, K, p
        )
        print(f"[forward] Phase 2: done ({time.perf_counter()-t_phase2:.2f}s)", flush=True)

        # Insert <answer> token
        answer_marker = self.answer_token_emb.expand(B, -1, -1)  # [B, 1, d_model]
        hidden_states = torch.cat([hidden_states, answer_marker], dim=1)
        extended_mask = torch.cat(
            [extended_mask, torch.ones(B, 1, dtype=extended_mask.dtype, device=device)],
            dim=1,
        )

        # Phase 3: Decode — full sequence with gradient checkpointing (no KV cache).
        t_phase3 = time.perf_counter()
        print(f"[forward] Phase 3: Decoding (a_len={answer_ids.shape[1]})...", flush=True)
        loss, logits = self.phase3_decode(
            hidden_states, extended_mask, answer_ids, answer_mask,
        )
        print(f"[forward] Phase 3: done ({time.perf_counter()-t_phase3:.2f}s)", flush=True)
        print(f"[forward] Total forward: {time.perf_counter()-t_start:.2f}s, loss={loss.item():.4f}", flush=True)

        result = {"loss": loss, "logits": logits}

        # Include stitching tensors during training
        if self.training:
            result["thought_inputs"] = getattr(self, '_thought_inputs', None)
            result["thought_outputs"] = getattr(self, '_thought_outputs', None)

        return result

    @torch.no_grad()
    def generate_answer(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        K: int,
        p: float,
        max_new_tokens: int = 64,
    ) -> torch.Tensor:
        """Inference: encode -> latent steps -> <answer> -> autoregressive greedy decoding.

        Args:
            question_ids: [B, q_len]
            question_mask: [B, q_len]
            K: latent iterations
            p: interpolation weight
            max_new_tokens: max tokens to generate

        Returns:
            generated_ids: [B, max_new_tokens] generated token IDs
        """
        B = question_ids.shape[0]
        device = question_ids.device
        eos_id = self.base_model.config.eos_token_id
        if isinstance(eos_id, list):
            eos_id = eos_id[0]

        # Phase 1: Encode
        hidden_states = self.phase1_encode(question_ids)

        # Insert <thinking> token
        thinking_marker = self.thinking_token_emb.expand(B, -1, -1)
        hidden_states = torch.cat([hidden_states, thinking_marker], dim=1)
        question_mask = torch.cat(
            [question_mask, torch.ones(B, 1, dtype=question_mask.dtype, device=device)],
            dim=1,
        )

        # Phase 2: Latent steps with KV cache
        hidden_states, extended_mask, kv_cache = self.phase2_latent_steps(
            hidden_states, question_mask, K, p
        )

        # Process <answer> token through transformer with KV cache (1 token)
        answer_marker = self.answer_token_emb.expand(B, -1, -1)
        current_mask = torch.cat(
            [extended_mask, torch.ones(B, 1, dtype=extended_mask.dtype, device=device)],
            dim=1,
        )
        hidden_out, kv_cache = self._run_transformer(
            answer_marker, current_mask, past_key_values=kv_cache, use_cache=True
        )

        # Get logits from <answer> position
        last_hidden = hidden_out[:, -1:, :]
        logits = self.lm_head(last_hidden)
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)

        next_token = logits.argmax(dim=-1)  # [B, 1]
        generated = [next_token]

        for step in range(1, max_new_tokens):
            # Check for EOS
            if eos_id is not None and (next_token == eos_id).all():
                break

            # Process only the new token with cached KV
            next_emb = self.embed_tokens(next_token) * self.normalizer  # [B, 1, d_model]
            current_mask = torch.cat(
                [current_mask, torch.ones(B, 1, dtype=current_mask.dtype, device=device)],
                dim=1,
            )
            hidden_out, kv_cache = self._run_transformer(
                next_emb, current_mask, past_key_values=kv_cache, use_cache=True
            )

            last_hidden = hidden_out[:, -1:, :]
            logits = self.lm_head(last_hidden)
            if self.final_logit_softcapping is not None:
                cap = self.final_logit_softcapping
                logits = cap * torch.tanh(logits / cap)

            next_token = logits.argmax(dim=-1)
            generated.append(next_token)

        if generated:
            return torch.cat(generated, dim=1)  # [B, num_generated]
        return torch.zeros(B, 0, dtype=torch.long, device=device)
