import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class LatentReasoningModel(nn.Module):
    """Three-phase latent reasoning model built on Gemma 2 2B.

    Phase 1 (Encode): Embed question tokens.
    Phase 2 (Latent): K autoregressive latent steps (each appends one position).
    Phase 3 (Decode): Teacher-forced answer decoding with direct lm_head projection.
    """

    def __init__(self, config):
        super().__init__()

        # XLA/TPU works best with eager attention; CUDA benefits from SDPA
        backend = getattr(config.distributed, "backend", "cuda")
        attn_impl = "eager" if backend == "xla" else "sdpa"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )

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

        # Per-step gradient norms populated by backward hooks in phase2
        self._latent_grad_norms = {}

        # Enable gradient checkpointing on the base model
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Check for final logit softcapping (Gemma 2 feature)
        self.final_logit_softcapping = getattr(
            self.base_model.config, "final_logit_softcapping", None
        )

    def _disable_layer_grad_checkpointing(self):
        """Disable per-layer gradient checkpointing to allow KV cache."""
        self._saved_grad_ckpt = {}
        for i, layer in enumerate(self.base_model.model.layers):
            self._saved_grad_ckpt[i] = getattr(layer, 'gradient_checkpointing', False)
            layer.gradient_checkpointing = False

    def _restore_layer_grad_checkpointing(self):
        """Restore per-layer gradient checkpointing flags."""
        for i, layer in enumerate(self.base_model.model.layers):
            layer.gradient_checkpointing = self._saved_grad_ckpt.get(i, False)

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

    def phase2_latent_steps(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        K: int,
        p: float,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        """Phase 2: K autoregressive latent steps with KV cache.

        Step 0 runs the full transformer on all input positions with use_cache=True.
        Steps 1..K-1 each process only the new thought vector (1 token) with cached KV.

        Args:
            hidden_states: [B, q_len, d_model] from Phase 1
            attention_mask: [B, q_len]
            K: number of latent iterations
            p: interpolation weight (1.0 = pure token, 0.0 = pure latent)

        Returns:
            (hidden_states, extended_mask, kv_cache):
                hidden_states: [B, q_len + K, d_model]
                extended_mask: [B, q_len + K]
                kv_cache: past_key_values from transformer
        """
        B = hidden_states.shape[0]
        device = hidden_states.device
        self._latent_grad_norms = {}
        thoughts = []

        self._disable_layer_grad_checkpointing()
        try:
            # Step 0: full pass over all input positions, build KV cache
            hidden_out, kv_cache = self._run_transformer(
                hidden_states, attention_mask, use_cache=True
            )
            last_hidden = hidden_out[:, -1:, :]  # [B, 1, d_model]

            # Blend step 0
            with torch.no_grad():
                logits = self.lm_head(last_hidden)
                token_ids = logits.argmax(dim=-1)
                token_emb = self.embed_tokens(token_ids) * self.normalizer
            blended = token_emb * p + last_hidden * (1 - p)

            def _make_grad_hook(step):
                def _grad_hook(grad):
                    self._latent_grad_norms[step] = grad.norm().item()
                return _grad_hook
            blended.register_hook(_make_grad_hook(0))
            thoughts.append(blended)

            # Extend mask for thought_0
            attention_mask = torch.cat(
                [attention_mask, torch.ones(B, 1, dtype=attention_mask.dtype, device=device)],
                dim=1,
            )

            # Steps 1..K-1: process only the new thought vector (1 token)
            for k in range(1, K):
                # Feed only the latest thought vector, with full mask
                hidden_out, kv_cache = self._run_transformer(
                    blended, attention_mask, past_key_values=kv_cache, use_cache=True
                )
                last_hidden = hidden_out[:, -1:, :]  # [B, 1, d_model]

                with torch.no_grad():
                    logits = self.lm_head(last_hidden)
                    token_ids = logits.argmax(dim=-1)
                    token_emb = self.embed_tokens(token_ids) * self.normalizer
                blended = token_emb * p + last_hidden * (1 - p)

                blended.register_hook(_make_grad_hook(k))
                thoughts.append(blended)

                # Extend mask for this thought
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(B, 1, dtype=attention_mask.dtype, device=device)],
                    dim=1,
                )

        finally:
            self._restore_layer_grad_checkpointing()

        # Concatenate all thought vectors and append to original hidden_states
        all_thoughts = torch.cat(thoughts, dim=1)  # [B, K, d_model]
        hidden_states = torch.cat([hidden_states, all_thoughts], dim=1)

        return hidden_states, attention_mask, kv_cache

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

            self._disable_layer_grad_checkpointing()
            try:
                hidden_out = self._run_transformer(
                    decode_input, decode_mask, past_key_values=kv_cache, use_cache=False
                )
            finally:
                self._restore_layer_grad_checkpointing()

            # Output covers only the new tokens: [<answer>, ans_0, ans_1, ...]
            # Skip <answer> position, take answer positions
            answer_hidden = hidden_out[:, 1:, :]  # [B, a_len-1, d_model]
        else:
            # Fallback: no cache, process full sequence
            decode_input = torch.cat([latent_states, answer_embeds], dim=1)
            decode_mask = torch.cat(
                [prefix_mask, answer_mask[:, :-1]], dim=1
            )
            hidden_out = self._run_transformer(decode_input, decode_mask)
            prefix_len = latent_states.shape[1]
            answer_hidden = hidden_out[:, prefix_len:, :]  # [B, a_len-1, d_model]

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
            dict with 'loss' and 'logits'
        """
        # Phase 1: Encode
        hidden_states = self.phase1_encode(question_ids)

        # Insert <thinking> token before latent steps
        B = hidden_states.shape[0]
        device = hidden_states.device
        thinking_marker = self.thinking_token_emb.expand(B, -1, -1)  # [B, 1, d_model]
        hidden_states = torch.cat([hidden_states, thinking_marker], dim=1)
        question_mask = torch.cat(
            [question_mask, torch.ones(B, 1, dtype=question_mask.dtype, device=device)],
            dim=1,
        )

        # Phase 2: Latent steps with KV cache
        hidden_states, extended_mask, kv_cache = self.phase2_latent_steps(
            hidden_states, question_mask, K, p
        )

        # Insert <answer> token
        answer_marker = self.answer_token_emb.expand(B, -1, -1)  # [B, 1, d_model]
        hidden_states = torch.cat([hidden_states, answer_marker], dim=1)
        extended_mask = torch.cat(
            [extended_mask, torch.ones(B, 1, dtype=extended_mask.dtype, device=device)],
            dim=1,
        )

        # Phase 3: Decode with cached prefix
        loss, logits = self.phase3_decode(
            hidden_states, extended_mask, answer_ids, answer_mask,
            kv_cache=kv_cache,
        )

        return {"loss": loss, "logits": logits}

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
