import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from .bridge import BridgeLayer


class LatentReasoningModel(nn.Module):
    """Three-phase latent reasoning model built on Gemma 2 2B.

    Phase 1 (Encode): Embed question tokens.
    Phase 2 (Latent): K iterations of annealed interpolation + transformer pass.
    Phase 3 (Decode): Bridge + teacher-forced answer decoding.
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

        self.bridge = BridgeLayer(
            d_model=self.d_model,
            identity_init_scale=config.model.bridge.identity_init_scale,
        )

        # Freeze embedding and lm_head permanently (they share tied weights)
        self.base_model.model.embed_tokens.weight.requires_grad_(False)
        # lm_head is tied to embed_tokens in Gemma 2, but set explicitly in case untied
        if hasattr(self.base_model, "lm_head"):
            for p in self.base_model.lm_head.parameters():
                p.requires_grad_(False)

        # All transformer layer params stay trainable (requires_grad=True by default)

        # Enable gradient checkpointing on the base model
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
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

    def _run_transformer(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        """Run inputs through the transformer, pre-dividing by normalizer.

        Gemma 2 internally multiplies inputs_embeds by sqrt(d_model). By pre-dividing,
        we cancel this scaling so our already-scaled embeddings pass through correctly.
        """
        outputs = self.transformer(
            inputs_embeds=inputs_embeds / self.normalizer,
            attention_mask=attention_mask,
            use_cache=False,
        )
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
    ) -> torch.Tensor:
        """Phase 2: K iterations of annealed interpolation + transformer.

        Args:
            hidden_states: [B, seq_len, d_model] from Phase 1
            attention_mask: [B, seq_len]
            K: number of latent iterations
            p: interpolation weight (1.0 = pure token, 0.0 = pure latent)

        Returns:
            hidden_states: [B, seq_len, d_model] after K iterations
        """
        for _ in range(K):
            hidden_states = checkpoint(
                self._latent_step,
                hidden_states,
                attention_mask,
                torch.tensor(p),  # must be a tensor for checkpoint
                use_reentrant=False,
            )
        return hidden_states

    def _latent_step(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        p_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Single latent iteration: anneal interpolation + transformer pass."""
        p = p_tensor.item()

        # Annealed interpolation: blend discrete token embeddings with continuous latent
        with torch.no_grad():
            logits = self.lm_head(hidden_states)
            token_ids = logits.argmax(dim=-1)
            token_emb = self.embed_tokens(token_ids) * self.normalizer

        # Gradients flow only through hidden_states (the (1-p) branch)
        blended = token_emb * p + hidden_states * (1 - p)

        # Run through transformer
        hidden_states = self._run_transformer(blended, attention_mask)
        return hidden_states

    def phase3_decode(
        self,
        latent_states: torch.Tensor,
        question_mask: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase 3: Bridge + teacher-forced answer decoding.

        Args:
            latent_states: [B, q_len, d_model] from Phase 2
            question_mask: [B, q_len]
            answer_ids: [B, a_len] answer token IDs
            answer_mask: [B, a_len]

        Returns:
            (loss, logits) where loss is CE on answer positions
        """
        # Bridge latent states
        bridged = self.bridge(latent_states)  # [B, q_len, d_model]

        # Teacher-forced answer embeddings (shifted right: use all but last token as input)
        answer_embeds = self.embed_tokens(answer_ids[:, :-1]) * self.normalizer  # [B, a_len-1, d_model]

        # Concatenate bridged question states with answer embeddings
        decode_input = torch.cat([bridged, answer_embeds], dim=1)  # [B, q_len + a_len - 1, d_model]

        # Build attention mask for decode
        decode_mask = torch.cat(
            [question_mask, answer_mask[:, :-1]], dim=1
        )  # [B, q_len + a_len - 1]

        # Run through transformer (frozen weights but grads flow through activations)
        hidden_out = self._run_transformer(decode_input, decode_mask)

        # Extract answer-position hidden states
        q_len = latent_states.shape[1]
        answer_hidden = hidden_out[:, q_len:, :]  # [B, a_len-1, d_model]

        # Project to vocabulary
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
        """Full forward pass: encode -> latent steps -> decode.

        Returns:
            dict with 'loss' and 'logits'
        """
        # Phase 1: Encode
        hidden_states = self.phase1_encode(question_ids)

        # Phase 2: Latent iterations
        hidden_states = self.phase2_latent_steps(hidden_states, question_mask, K, p)

        # Phase 3: Decode
        loss, logits = self.phase3_decode(hidden_states, question_mask, answer_ids, answer_mask)

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
        """Inference: encode -> latent steps -> bridge -> autoregressive greedy decoding.

        Args:
            question_ids: [B, q_len]
            question_mask: [B, q_len]
            K: latent iterations
            p: interpolation weight
            max_new_tokens: max tokens to generate

        Returns:
            generated_ids: [B, max_new_tokens] generated token IDs
        """
        # Phase 1 + Phase 2
        hidden_states = self.phase1_encode(question_ids)
        hidden_states = self.phase2_latent_steps(hidden_states, question_mask, K, p)

        # Bridge
        bridged = self.bridge(hidden_states)  # [B, q_len, d_model]

        B = question_ids.shape[0]
        device = question_ids.device
        eos_id = self.base_model.config.eos_token_id

        generated = []
        current_input = bridged
        current_mask = question_mask.clone()

        for step in range(max_new_tokens):
            # Run transformer on current sequence
            hidden_out = self._run_transformer(current_input, current_mask)

            # Get logits for the last position
            last_hidden = hidden_out[:, -1:, :]  # [B, 1, d_model]
            logits = self.lm_head(last_hidden)    # [B, 1, vocab]

            if self.final_logit_softcapping is not None:
                cap = self.final_logit_softcapping
                logits = cap * torch.tanh(logits / cap)

            next_token = logits.argmax(dim=-1)  # [B, 1]
            generated.append(next_token)

            # Check for EOS
            if eos_id is not None and (next_token == eos_id).all():
                break

            # Prepare next input: append new token embedding
            next_emb = self.embed_tokens(next_token) * self.normalizer  # [B, 1, d_model]
            current_input = torch.cat([current_input, next_emb], dim=1)
            current_mask = torch.cat(
                [current_mask, torch.ones(B, 1, dtype=current_mask.dtype, device=device)],
                dim=1,
            )

        if generated:
            return torch.cat(generated, dim=1)  # [B, num_generated]
        return torch.zeros(B, 0, dtype=torch.long, device=device)
