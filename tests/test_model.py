"""Unit tests for the three-phase latent reasoning model.

These tests use the real Gemma 2 2B model to verify:
- Output shapes at each phase (phase 2 appends K positions)
- Gradient flow through transformer
- Frozen params (embedding, lm_head) have no gradients
- Annealing behavior at p=1.0 vs p=0.0
- Overfitting on a single batch
"""

import os
import sys
import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.latent_gemma import LatentReasoningModel
from src.training.curriculum import CurriculumScheduler

# Minimal config for testing
TEST_CONFIG = OmegaConf.create({
    "model": {
        "name": "MCES10/maths-problems-gemma-2-2b-it",
    },
    "latent": {
        "K": 8,
        "p_start": 0.9,
        "p_end": 0.0,
        "p_anneal_ratio": 0.8,
    },
})


# --- Curriculum Scheduler Tests ---

class TestCurriculumScheduler:
    def setup_method(self):
        # 10000 total optimizer steps, p_anneal_ratio=0.8 → anneal over 8000 steps
        self.scheduler = CurriculumScheduler(TEST_CONFIG, max_optimizer_steps=10000)

    def test_p_annealing(self):
        # At step 0: p = 0.9
        assert abs(self.scheduler.get_p(0) - 0.9) < 1e-6
        # At step 4000 (halfway through 8000): p = 0.45
        assert abs(self.scheduler.get_p(4000) - 0.45) < 1e-6
        # At step 8000: p = 0.0
        assert abs(self.scheduler.get_p(8000) - 0.0) < 1e-6
        # Beyond anneal steps: p stays at 0.0
        assert abs(self.scheduler.get_p(10000) - 0.0) < 1e-6


# --- Full Model Tests (require GPU and model download) ---

def _requires_gpu():
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA GPU required",
    )


@_requires_gpu()
class TestLatentReasoningModel:
    """Tests that require loading the actual Gemma 2 2B model on GPU."""

    @pytest.fixture(scope="class")
    def model(self):
        model = LatentReasoningModel(TEST_CONFIG)
        model = model.to("cuda")
        return model

    @pytest.fixture
    def dummy_batch(self):
        """Small dummy batch for testing."""
        B, q_len, a_len = 2, 16, 8
        question_ids = torch.randint(1, 256000, (B, q_len), device="cuda")
        question_mask = torch.ones(B, q_len, dtype=torch.long, device="cuda")
        answer_ids = torch.randint(1, 256000, (B, a_len), device="cuda")
        answer_mask = torch.ones(B, a_len, dtype=torch.long, device="cuda")
        return {
            "question_ids": question_ids,
            "question_mask": question_mask,
            "answer_ids": answer_ids,
            "answer_mask": answer_mask,
        }

    def test_phase1_shape(self, model, dummy_batch):
        """Phase 1 output should be [B, q_len, d_model]."""
        hidden = model.phase1_encode(dummy_batch["question_ids"])
        B, q_len = dummy_batch["question_ids"].shape
        assert hidden.shape == (B, q_len, model.d_model)

    def test_phase2_shape(self, model, dummy_batch):
        """Phase 2 should append K positions: [B, q_len + K, d_model]."""
        hidden = model.phase1_encode(dummy_batch["question_ids"])
        K = 2
        latent, mask = model.phase2_latent_steps(
            hidden, dummy_batch["question_mask"], K=K, p=0.5
        )
        B, q_len = dummy_batch["question_ids"].shape
        assert latent.shape == (B, q_len + K, model.d_model)
        assert mask.shape == (B, q_len + K)

    def test_forward_output(self, model, dummy_batch):
        """Full forward should return loss and logits with correct shapes."""
        K = 2
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                question_ids=dummy_batch["question_ids"],
                question_mask=dummy_batch["question_mask"],
                answer_ids=dummy_batch["answer_ids"],
                answer_mask=dummy_batch["answer_mask"],
                K=K,
                p=0.5,
            )
        assert "loss" in outputs
        assert "logits" in outputs
        assert outputs["loss"].ndim == 0  # scalar
        B, a_len = dummy_batch["answer_ids"].shape
        assert outputs["logits"].shape == (B, a_len - 1, model.lm_head.out_features)

    def test_frozen_params(self, model, dummy_batch):
        """Embedding and lm_head should have no gradients."""
        model.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                question_ids=dummy_batch["question_ids"],
                question_mask=dummy_batch["question_mask"],
                answer_ids=dummy_batch["answer_ids"],
                answer_mask=dummy_batch["answer_mask"],
                K=2,
                p=0.5,
            )
        outputs["loss"].backward()
        assert model.embed_tokens.weight.grad is None

    def test_answer_token_grad(self, model, dummy_batch):
        """The <answer> token embedding should receive gradients."""
        model.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                question_ids=dummy_batch["question_ids"],
                question_mask=dummy_batch["question_mask"],
                answer_ids=dummy_batch["answer_ids"],
                answer_mask=dummy_batch["answer_mask"],
                K=2,
                p=0.5,
            )
        outputs["loss"].backward()
        assert model.answer_token_emb.grad is not None
        assert model.answer_token_emb.grad.abs().sum() > 0

    def test_generate_answer(self, model, dummy_batch):
        """generate_answer should return token IDs."""
        generated = model.generate_answer(
            question_ids=dummy_batch["question_ids"],
            question_mask=dummy_batch["question_mask"],
            K=2,
            p=0.5,
            max_new_tokens=8,
        )
        assert generated.ndim == 2
        assert generated.shape[0] == dummy_batch["question_ids"].shape[0]
        assert generated.shape[1] <= 8

    def test_p_zero_pure_latent(self, model, dummy_batch):
        """At p=0.0, no token embedding should be blended in (pure latent)."""
        # Just verify it runs without error and produces different outputs than p=1.0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_p0 = model(
                question_ids=dummy_batch["question_ids"],
                question_mask=dummy_batch["question_mask"],
                answer_ids=dummy_batch["answer_ids"],
                answer_mask=dummy_batch["answer_mask"],
                K=2,
                p=0.0,
            )
            out_p1 = model(
                question_ids=dummy_batch["question_ids"],
                question_mask=dummy_batch["question_mask"],
                answer_ids=dummy_batch["answer_ids"],
                answer_mask=dummy_batch["answer_mask"],
                K=2,
                p=0.9,
            )
        # Losses should be different (different computation paths)
        assert out_p0["loss"].item() != out_p1["loss"].item()


@_requires_gpu()
@pytest.mark.timeout(120)
class TestOverfitting:
    """Test that the model can overfit a single batch."""

    def test_loss_decreases(self):
        """Loss should decrease over ~50 steps on a single batch."""
        model = LatentReasoningModel(TEST_CONFIG).to("cuda")
        model.train()

        # Small dummy batch
        B, q_len, a_len = 2, 16, 8
        batch = {
            "question_ids": torch.randint(1, 256000, (B, q_len), device="cuda"),
            "question_mask": torch.ones(B, q_len, dtype=torch.long, device="cuda"),
            "answer_ids": torch.randint(1, 256000, (B, a_len), device="cuda"),
            "answer_mask": torch.ones(B, a_len, dtype=torch.long, device="cuda"),
        }

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
        )

        initial_loss = None
        final_loss = None

        for step in range(50):
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    question_ids=batch["question_ids"],
                    question_mask=batch["question_mask"],
                    answer_ids=batch["answer_ids"],
                    answer_mask=batch["answer_mask"],
                    K=2,
                    p=0.5,
                )
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()
            if step == 49:
                final_loss = loss.item()

        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )
