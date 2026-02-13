"""Unit tests for the three-phase latent reasoning model.

These tests use the real Gemma 2 2B model to verify:
- Output shapes at each phase
- Gradient flow through bridge and transformer
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
from src.model.bridge import BridgeLayer
from src.training.curriculum import CurriculumScheduler

# Minimal config for testing
TEST_CONFIG = OmegaConf.create({
    "model": {
        "name": "MCES10/maths-problems-gemma-2-2b-it",
        "bridge": {"identity_init_scale": 0.01},
    },
    "latent": {
        "initial_K": 2,
        "max_K": 8,
        "k_schedule": [[0, 2], [10000, 4], [25000, 8]],
        "p_start": 0.9,
        "p_end": 0.0,
        "p_anneal_steps": 50000,
    },
})


# --- Bridge Layer Tests (no GPU needed) ---

class TestBridgeLayer:
    def test_output_shape(self):
        d_model = 64
        bridge = BridgeLayer(d_model=d_model)
        x = torch.randn(2, 10, d_model)
        out = bridge(x)
        assert out.shape == (2, 10, d_model)

    def test_near_identity_at_init(self):
        d_model = 64
        bridge = BridgeLayer(d_model=d_model, identity_init_scale=0.001)
        x = torch.randn(2, 10, d_model)
        out = bridge(x)
        # With near-identity init and LayerNorm, output should be close to LayerNorm(x)
        ln = bridge.layer_norm(x)
        diff = (out - ln).abs().max().item()
        assert diff < 0.5, f"Bridge output too far from near-identity: max diff = {diff}"

    def test_gradient_flow(self):
        d_model = 64
        bridge = BridgeLayer(d_model=d_model)
        x = torch.randn(2, 10, d_model, requires_grad=True)
        out = bridge(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert bridge.linear.weight.grad is not None


# --- Curriculum Scheduler Tests ---

class TestCurriculumScheduler:
    def setup_method(self):
        self.scheduler = CurriculumScheduler(TEST_CONFIG)

    def test_k_schedule(self):
        assert self.scheduler.get_K(0) == 2
        assert self.scheduler.get_K(5000) == 2
        assert self.scheduler.get_K(10000) == 4
        assert self.scheduler.get_K(15000) == 4
        assert self.scheduler.get_K(25000) == 8
        assert self.scheduler.get_K(50000) == 8

    def test_p_annealing(self):
        # At step 0: p = 0.9
        assert abs(self.scheduler.get_p(0) - 0.9) < 1e-6
        # At step 25000 (halfway): p = 0.45
        assert abs(self.scheduler.get_p(25000) - 0.45) < 1e-6
        # At step 50000: p = 0.0
        assert abs(self.scheduler.get_p(50000) - 0.0) < 1e-6
        # Beyond anneal steps: p stays at 0.0
        assert abs(self.scheduler.get_p(60000) - 0.0) < 1e-6


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
        """Phase 2 output should preserve shape."""
        hidden = model.phase1_encode(dummy_batch["question_ids"])
        latent = model.phase2_latent_steps(hidden, dummy_batch["question_mask"], K=2, p=0.5)
        assert latent.shape == hidden.shape

    def test_forward_output(self, model, dummy_batch):
        """Full forward should return loss and logits."""
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                question_ids=dummy_batch["question_ids"],
                question_mask=dummy_batch["question_mask"],
                answer_ids=dummy_batch["answer_ids"],
                answer_mask=dummy_batch["answer_mask"],
                K=2,
                p=0.5,
            )
        assert "loss" in outputs
        assert "logits" in outputs
        assert outputs["loss"].ndim == 0  # scalar
        assert outputs["logits"].shape[0] == dummy_batch["answer_ids"].shape[0]

    def test_gradient_flow_bridge(self, model, dummy_batch):
        """Verify gradients flow through bridge after backward."""
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
        assert model.bridge.linear.weight.grad is not None
        assert model.bridge.linear.weight.grad.abs().sum() > 0

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

        # Only optimize bridge + a few transformer params for speed
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
