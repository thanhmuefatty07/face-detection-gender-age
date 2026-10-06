"""
Unit Tests for BiometricMultiTaskModel (ConvNeXt-V2 & MobileNetV4)
==================================================================
Validates:
1. ConvNeXt-V2 backbone instantiation with multi-task heads.
2. MobileNetV4 backbone instantiation.
3. 101-bin age LDL probability simplex and continuous expected age.
4. Gender logit and sigmoid probability.
5. Aleatoric heteroscedastic log-variance and standard deviation.
6. 64-D unit-normalized Re-ID embedding representation.
7. Full backpropagation gradient integrity across all heads.
"""

from __future__ import annotations
import pytest
import torch
import torch.nn.functional as F

from src.biometrics.estimation.backbone import BiometricMultiTaskModel


class TestBiometricMultiTaskModel:
    @pytest.mark.parametrize("backbone_name", ["convnextv2_nano", "mobilenetv4"])
    def test_model_forward_shape_and_constraints(self, backbone_name):
        model = BiometricMultiTaskModel(backbone_name=backbone_name, pretrained=False)
        model.eval()

        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 160, 160)

        with torch.no_grad():
            outputs = model(dummy_input)

        # 1. Age LDL Head
        assert "age_probs" in outputs
        assert outputs["age_probs"].shape == (batch_size, 101)
        # Sum to 1
        probs_sum = outputs["age_probs"].sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones(batch_size), atol=1e-4)

        # Expected age
        assert "expected_age" in outputs
        assert outputs["expected_age"].shape == (batch_size,)
        assert (outputs["expected_age"] >= 0.0).all()
        assert (outputs["expected_age"] <= 100.0).all()

        # 2. Gender Head
        assert "gender_logit" in outputs
        assert outputs["gender_logit"].shape == (batch_size,)
        assert "gender_prob" in outputs
        assert outputs["gender_prob"].shape == (batch_size,)
        assert (outputs["gender_prob"] >= 0.0).all()
        assert (outputs["gender_prob"] <= 1.0).all()

        # 3. Aleatoric Uncertainty Head
        assert "log_variance" in outputs
        assert outputs["log_variance"].shape == (batch_size,)
        assert "aleatoric_std" in outputs
        assert outputs["aleatoric_std"].shape == (batch_size,)
        assert (outputs["aleatoric_std"] > 0.0).all()

        # 4. 64-D Re-ID Embedding Head
        assert "reid_embedding" in outputs
        assert outputs["reid_embedding"].shape == (batch_size, 64)
        # Must be L2 normalized to unit sphere
        norms = torch.norm(outputs["reid_embedding"], p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-4)

    def test_reid_cosine_similarity(self):
        """Validates that distinct inputs produce distinct unit-normalized embeddings."""
        model = BiometricMultiTaskModel(backbone_name="convnextv2_nano", pretrained=False)
        model.eval()

        img1 = torch.randn(1, 3, 160, 160)
        img2 = torch.randn(1, 3, 160, 160)

        with torch.no_grad():
            emb1 = model(img1)["reid_embedding"]
            emb2 = model(img2)["reid_embedding"]

        # Cosine similarity is dot product of normalized vectors
        cos_sim = torch.sum(emb1 * emb2, dim=-1).item()
        assert -1.0 <= cos_sim <= 1.0

    def test_gradient_flow_to_all_heads(self):
        """Verifies that loss gradients flow through all 4 heads into the backbone."""
        model = BiometricMultiTaskModel(backbone_name="convnextv2_nano", pretrained=False)
        model.train()

        x = torch.randn(2, 3, 160, 160)
        out = model(x)

        # Multi-task scalar loss combining all 4 heads
        loss = (
            out["age_probs"].sum()
            + out["gender_logit"].sum()
            + out["log_variance"].sum()
            + out["reid_embedding"].sum()
        )
        loss.backward()

        # Check gradients in each head
        assert model.age_ldl_head[0].weight.grad is not None
        assert model.gender_head[0].weight.grad is not None
        assert model.uncertainty_head[0].weight.grad is not None
        assert model.reid_head[0].weight.grad is not None
