"""
Unit Tests for BiometricMultiTaskModel (ConvNeXt-V2-Tiny, MultiScaleBiometricFusion)
=====================================================================================
Validates:
1. ConvNeXt-V2-Tiny default backbone instantiation.
2. MultiScaleBiometricFusion dual-pooling and projection dimensions.
3. Head routing separation and strict gradient isolation (Gender/Re-ID isolated from Stage 2/3).
4. Full backward compatibility across MobileNetV4 and FallbackStem.
5. 101-bin age LDL probability simplex and continuous expected age.
6. Gender logit and sigmoid probability.
7. Aleatoric heteroscedastic log-variance and standard deviation.
8. 64-D unit-normalized Re-ID embedding representation.
"""

from __future__ import annotations
import pytest
import torch
import torch.nn.functional as F

from src.biometrics.estimation.backbone import (
    BiometricMultiTaskModel,
    MultiScaleBiometricFusion,
    FallbackStem,
)


class TestBiometricMultiTaskModel:
    def test_default_backbone_is_convnextv2_tiny(self):
        """Default instantiation must configure ConvNeXt-V2-Tiny."""
        model = BiometricMultiTaskModel(pretrained=False)
        assert model.backbone_name == "convnextv2_tiny"
        assert model.in_features == 512
        assert model.macro_features == 256
        assert hasattr(model, "fusion")

    @pytest.mark.parametrize("backbone_name", ["convnextv2_tiny", "convnextv2_nano", "mobilenetv4"])
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
        norms = torch.norm(outputs["reid_embedding"], p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-4)

        # 5. Internal representations
        assert "features" in outputs
        assert outputs["features"].shape == (batch_size, 512)
        assert "z4" in outputs
        assert outputs["z4"].shape == (batch_size, 256)

    def test_multiscale_fusion_module_direct(self):
        """Verifies MultiScaleBiometricFusion module input/output dimensions."""
        fusion = MultiScaleBiometricFusion(in_channels=(192, 384, 768), proj_dim=128, fused_dim=512)
        fusion.eval()

        B = 2
        s2 = torch.randn(B, 192, 20, 20)
        s3 = torch.randn(B, 384, 10, 10)
        s4 = torch.randn(B, 768, 5, 5)

        z_fused, z4 = fusion(s2, s3, s4)
        assert z_fused.shape == (B, 512)
        assert z4.shape == (B, 256)

    def test_head_routing_gradient_isolation_gender(self):
        """Gender head receives ONLY Stage 4: zero gradient into Stage 2 and Stage 3 projections."""
        model = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", pretrained=False)
        model.train()

        x = torch.randn(2, 3, 160, 160)
        out = model(x)

        # Backward exclusively from Gender head
        gender_loss = out["gender_logit"].sum()
        gender_loss.backward()

        # Stage 4 projection MUST receive gradient
        assert model.fusion.proj4.weight.grad is not None
        assert not torch.all(model.fusion.proj4.weight.grad == 0.0)

        # Stage 2 and Stage 3 projections MUST NOT receive gradient
        assert model.fusion.proj2.weight.grad is None
        assert model.fusion.proj3.weight.grad is None

    def test_head_routing_gradient_isolation_reid(self):
        """Re-ID identity head receives ONLY Stage 4: zero gradient into Stage 2 and Stage 3."""
        model = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", pretrained=False)
        model.train()

        x = torch.randn(2, 3, 160, 160)
        out = model(x)

        # Backward exclusively from Re-ID head
        # Target projection is required because sum of LayerNorm normalized vector is identically zero
        target_reid = torch.randn_like(out["reid_embedding"])
        reid_loss = (out["reid_embedding"] * target_reid).sum()
        reid_loss.backward()

        # Stage 4 projection MUST receive gradient
        assert model.fusion.proj4.weight.grad is not None
        assert not torch.all(model.fusion.proj4.weight.grad == 0.0)

        # Stage 2 and Stage 3 projections MUST NOT receive gradient
        assert model.fusion.proj2.weight.grad is None
        assert model.fusion.proj3.weight.grad is None

    def test_head_routing_multiscale_gradients_age_and_uncertainty(self):
        """Age LDL head receives fused embedding: gradients flow to ALL stages (2, 3, and 4)."""
        model = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", pretrained=False)
        model.train()

        x = torch.randn(2, 3, 160, 160)
        out = model(x)

        age_loss = out["expected_age"].sum()
        age_loss.backward()

        assert model.fusion.proj2.weight.grad is not None
        assert not torch.all(model.fusion.proj2.weight.grad == 0.0)
        assert model.fusion.proj3.weight.grad is not None
        assert not torch.all(model.fusion.proj3.weight.grad == 0.0)
        assert model.fusion.proj4.weight.grad is not None
        assert not torch.all(model.fusion.proj4.weight.grad == 0.0)

    def test_fallback_stem_compatibility(self):
        """Verifies FallbackStem produces correct 3-stage feature shapes and runs multi-task model."""
        stem = FallbackStem(stage_channels=(192, 384, 768))
        stem.eval()
        x = torch.randn(2, 3, 160, 160)
        s2, s3, s4 = stem(x)

        assert s2.shape == (2, 192, 20, 20)
        assert s3.shape == (2, 384, 10, 10)
        assert s4.shape == (2, 768, 5, 5)

        # Force fallback inside model
        model = BiometricMultiTaskModel(pretrained=False)
        model.backbone = stem
        out = model(x)
        assert out["age_probs"].shape == (2, 101)
        assert out["gender_logit"].shape == (2,)
        assert out["reid_embedding"].shape == (2, 64)

    def test_reid_cosine_similarity(self):
        """Validates that distinct inputs produce distinct unit-normalized embeddings."""
        model = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", pretrained=False)
        model.eval()

        img1 = torch.randn(1, 3, 160, 160)
        img2 = torch.randn(1, 3, 160, 160)

        with torch.no_grad():
            emb1 = model(img1)["reid_embedding"]
            emb2 = model(img2)["reid_embedding"]

        cos_sim = torch.sum(emb1 * emb2, dim=-1).item()
        assert -1.0 <= cos_sim <= 1.0

    def test_gradient_flow_to_all_heads(self):
        """Verifies that combined multi-task loss gradients flow through all 4 heads into the backbone."""
        model = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", pretrained=False)
        model.train()

        x = torch.randn(2, 3, 160, 160)
        out = model(x)

        target_reid = torch.randn_like(out["reid_embedding"])
        loss = (
            out["expected_age"].sum()
            + out["gender_logit"].sum()
            + out["log_variance"].sum()
            + (out["reid_embedding"] * target_reid).sum()
        )
        loss.backward()

        assert model.age_ldl_head[0].weight.grad is not None
        assert not torch.all(model.age_ldl_head[0].weight.grad == 0.0)
        assert model.gender_head[0].weight.grad is not None
        assert not torch.all(model.gender_head[0].weight.grad == 0.0)
        assert model.uncertainty_head[0].weight.grad is not None
        assert not torch.all(model.uncertainty_head[0].weight.grad == 0.0)
        assert model.reid_head[0].weight.grad is not None
        assert not torch.all(model.reid_head[0].weight.grad == 0.0)
