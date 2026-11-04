"""
Unit Tests for Biometric Loss Functions
=======================================
Tests Cramér Distance (squared EMD), Soft-KL Divergence (FP16 Protected),
Soft-L1 MAE, Masked Dimorphic BCE (Sample Normalized), Aleatoric Heteroscedastic Uncertainty (L1-Detached Laplace),
and UnifiedBiometricLoss.
"""

from __future__ import annotations
import pytest
import torch
import torch.nn.functional as F

from src.biometrics.estimation.losses import (
    cramer_distance_loss,
    soft_kl_divergence_loss,
    soft_l1_mae_loss,
    masked_dimorphic_bce_loss,
    aleatoric_heteroscedastic_loss,
    UnifiedBiometricLoss,
)


class TestCramerDistanceLoss:
    def test_zero_cramer_distance_for_identical_distributions(self):
        p = torch.softmax(torch.randn(4, 101), dim=-1)
        loss = cramer_distance_loss(p, p)
        assert pytest.approx(loss.item(), abs=1e-6) == 0.0

    def test_cramer_distance_monotonicity(self):
        """Mass located further from target distribution should incur strictly higher Cramér loss."""
        target = torch.zeros(1, 101)
        target[0, 50] = 1.0  # Spike at 50

        # Close distribution: spike at 51
        pred_close = torch.zeros(1, 101)
        pred_close[0, 51] = 1.0

        # Far distribution: spike at 70
        pred_far = torch.zeros(1, 101)
        pred_far[0, 70] = 1.0

        loss_close = cramer_distance_loss(pred_close, target)
        loss_far = cramer_distance_loss(pred_far, target)

        assert loss_far.item() > loss_close.item()

    def test_cramer_distance_gradient(self):
        logits = torch.randn(4, 101, requires_grad=True)
        probs = torch.softmax(logits, dim=-1)
        target = torch.softmax(torch.randn(4, 101), dim=-1)

        loss = cramer_distance_loss(probs, target)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert not torch.all(logits.grad == 0.0)


class TestSoftKLDivergenceLoss:
    def test_kl_divergence_non_negative(self):
        logits = torch.randn(4, 101)
        probs = torch.softmax(logits, dim=-1)
        target = torch.softmax(torch.randn(4, 101), dim=-1)

        loss = soft_kl_divergence_loss(probs, target)
        assert loss.item() >= -1e-6

    def test_kl_divergence_zero_identical(self):
        probs = torch.softmax(torch.randn(4, 101), dim=-1)
        loss = soft_kl_divergence_loss(probs, probs)
        assert pytest.approx(loss.item(), abs=1e-5) == 0.0

    def test_soft_kl_fp16_underflow_protection(self):
        """Simulate near-zero probability simplex in float16; should not produce NaN."""
        logits = torch.randn(2, 101)
        probs = torch.softmax(logits, dim=-1).half()
        # Force subnormals
        probs[:, 50:] = 1e-7

        targets = torch.softmax(torch.randn(2, 101), dim=-1).half()

        # 1. With logits passed
        loss_with_logits = soft_kl_divergence_loss(probs, targets, pred_logits=logits.half())
        assert not torch.isnan(loss_with_logits)
        assert not torch.isinf(loss_with_logits)

        # 2. With only probs passed (float32 cast protection)
        loss_with_probs = soft_kl_divergence_loss(probs, targets)
        assert not torch.isnan(loss_with_probs)
        assert not torch.isinf(loss_with_probs)


class TestSoftL1MAELoss:
    def test_mae_calculation(self):
        pred_age = torch.tensor([25.0, 40.0, 60.0])
        target_age = torch.tensor([27.0, 35.0, 60.0])
        loss = soft_l1_mae_loss(pred_age, target_age)
        expected = (2.0 + 5.0 + 0.0) / 3.0
        assert pytest.approx(loss.item(), rel=1e-4) == expected


class TestMaskedDimorphicBCELoss:
    def test_zero_loss_when_all_masked_out(self):
        pred_logits = torch.tensor([2.0, -1.0, 0.5])
        targets = torch.tensor([1.0, 0.0, 1.0])
        mask = torch.tensor([0.0, 0.0, 0.0])  # All unlabelled
        dimorphism = torch.tensor([1.0, 1.0, 1.0])

        loss = masked_dimorphic_bce_loss(pred_logits, targets, mask, dimorphism)
        assert loss.item() == 0.0
        assert not torch.isnan(loss)

    def test_gradient_flows_only_to_unmasked(self):
        pred_logits = torch.tensor([2.0, -1.0, 0.5], requires_grad=True)
        targets = torch.tensor([1.0, 0.0, 1.0])
        mask = torch.tensor([1.0, 0.0, 1.0])  # Sample 1 masked out
        dimorphism = torch.tensor([1.0, 0.5, 0.8])

        loss = masked_dimorphic_bce_loss(pred_logits, targets, mask, dimorphism)
        loss.backward()

        assert pred_logits.grad is not None
        assert pred_logits.grad[1].item() == 0.0  # Index 1 was masked out
        assert pred_logits.grad[0].item() != 0.0
        assert pred_logits.grad[2].item() != 0.0

    def test_denominator_normalizes_strictly_by_valid_count(self):
        """Denominator must be valid_count (sum of mask), NOT sum of dimorphism weights."""
        pred_logits = torch.tensor([0.0, 0.0])  # BCE = log(2) ~ 0.693147
        targets = torch.tensor([1.0, 1.0])
        mask = torch.tensor([1.0, 1.0])         # valid_count = 2.0
        dimorphism = torch.tensor([0.2, 0.8])   # sum of dim = 1.0

        bce_val = F.binary_cross_entropy_with_logits(torch.tensor(0.0), torch.tensor(1.0)).item()
        # Numerator = bce * 0.2 + bce * 0.8 = bce * 1.0
        # Correct denominator = valid_count = 2.0
        # Expected loss = (bce * 1.0) / 2.0 = 0.5 * bce
        loss = masked_dimorphic_bce_loss(pred_logits, targets, mask, dimorphism)
        expected = (bce_val * 1.0) / 2.0
        assert pytest.approx(loss.item(), rel=1e-4) == expected

    def test_shape_mismatch_broadcasting_protection(self):
        """Ensures (B, 1) logits against (B,) targets does not create (B, B) matrix broadcasting."""
        pred_logits = torch.tensor([[2.0], [-1.0], [0.5]])  # Shape (3, 1)
        targets = torch.tensor([1.0, 0.0, 1.0])              # Shape (3,)
        mask = torch.tensor([[1.0], [1.0], [1.0]])           # Shape (3, 1)
        dimorphism = torch.tensor([1.0, 0.8, 0.9])           # Shape (3,)

        loss = masked_dimorphic_bce_loss(pred_logits, targets, mask, dimorphism)
        assert loss.ndim == 0
        assert not torch.isnan(loss)


class TestAleatoricUncertaintyLoss:
    def test_loss_penalizes_high_residual_with_low_variance(self):
        pred_age = torch.tensor([25.0])
        target_age = torch.tensor([45.0])  # error = 20
        # If model is overconfident (log_var = 0 -> s = 0)
        loss_overconfident = aleatoric_heteroscedastic_loss(pred_age, target_age, torch.tensor([0.0]))
        # If model knows it's uncertain (log_var = 4 -> s = 4)
        loss_honest = aleatoric_heteroscedastic_loss(pred_age, target_age, torch.tensor([4.0]))

        assert loss_overconfident.item() > loss_honest.item()

    def test_pred_mu_detached_zero_gradient_to_age(self):
        """Continuous expected age must be detached so uncertainty loss never biases age."""
        pred_age = torch.tensor([25.0, 40.0], requires_grad=True)
        target_age = torch.tensor([30.0, 42.0])
        log_var = torch.tensor([1.0, 1.5], requires_grad=True)

        loss = aleatoric_heteroscedastic_loss(pred_age, target_age, log_var)
        loss.backward()

        # pred_age must receive EXACTLY zero gradient
        assert pred_age.grad is None or torch.all(pred_age.grad == 0.0)
        # log_var MUST receive valid non-zero gradient
        assert log_var.grad is not None
        assert not torch.all(log_var.grad == 0.0)


class TestUnifiedBiometricLoss:
    def test_forward_and_backward(self):
        criterion = UnifiedBiometricLoss()
        B = 4
        # Predictions
        age_logits = torch.randn(B, 101, requires_grad=True)
        age_probs = torch.softmax(age_logits, dim=-1)
        bins = torch.arange(101, dtype=torch.float32)
        expected_age = torch.sum(age_probs * bins, dim=-1)

        gender_logits = torch.randn(B, requires_grad=True)
        log_var = torch.randn(B, requires_grad=True)

        preds = {
            "age_probs": age_probs,
            "age_logits": age_logits,
            "expected_age": expected_age,
            "gender_logit": gender_logits,
            "log_variance": log_var
        }

        targets = {
            "ldl_101": torch.softmax(torch.randn(B, 101), dim=-1),
            "age_continuous": torch.tensor([25.0, 30.0, 45.0, 60.0]),
            "gender": torch.tensor([1.0, 0.0, 1.0, 0.0]),
            "gender_mask": torch.tensor([1.0, 0.0, 1.0, 0.0]),
            "dimorphism_weight": torch.tensor([0.8, 0.5, 1.0, 0.9])
        }

        total_loss, metrics = criterion(preds, targets)
        assert total_loss.ndim == 0
        assert not torch.isnan(total_loss)
        assert "loss_cramer" in metrics
        assert "loss_kl" in metrics
        assert "loss_mae" in metrics
        assert "loss_gender" in metrics
        assert "loss_aleatoric" in metrics

        total_loss.backward()
        assert age_logits.grad is not None
        assert gender_logits.grad is not None
        assert log_var.grad is not None

    def test_edge_case_zero_gender_mask_no_nan(self):
        """When entire batch has no gender labels, total loss should remain smooth without NaN."""
        criterion = UnifiedBiometricLoss()
        B = 2
        age_logits = torch.randn(B, 101, requires_grad=True)
        age_probs = torch.softmax(age_logits, dim=-1)
        expected_age = torch.sum(age_probs * torch.arange(101, dtype=torch.float32), dim=-1)
        gender_logits = torch.randn(B, requires_grad=True)
        log_var = torch.randn(B, requires_grad=True)

        preds = {
            "age_probs": age_probs,
            "age_logits": age_logits,
            "expected_age": expected_age,
            "gender_logit": gender_logits,
            "log_variance": log_var
        }

        targets = {
            "ldl_101": torch.softmax(torch.randn(B, 101), dim=-1),
            "age_continuous": torch.tensor([25.0, 30.0]),
            "gender": torch.tensor([0.0, 0.0]),
            "gender_mask": torch.tensor([0.0, 0.0]),  # All unlabelled!
            "dimorphism_weight": torch.tensor([1.0, 1.0])
        }

        total_loss, metrics = criterion(preds, targets)
        assert not torch.isnan(total_loss)
        assert metrics["loss_gender"] == 0.0
        total_loss.backward()
        assert not torch.isnan(gender_logits.grad).any()

    def test_amp_fp16_loss_stability(self):
        """Unified loss with FP16 inputs does not NaN or overflow."""
        criterion = UnifiedBiometricLoss()
        B = 3
        age_logits = torch.randn(B, 101).requires_grad_()
        age_probs = torch.softmax(age_logits, dim=-1).half()
        expected_age = torch.sum(age_probs.float() * torch.arange(101, dtype=torch.float32), dim=-1).half()
        gender_logits = torch.randn(B).half().requires_grad_()
        log_var = torch.randn(B).half().requires_grad_()

        preds = {
            "age_probs": age_probs,
            "age_logits": age_logits.half(),
            "expected_age": expected_age,
            "gender_logit": gender_logits,
            "log_variance": log_var
        }

        targets = {
            "ldl_101": torch.softmax(torch.randn(B, 101), dim=-1).half(),
            "age_continuous": torch.tensor([15.0, 45.0, 70.0]).half(),
            "gender": torch.tensor([1.0, 0.0, 1.0]).half(),
            "gender_mask": torch.tensor([1.0, 1.0, 0.0]).half(),
            "dimorphism_weight": torch.tensor([0.3, 0.9, 0.5]).half(),
        }

        total_loss, metrics = criterion(preds, targets)
        assert not torch.isnan(total_loss)
        assert not torch.isinf(total_loss)

        total_loss.backward()
        assert age_logits.grad is not None
        assert not torch.isnan(age_logits.grad).any()
        assert gender_logits.grad is not None
        assert not torch.isnan(gender_logits.grad).any()

    def test_fine_interval_ldl_and_tikhonov_smoothness_and_photometric_consistency(self):
        """Verifies fine_interval_ldl_loss, ldl_tikhonov_smoothness_loss, and photometric_consistency_loss."""
        from src.biometrics.estimation.losses import (
            fine_interval_ldl_loss,
            ldl_tikhonov_smoothness_loss,
            photometric_consistency_loss,
        )

        # 1. 23-bracket fine interval ordinal loss monotonicity
        target_23 = torch.zeros(1, 23)
        target_23[0, 8] = 1.0  # bracket '26-30'
        pred_close_23 = torch.zeros(1, 23, requires_grad=True)
        with torch.no_grad():
            pred_close_23[0, 9] = 1.0
        pred_far_23 = torch.zeros(1, 23, requires_grad=True)
        with torch.no_grad():
            pred_far_23[0, 20] = 1.0

        l_close = fine_interval_ldl_loss(pred_close_23, target_23)
        l_far = fine_interval_ldl_loss(pred_far_23, target_23)
        assert l_far.item() > l_close.item()
        l_close.backward()
        assert pred_close_23.grad is not None

        # 2. Tikhonov second-order Laplacian smoothness penalty
        bins = torch.arange(101, dtype=torch.float32)
        smooth_p = torch.softmax(-0.5 * ((bins - 30.0) / 3.0) ** 2, dim=-1).unsqueeze(0)
        jagged_logits = torch.tensor([3.0 if i % 2 == 0 else -3.0 for i in range(101)]).unsqueeze(0)
        jagged_p = torch.softmax(jagged_logits, dim=-1)

        loss_smooth = ldl_tikhonov_smoothness_loss(smooth_p)
        loss_jagged = ldl_tikhonov_smoothness_loss(jagged_p)
        assert loss_jagged.item() > 50.0 * loss_smooth.item()

        # 3. Photometric dual-view consistency loss
        B = 2
        clean_preds = {
            "age_probs": torch.softmax(torch.randn(B, 101), dim=-1),
            "gender_logit": torch.tensor([2.0, -2.0]),
            "reid_embedding": F.normalize(torch.randn(B, 64), p=2, dim=-1),
        }
        # Identical corrupted predictions -> ~0 loss
        zero_cons = photometric_consistency_loss(clean_preds, clean_preds)
        assert pytest.approx(zero_cons.item(), abs=1e-5) == 0.0

        corrupt_logits = torch.randn(B, 101, requires_grad=True)
        corrupt_preds = {
            "age_probs": torch.softmax(corrupt_logits, dim=-1),
            "gender_logit": torch.tensor([-1.5, 1.5], requires_grad=True),
            "reid_embedding": F.normalize(torch.randn(B, 64, requires_grad=True), p=2, dim=-1),
        }
        pos_cons = photometric_consistency_loss(clean_preds, corrupt_preds)
        assert pos_cons.item() > 0.05
        pos_cons.backward()
        assert corrupt_logits.grad is not None
        assert not torch.isnan(corrupt_logits.grad).any()

    def test_photometric_consistency_zero_bin_no_nan_and_clean_detached(self):
        """Verifies exact zero bins in p_corr do not produce NaN gradients and clean teacher is detached."""
        from src.biometrics.estimation.losses import photometric_consistency_loss

        p_clean = torch.softmax(torch.randn(2, 101), dim=-1).requires_grad_(True)
        p_corr = torch.zeros(2, 101)
        p_corr[:, 30] = 1.0  # 100 exact zero bins per sample!
        p_corr.requires_grad_(True)

        loss = photometric_consistency_loss({"age_probs": p_clean}, {"age_probs": p_corr})
        loss.backward()

        assert p_corr.grad is not None
        assert not torch.isnan(p_corr.grad).any()
        assert torch.all(torch.isfinite(p_corr.grad))
        # Clean teacher must NOT receive backward gradients
        assert p_clean.grad is None or torch.all(p_clean.grad == 0.0)

    def test_unified_loss_supervises_fine_interval_logits_directly(self):
        """Verifies UnifiedBiometricLoss supervises fine_interval_logits alongside age_interval_probs."""
        criterion = UnifiedBiometricLoss(weight_fine_23=0.5)
        B = 2
        age_logits = torch.randn(B, 101, requires_grad=True)
        age_probs = torch.softmax(age_logits, dim=-1)
        expected_age = torch.sum(age_probs * torch.arange(101, dtype=torch.float32), dim=-1)
        fine_interval_logits = torch.randn(B, 23, requires_grad=True)
        gender_logits = torch.randn(B, requires_grad=True)
        log_var = torch.randn(B, requires_grad=True)

        preds = {
            "age_probs": age_probs,
            "age_logits": age_logits,
            "expected_age": expected_age,
            "fine_interval_logits": fine_interval_logits,
            "gender_logit": gender_logits,
            "log_variance": log_var,
        }
        targets = {
            "ldl_101": torch.softmax(torch.randn(B, 101), dim=-1),
            "ldl_fine_23": torch.softmax(torch.randn(B, 23), dim=-1),
            "age_continuous": torch.tensor([25.0, 60.0]),
            "gender": torch.tensor([1.0, 0.0]),
            "gender_mask": torch.tensor([1.0, 1.0]),
            "dimorphism_weight": torch.tensor([1.0, 1.0]),
        }

        total_loss, metrics = criterion(preds, targets)
        assert metrics["loss_fine_23"] > 0.0
        total_loss.backward()
        assert fine_interval_logits.grad is not None
        assert not torch.all(fine_interval_logits.grad == 0.0)


