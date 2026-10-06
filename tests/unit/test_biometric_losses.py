"""
Unit Tests for Biometric Loss Functions
=======================================
Tests Cramér Distance (squared EMD), Soft-KL Divergence, Soft-L1 MAE,
Masked Dimorphic BCE, Aleatoric Heteroscedastic Uncertainty, and UnifiedBiometricLoss.
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
        # If model is overconfident (log_var = 0 -> sigma = 1)
        loss_overconfident = aleatoric_heteroscedastic_loss(pred_age, target_age, torch.tensor([0.0]))
        # If model knows it's uncertain (log_var = 4 -> sigma ~ 7.4)
        loss_honest = aleatoric_heteroscedastic_loss(pred_age, target_age, torch.tensor([4.0]))

        assert loss_overconfident.item() > loss_honest.item()


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
