"""
Unit Tests for SOTA Biometric Upgrades
======================================
Tests:
1. CoordinateAttention: 1D direction-aware pooling along X and Y axes (Hou et al. CVPR 2021).
2. Boundary-Calibrated Folded Gaussian LDL (BC-LDL): Eliminates infant boundary distortion,
   ensuring continuous expectation for age 1.0 is exactly 1.0000 +/- 0.0001.
3. SupervisedContrastiveLoss: Trains 64-D Re-ID embedding head with tau=0.07, lambda_reid=0.15.
4. CoupledHeteroscedasticLaplaceLoss: Huber bounds with smooth gradient downscaling for corrupt/occluded faces.
5. UnifiedBiometricLoss with Re-ID SupCon integration: Verified gradient flow into reid_head.
6. ByteTrack Gated Matching: Prevents identity swap when IoU >= 0.3 but appearance distance > 0.35.
"""

from __future__ import annotations
import math
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.biometrics.estimation.backbone import (
    CoordinateAttention,
    MultiScaleBiometricFusion,
    BiometricMultiTaskModel,
)
from src.biometrics.estimation.ldl_engine import (
    compute_gaussian_ldl,
    compute_boundary_calibrated_ldl,
    ContinuousLDLEngine,
)
from src.biometrics.estimation.losses import (
    SupervisedContrastiveLoss,
    CoupledHeteroscedasticLaplaceLoss,
    UnifiedBiometricLoss,
)
from src.biometrics.tracking.byte_tracker import (
    ByteTracker,
    Detection,
    STrack,
)


class TestCoordinateAttention:
    """Verifies Coordinate Attention module behavior and direction awareness."""

    def test_coordinate_attention_output_shape(self):
        B, C, H, W = 4, 192, 20, 20
        att = CoordinateAttention(in_channels=C, reduction=16)
        x = torch.randn(B, C, H, W)
        out = att(x)
        assert out.shape == (B, C, H, W)

    def test_coordinate_attention_direction_awareness(self):
        """Attention map should respond differentially to horizontal vs vertical patterns."""
        att = CoordinateAttention(in_channels=32, reduction=8)
        att.eval()

        # Horizontal stripe pattern (e.g., forehead wrinkles)
        x_horiz = torch.zeros(1, 32, 20, 20)
        x_horiz[:, :, 10, :] = 5.0

        # Vertical stripe pattern
        x_vert = torch.zeros(1, 32, 20, 20)
        x_vert[:, :, :, 10] = 5.0

        with torch.no_grad():
            out_h = att(x_horiz)
            out_v = att(x_vert)

        assert not torch.allclose(out_h, out_v)

    def test_coordinate_attention_gradient_flow(self):
        att = CoordinateAttention(in_channels=64, reduction=16)
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        out = att(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert att.conv_h.weight.grad is not None
        assert att.conv_w.weight.grad is not None
        assert not torch.all(att.conv_h.weight.grad == 0.0)
        assert not torch.all(att.conv_w.weight.grad == 0.0)


class TestBoundaryCalibratedLDL:
    """Verifies BC-LDL boundary calibration and exact expectation recovery."""

    def test_infant_boundary_exact_expectation(self):
        """Continuous expectation for age 1.0 must be exactly 1.0000 +/- 0.0001."""
        engine = ContinuousLDLEngine(num_bins=101)
        dist = compute_gaussian_ldl(mean_age=1.0, sigma=2.5, num_bins=101)
        exp_age = engine.expected_age(dist)
        assert abs(exp_age - 1.0) <= 0.0001, f"Expected 1.0000 +/- 0.0001, got {exp_age:.6f}"

    def test_newborn_boundary_expectation(self):
        """Continuous expectation for newborn age 0.0 must be 0.0000 +/- 0.0001."""
        engine = ContinuousLDLEngine(num_bins=101)
        dist = compute_gaussian_ldl(mean_age=0.0, sigma=2.5, num_bins=101)
        exp_age = engine.expected_age(dist)
        assert abs(exp_age - 0.0) <= 0.0001, f"Expected 0.0000 +/- 0.0001, got {exp_age:.6f}"

    def test_geriatric_boundary_exact_expectation(self):
        """Continuous expectation for geriatric boundary age 99.0 must be 99.0000 +/- 0.0001."""
        engine = ContinuousLDLEngine(num_bins=101)
        dist = compute_gaussian_ldl(mean_age=99.0, sigma=2.5, num_bins=101)
        exp_age = engine.expected_age(dist)
        assert abs(exp_age - 99.0) <= 0.0001, f"Expected 99.0000 +/- 0.0001, got {exp_age:.6f}"

    @pytest.mark.parametrize("age", [0.0, 0.5, 1.0, 2.0, 3.5, 96.0, 98.0, 99.5, 100.0])
    def test_boundary_simplex_conservation(self, age):
        """All distributions must strictly conserve probability simplex sum(p) == 1.0."""
        dist = compute_boundary_calibrated_ldl(mean_age=age, sigma=2.5, num_bins=101)
        assert np.isclose(np.sum(dist), 1.0, atol=1e-5)
        assert np.all(dist >= 0.0)

    def test_interior_age_unaffected(self):
        """Interior ages (e.g. 30.0) remain accurate and match expectation."""
        engine = ContinuousLDLEngine(num_bins=101)
        dist = compute_gaussian_ldl(mean_age=30.0, sigma=2.5, num_bins=101)
        exp_age = engine.expected_age(dist)
        assert abs(exp_age - 30.0) <= 0.05


class TestSupervisedContrastiveLoss:
    """Verifies Supervised Contrastive Loss for 64-D Re-ID head."""

    def test_supcon_identical_labels_cluster(self):
        """Same-class embeddings should produce significantly lower loss than mixed classes."""
        supcon = SupervisedContrastiveLoss(temperature=0.07)

        # 4 samples: 2 of class 0, 2 of class 1
        labels = torch.tensor([0, 0, 1, 1])

        # Clustered embeddings: identical within class, orthogonal across classes
        emb_clustered = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=torch.float32)

        # Unclustered embeddings: opposite
        emb_unclustered = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=torch.float32)

        loss_clustered = supcon(emb_clustered, labels)
        loss_unclustered = supcon(emb_unclustered, labels)

        assert loss_clustered.item() < loss_unclustered.item()

    def test_supcon_gradient_flow(self):
        supcon = SupervisedContrastiveLoss(temperature=0.07)
        emb = torch.randn(4, 64, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])

        loss = supcon(emb, labels)
        loss.backward()

        assert emb.grad is not None
        assert not torch.isnan(emb.grad).any()
        assert not torch.all(emb.grad == 0.0)

    def test_supcon_no_positives_returns_zero_safely(self):
        """When all labels are distinct (no positive pairs), loss returns 0 without crashing."""
        supcon = SupervisedContrastiveLoss(temperature=0.07)
        emb = torch.randn(4, 64, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3])  # All distinct

        loss = supcon(emb, labels)
        assert pytest.approx(loss.item(), abs=1e-6) == 0.0


class TestCoupledHeteroscedasticLaplaceLoss:
    """Verifies Coupled Heteroscedastic Laplace Loss with Huber bounds."""

    def test_coupled_laplace_smooth_huber_region(self):
        loss_fn = CoupledHeteroscedasticLaplaceLoss(delta=1.0)
        pred_mu = torch.tensor([25.0], requires_grad=True)
        target = torch.tensor([25.5])  # small residual = 0.5 <= delta=1.0
        log_var = torch.tensor([0.0], requires_grad=True)

        loss = loss_fn(pred_mu, target, log_var)
        loss.backward()

        assert pred_mu.grad is not None
        assert log_var.grad is not None
        assert not torch.isnan(pred_mu.grad).any()

    def test_corrupt_face_uncertainty_downscales_gradient(self):
        """When uncertainty s is high (corrupt face), exp(-s) smoothly attenuates gradient."""
        loss_fn = CoupledHeteroscedasticLaplaceLoss(delta=1.0)

        # High confidence (log_var = 0.0)
        mu_conf = torch.tensor([20.0], requires_grad=True)
        loss_conf = loss_fn(mu_conf, torch.tensor([50.0]), torch.tensor([0.0]))
        loss_conf.backward()
        grad_conf = abs(mu_conf.grad.item())

        # High uncertainty (log_var = 4.0) -> corrupt/occluded face
        mu_unc = torch.tensor([20.0], requires_grad=True)
        loss_unc = loss_fn(mu_unc, torch.tensor([50.0]), torch.tensor([4.0]))
        loss_unc.backward()
        grad_unc = abs(mu_unc.grad.item())

        assert grad_unc < grad_conf * 0.1, "Uncertainty did not smoothly scale down outlier gradient"


class TestUnifiedBiometricLossWithReID:
    """Verifies that UnifiedBiometricLoss trains the Re-ID head."""

    def test_reid_head_receives_gradient(self):
        criterion = UnifiedBiometricLoss(weight_reid=0.15, reid_temperature=0.07)
        B = 4
        age_logits = torch.randn(B, 101, requires_grad=True)
        age_probs = torch.softmax(age_logits, dim=-1)
        expected_age = torch.sum(age_probs * torch.arange(101, dtype=torch.float32), dim=-1)
        gender_logits = torch.randn(B, requires_grad=True)
        log_var = torch.randn(B, requires_grad=True)
        reid_emb = torch.randn(B, 64, requires_grad=True)

        preds = {
            "age_probs": age_probs,
            "age_logits": age_logits,
            "expected_age": expected_age,
            "gender_logit": gender_logits,
            "log_variance": log_var,
            "reid_embedding": reid_emb,
        }

        targets = {
            "ldl_101": torch.softmax(torch.randn(B, 101), dim=-1),
            "age_continuous": torch.tensor([20.0, 20.0, 50.0, 50.0]),
            "gender": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "identity": torch.tensor([0, 0, 1, 1]),
        }

        total_loss, metrics = criterion(preds, targets)
        assert "loss_reid" in metrics
        assert metrics["loss_reid"] > 0.0

        total_loss.backward()
        assert reid_emb.grad is not None
        assert not torch.all(reid_emb.grad == 0.0)


class TestByteTrackerGatedMatching:
    """Verifies strict appearance gating prevents identity swap."""

    def test_gated_matching_rejects_appearance_mismatch_even_with_high_iou(self):
        """
        Two subjects: Track A (Person A) becomes lost.
        A high-confidence detection appears with HIGH IoU (cost_iou <= 0.7),
        BUT completely different appearance (cost_reid > 0.35).
        Gated matching MUST REJECT the match and not swap IDs!
        """
        tracker = ByteTracker(track_thresh=0.6, low_thresh=0.2, reid_thresh=0.35, max_time_lost=20)

        feat_a = np.array([1.0] + [0.0] * 63, dtype=np.float32)
        feat_b = np.array([0.0, 1.0] + [0.0] * 62, dtype=np.float32)  # Orthogonal to A, distance = 1.0

        # Frame 1: Person A tracked at (100, 100, 150, 150)
        det_a = Detection(bbox=(100.0, 100.0, 150.0, 150.0), score=0.9, reid_feat=feat_a)
        tracks1 = tracker.update([det_a])
        assert len(tracks1) == 1
        tid_a = tracks1[0].track_id

        # Frame 2: Person A occluded (no detection) -> track becomes lost
        tracks2 = tracker.update([])
        assert len(tracks2) == 0
        assert len(tracker.lost_tracks) == 1

        # Frame 3: Detection with overlapping bounding box (IoU > 0.5) but Person B appearance!
        det_b = Detection(bbox=(105.0, 105.0, 155.0, 155.0), score=0.9, reid_feat=feat_b)
        tracks3 = tracker.update([det_b])

        # Person B MUST NOT steal Person A's track_id! It must be assigned a new ID!
        assert len(tracks3) == 1
        assert tracks3[0].track_id != tid_a, "Fatal ID Swap: Person B stole Person A track ID despite appearance mismatch!"


class TestSOTAEdgeCasesAndExport:
    """Verifies edge case robustness and hardware deployment export."""

    def test_supcon_fp16_amp_execution(self):
        """FP16 embeddings under AMP should compute safely without CPU Half addmm crash."""
        supcon = SupervisedContrastiveLoss(temperature=0.07)
        emb_fp16 = torch.randn(8, 64, dtype=torch.float16, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        loss = supcon(emb_fp16, labels)
        loss.backward()

        assert not torch.isnan(loss)
        assert emb_fp16.grad is not None
        assert not torch.isnan(emb_fp16.grad).any()

    def test_supcon_negative_labels_unlabeled_isolation(self):
        """Samples with label -1 (unlabeled) must NEVER pair with each other."""
        supcon = SupervisedContrastiveLoss(temperature=0.07)
        emb = torch.randn(4, 64, requires_grad=True)
        # 2 unlabeled samples (-1, -1) and 2 distinct IDs (0, 1) -> NO valid positive pairs!
        labels = torch.tensor([-1, -1, 0, 1])

        loss = supcon(emb, labels)
        assert pytest.approx(loss.item(), abs=1e-6) == 0.0

    def test_boundary_transition_continuity_no_jumps(self):
        """No artificial step discontinuities across boundary transitions (e.g. age 4.9 -> 5.0)."""
        engine = ContinuousLDLEngine(num_bins=101)
        for a in [4.0, 4.5, 4.9, 5.0, 5.1, 5.5, 6.0, 94.0, 94.9, 95.0, 95.1, 96.0]:
            dist = compute_gaussian_ldl(mean_age=a, sigma=2.5, num_bins=101)
            exp_a = engine.expected_age(dist)
            assert abs(exp_a - a) <= 0.0002, f"Discontinuity at age {a}: expected {a}, got {exp_a:.6f}"

    def test_batched_alpha_sample_weight_scaling(self):
        """Batched alpha_sample_weight (B,) scales total loss by batch mean rather than falling back to 1.0."""
        criterion = UnifiedBiometricLoss(weight_reid=0.15)
        B = 4
        age_logits = torch.randn(B, 101)
        age_probs = torch.softmax(age_logits, dim=-1)
        expected_age = torch.sum(age_probs * torch.arange(101, dtype=torch.float32), dim=-1)

        preds = {
            "age_probs": age_probs,
            "age_logits": age_logits,
            "expected_age": expected_age,
            "gender_logit": torch.randn(B),
            "log_variance": torch.randn(B),
            "reid_embedding": torch.randn(B, 64),
        }
        targets = {
            "ldl_101": torch.softmax(torch.randn(B, 101), dim=-1),
            "age_continuous": torch.tensor([20.0, 30.0, 40.0, 50.0]),
            "gender": torch.tensor([1.0, 0.0, 1.0, 0.0]),
            "identity": torch.tensor([0, 0, 1, 1]),
            "alpha_sample_weight": torch.tensor([2.0, 2.0, 2.0, 2.0]),  # 2.0x scale
        }

        total_loss_2x, _ = criterion(preds, targets)

        # Baseline with 1.0x weights
        targets_1x = dict(targets)
        targets_1x["alpha_sample_weight"] = torch.tensor([1.0, 1.0, 1.0, 1.0])
        total_loss_1x, _ = criterion(preds, targets_1x)

        assert pytest.approx(total_loss_2x.item(), rel=1e-3) == total_loss_1x.item() * 2.0

    def test_unified_loss_with_fp16_reid_embedding(self):
        """UnifiedBiometricLoss gracefully handles FP16 reid_embedding without throwing."""
        criterion = UnifiedBiometricLoss(weight_reid=0.15)
        B = 4
        preds = {
            "age_probs": torch.softmax(torch.randn(B, 101), dim=-1),
            "age_logits": torch.randn(B, 101),
            "expected_age": torch.tensor([20.0, 25.0, 30.0, 35.0]),
            "gender_logit": torch.randn(B),
            "log_variance": torch.randn(B),
            "reid_embedding": torch.randn(B, 64, dtype=torch.float16),  # Half precision
        }
        targets = {
            "ldl_101": torch.softmax(torch.randn(B, 101), dim=-1),
            "age_continuous": torch.tensor([20.0, 25.0, 30.0, 35.0]),
            "gender": torch.tensor([1.0, 0.0, 1.0, 0.0]),
            "identity": torch.tensor([0, 0, 1, 1]),
        }
        total_loss, metrics = criterion(preds, targets)
        assert not torch.isnan(total_loss)
        assert metrics["loss_reid"] >= 0.0

    def test_torchscript_and_onnx_export_compatibility(self):
        """BiometricMultiTaskModel must trace cleanly into TorchScript and export to ONNX."""
        import io
        model = BiometricMultiTaskModel(backbone_name="convnextv2_nano", pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 160, 160)

        # TorchScript tracing
        traced = torch.jit.trace(model, x)
        outputs = traced(x)
        assert len(outputs) == 4
        assert outputs[0].shape == (1, 101)  # age_logits
        assert outputs[1].shape == (1,)      # gender_logit
        assert outputs[2].shape == (1,)      # log_var
        assert outputs[3].shape == (1, 64)   # reid_emb

        # ONNX export
        buf = io.BytesIO()
        torch.onnx.export(
            model, x, buf,
            input_names=["input"],
            output_names=["age_logits", "gender_logit", "log_variance", "reid_embedding"],
            dynamic_axes={"input": {0: "batch"}},
            opset_version=14
        )
        assert len(buf.getvalue()) > 1000

    def test_reid_loss_does_not_cluster_demographic_cohorts_without_identity(self):
        """When identity is absent or -1, UnifiedBiometricLoss must NOT fall back to age/gender cohort pseudo-labels."""
        criterion = UnifiedBiometricLoss(weight_reid=0.15)
        B = 4
        reid_emb = torch.randn(B, 64, requires_grad=True)
        preds = {
            "age_probs": torch.softmax(torch.randn(B, 101), dim=-1),
            "age_logits": torch.randn(B, 101),
            "expected_age": torch.tensor([22.0, 22.0, 22.0, 22.0]),
            "gender_logit": torch.randn(B),
            "log_variance": torch.randn(B),
            "reid_embedding": reid_emb,
        }
        targets = {
            "ldl_101": torch.softmax(torch.randn(B, 101), dim=-1),
            "age_continuous": torch.tensor([22.0, 22.0, 22.0, 22.0]),
            "gender": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        }
        total_loss, metrics = criterion(preds, targets)
        assert metrics["loss_reid"] == 0.0
        total_loss.backward()
        assert reid_emb.grad is None or torch.all(reid_emb.grad == 0.0)

    def test_age_mask_zeros_invalid_age_gradients(self):
        """Samples with age_mask == 0.0 must receive zero gradient from all age loss heads."""
        criterion = UnifiedBiometricLoss(weight_interval=0.25)
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
            "log_variance": log_var,
        }
        targets = {
            "ldl_101": torch.softmax(torch.randn(B, 101), dim=-1),
            "age_continuous": torch.tensor([30.0, 25.0]),
            "age_min": torch.tensor([28.0, 25.0]),
            "age_max": torch.tensor([32.0, 25.0]),
            "age_mask": torch.tensor([1.0, 0.0]),  # Second sample has invalid/missing age
            "gender": torch.tensor([1.0, 0.0]),
            "gender_mask": torch.tensor([1.0, 1.0]),
            "dimorphism_weight": torch.tensor([1.0, 1.0]),
        }
        total_loss, _ = criterion(preds, targets)
        total_loss.backward()
        assert age_logits.grad is not None
        assert torch.all(age_logits.grad[1] == 0.0)
        assert not torch.all(age_logits.grad[0] == 0.0)
        assert log_var.grad is not None
        assert log_var.grad[1].item() == 0.0
        assert log_var.grad[0].item() != 0.0

    @pytest.mark.parametrize("use_coupled_laplace", [False, True])
    def test_heterogeneous_alpha_sample_weight_zeros_untrusted_sample_gradients(self, use_coupled_laplace):
        """Per-sample alpha_sample_weight == 0.0 must zero out all gradients for that specific sample."""
        criterion = UnifiedBiometricLoss(
            weight_interval=0.25,
            weight_reid=0.15,
            use_coupled_laplace=use_coupled_laplace,
        )
        B = 6
        age_logits = torch.randn(B, 101, requires_grad=True)
        age_probs = torch.softmax(age_logits, dim=-1)
        expected_age = torch.sum(age_probs * torch.arange(101, dtype=torch.float32), dim=-1)
        gender_logits = torch.randn(B, requires_grad=True)
        log_var = torch.randn(B, requires_grad=True)
        reid_emb = torch.randn(B, 64, requires_grad=True)

        preds = {
            "age_probs": age_probs,
            "age_logits": age_logits,
            "expected_age": expected_age,
            "gender_logit": gender_logits,
            "log_variance": log_var,
            "reid_embedding": reid_emb,
        }
        targets = {
            "ldl_101": torch.softmax(torch.randn(B, 101), dim=-1),
            "age_continuous": torch.tensor([25.0, 45.0, 25.0, 35.0, 35.0, 45.0]),
            "age_min": torch.tensor([23.0, 43.0, 23.0, 33.0, 33.0, 43.0]),
            "age_max": torch.tensor([27.0, 47.0, 27.0, 37.0, 37.0, 47.0]),
            "age_mask": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            "gender": torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
            "gender_mask": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            "dimorphism_weight": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            "identity": torch.tensor([7, 8, 7, 9, 9, 8]),
            "alpha_sample_weight": torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 0.0]),  # Samples 1 and 5 have 0 confidence
        }

        total_loss, _ = criterion(preds, targets)
        total_loss.backward()

        for idx_zero in (1, 5):
            assert torch.all(age_logits.grad[idx_zero] == 0.0)
            assert gender_logits.grad[idx_zero].item() == 0.0
            assert log_var.grad[idx_zero].item() == 0.0
            assert torch.all(reid_emb.grad[idx_zero] == 0.0)
        for idx_active in (0, 2, 3, 4):
            assert not torch.all(age_logits.grad[idx_active] == 0.0)
            assert gender_logits.grad[idx_active].item() != 0.0
            assert log_var.grad[idx_active].item() != 0.0
            assert not torch.all(reid_emb.grad[idx_active] == 0.0)



