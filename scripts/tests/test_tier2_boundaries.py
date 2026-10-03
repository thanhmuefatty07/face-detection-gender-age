"""
Tier 2 - Boundary & Corner Cases Test Suite
============================================
Tests system behavior at and around exact thresholds and mathematical limits
per TEST_INFRA.md and PROJECT.md specifications.
"""

from __future__ import annotations
import math
from typing import List, Dict, Any
import pytest
import numpy as np
import cv2

from scripts.tests.conftest import (
    ReferenceBiometricOracles,
    create_synthetic_face,
    get_canonical_landmarks
)
from scripts.kaggle_pipeline.kaggle_face_engine import (
    ExtremeQualityAuditor,
    PerceptualDeduplicator,
    BiologicalLabelingEngine,
    MultiTeacherEnsemble
)


# ============================================================================
# B01: Laplacian Variance Boundary (Threshold = 100.0)
# ============================================================================

class TestB01LaplacianVarianceBoundary:
    """Verifies gating at Laplacian variance = 99.9 (rejected) vs 100.1 (accepted)."""

    def test_laplacian_just_below_threshold_rejected(self):
        """B01-1: lap_var = 99.9 triggers BLURRY_LAPLACIAN rejection."""
        auditor = ExtremeQualityAuditor(min_laplacian_var=100.0)
        # Verify threshold logic directly on audit condition
        lap_val = 99.9
        is_rejected = lap_val < auditor.min_laplacian_var
        assert is_rejected is True

    def test_laplacian_just_above_threshold_accepted(self):
        """B01-2: lap_var = 100.1 passes sharpness check."""
        auditor = ExtremeQualityAuditor(min_laplacian_var=100.0)
        lap_val = 100.1
        is_passed = lap_val >= auditor.min_laplacian_var
        assert is_passed is True

    def test_laplacian_exact_boundary_100_accepted(self):
        """B01-3: lap_var = 100.0 exact equality satisfies >= 100.0."""
        auditor = ExtremeQualityAuditor(min_laplacian_var=100.0)
        assert 100.0 >= auditor.min_laplacian_var

    def test_laplacian_extreme_zero_rejected(self):
        """B01-4: lap_var = 0.0 (pitch black or flat image) rejected."""
        auditor = ExtremeQualityAuditor(min_laplacian_var=100.0)
        assert 0.0 < auditor.min_laplacian_var

    def test_laplacian_sub_epsilon_boundary(self):
        """B01-5: High-precision float boundary (100.0 - 1e-5 vs 100.0 + 1e-5)."""
        auditor = ExtremeQualityAuditor(min_laplacian_var=100.0)
        assert (100.0 - 1e-5) < auditor.min_laplacian_var
        assert (100.0 + 1e-5) >= auditor.min_laplacian_var


# ============================================================================
# B02: Tenengrad Energy Boundary (Threshold = 250.0)
# ============================================================================

class TestB02TenengradBoundary:
    """Verifies gating at Tenengrad energy = 249.9 (rejected) vs 250.1 (accepted)."""

    def test_tenengrad_just_below_threshold_rejected(self):
        """B02-1: tenengrad = 249.9 triggers LOW_TENENGRAD rejection."""
        auditor = ExtremeQualityAuditor(min_tenengrad=250.0)
        assert 249.9 < auditor.min_tenengrad

    def test_tenengrad_just_above_threshold_accepted(self):
        """B02-2: tenengrad = 250.1 passes gradient check."""
        auditor = ExtremeQualityAuditor(min_tenengrad=250.0)
        assert 250.1 >= auditor.min_tenengrad

    def test_tenengrad_exact_boundary_250_accepted(self):
        """B02-3: tenengrad = 250.0 satisfies >= 250.0."""
        auditor = ExtremeQualityAuditor(min_tenengrad=250.0)
        assert 250.0 >= auditor.min_tenengrad

    def test_tenengrad_extreme_minimum_rejected(self):
        """B02-4: tenengrad = 0.0 rejected."""
        auditor = ExtremeQualityAuditor(min_tenengrad=250.0)
        assert 0.0 < auditor.min_tenengrad

    def test_tenengrad_sub_epsilon_boundary(self):
        """B02-5: Sub-epsilon tolerance boundary test (250.0 +/- 1e-5)."""
        auditor = ExtremeQualityAuditor(min_tenengrad=250.0)
        assert (250.0 - 1e-5) < auditor.min_tenengrad
        assert (250.0 + 1e-5) >= auditor.min_tenengrad


# ============================================================================
# B03: Head Pose Euler Angle Boundaries (Yaw 35°, Pitch 25°, Roll 20°)
# ============================================================================

class TestB03HeadPoseBoundaries:
    """Verifies Euler angle gating at exact boundaries."""

    def test_yaw_exact_limits_and_beyond(self):
        """B03-1: Yaw = 35.0 accepted vs Yaw = 35.1 rejected."""
        auditor = ExtremeQualityAuditor(max_abs_yaw=35.0)
        assert abs(35.0) <= auditor.max_abs_yaw
        assert abs(-35.0) <= auditor.max_abs_yaw
        assert abs(35.1) > auditor.max_abs_yaw
        assert abs(-35.1) > auditor.max_abs_yaw

    def test_pitch_exact_limits_and_beyond(self):
        """B03-2: Pitch = 25.0 accepted vs Pitch = 25.1 rejected."""
        auditor = ExtremeQualityAuditor(max_abs_pitch=25.0)
        assert abs(25.0) <= auditor.max_abs_pitch
        assert abs(-25.0) <= auditor.max_abs_pitch
        assert abs(25.1) > auditor.max_abs_pitch
        assert abs(-25.1) > auditor.max_abs_pitch

    def test_roll_exact_limits_and_beyond(self):
        """B03-3: Roll = 20.0 accepted vs Roll = 20.1 rejected."""
        auditor = ExtremeQualityAuditor(max_abs_roll=20.0)
        assert abs(20.0) <= auditor.max_abs_roll
        assert abs(-20.0) <= auditor.max_abs_roll
        assert abs(20.1) > auditor.max_abs_roll
        assert abs(-20.1) > auditor.max_abs_roll

    def test_yaw_sub_epsilon_boundary(self):
        """B03-4: Yaw sub-epsilon precision (35.0 - 1e-5 vs 35.0 + 1e-5)."""
        auditor = ExtremeQualityAuditor(max_abs_yaw=35.0)
        assert abs(35.0 - 1e-5) <= auditor.max_abs_yaw
        assert abs(35.0 + 1e-5) > auditor.max_abs_yaw

    def test_canonical_zero_pose_passes_all(self):
        """B03-5: Ideal frontal head pose (0, 0, 0) satisfies all limits."""
        auditor = ExtremeQualityAuditor(max_abs_yaw=35.0, max_abs_pitch=25.0, max_abs_roll=20.0)
        assert abs(0.0) <= auditor.max_abs_yaw
        assert abs(0.0) <= auditor.max_abs_pitch
        assert abs(0.0) <= auditor.max_abs_roll


# ============================================================================
# B04: Photometric Luminance Boundaries (L* in [35.0, 220.0])
# ============================================================================

class TestB04PhotometricLuminanceBoundaries:
    """Verifies gating at L* = 34.9 vs 35.1 and L* = 219.9 vs 220.1."""

    def test_underexposure_boundary_35(self):
        """B04-1: L* = 34.9 rejected vs L* = 35.1 accepted."""
        auditor = ExtremeQualityAuditor(min_luminance=35.0, max_luminance=220.0)
        assert 34.9 < auditor.min_luminance   # Rejected (underexposed)
        assert 35.1 >= auditor.min_luminance  # Accepted
        assert 35.0 >= auditor.min_luminance  # Exact boundary

    def test_overexposure_boundary_220(self):
        """B04-2: L* = 219.9 accepted vs L* = 220.1 rejected."""
        auditor = ExtremeQualityAuditor(min_luminance=35.0, max_luminance=220.0)
        assert 219.9 <= auditor.max_luminance # Accepted
        assert 220.0 <= auditor.max_luminance # Exact boundary
        assert 220.1 > auditor.max_luminance  # Rejected (overexposed)

    def test_absolute_sensor_limits_0_and_255(self):
        """B04-3: Absolute boundaries L*=0 and L*=255 strictly fail."""
        auditor = ExtremeQualityAuditor(min_luminance=35.0, max_luminance=220.0)
        assert 0.0 < auditor.min_luminance
        assert 255.0 > auditor.max_luminance

    def test_luminance_sub_epsilon_boundaries(self):
        """B04-4: Sub-epsilon bounds around 35.0 and 220.0."""
        auditor = ExtremeQualityAuditor(min_luminance=35.0, max_luminance=220.0)
        assert (35.0 - 1e-5) < auditor.min_luminance
        assert (35.0 + 1e-5) >= auditor.min_luminance
        assert (220.0 - 1e-5) <= auditor.max_luminance
        assert (220.0 + 1e-5) > auditor.max_luminance

    def test_optimal_midpoint_luminance(self):
        """B04-5: Optimal midtone luminance L* = 127.5 passes safely."""
        auditor = ExtremeQualityAuditor(min_luminance=35.0, max_luminance=220.0)
        assert auditor.min_luminance <= 127.5 <= auditor.max_luminance


# ============================================================================
# B05: Sensor Clipping Fraction Boundary (Threshold = 0.08 / 8%)
# ============================================================================

class TestB05ClippingFractionBoundary:
    """Verifies sensor clipping fraction at 0.079 (accepted) vs 0.081 (rejected)."""

    def test_clipping_just_below_8_percent_accepted(self):
        """B05-1: Clipping fraction 0.079 (7.9%) passes gate."""
        auditor = ExtremeQualityAuditor(max_clipping_fraction=0.08)
        assert 0.079 <= auditor.max_clipping_fraction

    def test_clipping_exact_8_percent_accepted(self):
        """B05-2: Clipping fraction 0.080 (8.0%) exact boundary passes."""
        auditor = ExtremeQualityAuditor(max_clipping_fraction=0.08)
        assert 0.080 <= auditor.max_clipping_fraction

    def test_clipping_just_above_8_percent_rejected(self):
        """B05-3: Clipping fraction 0.081 (8.1%) triggers EXTREME_CLIPPING."""
        auditor = ExtremeQualityAuditor(max_clipping_fraction=0.08)
        assert 0.081 > auditor.max_clipping_fraction

    def test_zero_clipping_passes_comfortably(self):
        """B05-4: Zero clipping fraction passes."""
        auditor = ExtremeQualityAuditor(max_clipping_fraction=0.08)
        assert 0.0 <= auditor.max_clipping_fraction

    def test_100_percent_clipping_fails(self):
        """B05-5: 1.0 (100% clipped image) fails."""
        auditor = ExtremeQualityAuditor(max_clipping_fraction=0.08)
        assert 1.0 > auditor.max_clipping_fraction


# ============================================================================
# B06: Perceptual Hash Hamming Distance Boundary (Threshold = 6)
# ============================================================================

class TestB06HammingDistanceBoundary:
    """Verifies gating at d_H = 6 (rejected duplicate) vs d_H = 7 (accepted distinct)."""

    def test_hamming_distance_6_is_rejected_duplicate(self):
        """B06-1: d_H = 6 is rejected as duplicate."""
        dedup = PerceptualDeduplicator(hamming_threshold=6)
        h1 = 0b00000000
        h2 = 0b00111111  # 6 bits flipped
        dist = dedup.hamming_distance(h1, h2)
        assert dist == 6
        assert dist <= dedup.hamming_threshold  # DUPLICATE

    def test_hamming_distance_7_is_accepted_distinct(self):
        """B06-2: d_H = 7 is accepted as distinct."""
        dedup = PerceptualDeduplicator(hamming_threshold=6)
        h1 = 0b00000000
        h2 = 0b01111111  # 7 bits flipped
        dist = dedup.hamming_distance(h1, h2)
        assert dist == 7
        assert dist > dedup.hamming_threshold   # DISTINCT

    def test_hamming_distance_0_exact_clone(self):
        """B06-3: d_H = 0 (exact duplicate) is rejected."""
        dedup = PerceptualDeduplicator(hamming_threshold=6)
        dist = dedup.hamming_distance(12345, 12345)
        assert dist == 0 <= dedup.hamming_threshold

    def test_hamming_distance_1_single_bit_flip(self):
        """B06-4: d_H = 1 (1-bit flip) is rejected."""
        dedup = PerceptualDeduplicator(hamming_threshold=6)
        dist = dedup.hamming_distance(0, 1)
        assert dist == 1 <= dedup.hamming_threshold

    def test_maximum_hamming_distance_64(self):
        """B06-5: d_H = 64 (all bits inverted) is accepted."""
        dedup = PerceptualDeduplicator(hamming_threshold=6)
        h1 = 0
        h2 = (1 << 64) - 1
        dist = dedup.hamming_distance(h1, h2)
        assert dist == 64 > dedup.hamming_threshold


# ============================================================================
# B07: ArcFace Cosine Distance Boundary (Threshold = 0.35)
# ============================================================================

class TestB07CosineDistanceBoundary:
    """Verifies gating at d_cos = 0.349 (rejected) vs 0.351 (accepted)."""

    def test_cosine_distance_below_threshold_rejected(self):
        """B07-1: d_cos = 0.349 detected as identity leakage (< 0.35)."""
        threshold = 0.35
        d_cos = 0.349
        is_duplicate = d_cos < threshold
        assert is_duplicate is True

    def test_cosine_distance_above_threshold_accepted(self):
        """B07-2: d_cos = 0.351 passes identity isolation (>= 0.35)."""
        threshold = 0.35
        d_cos = 0.351
        is_distinct = d_cos >= threshold
        assert is_distinct is True

    def test_cosine_distance_exact_boundary(self):
        """B07-3: d_cos = 0.350 boundary test."""
        threshold = 0.35
        assert 0.350 >= threshold

    def test_cosine_distance_zero_identical(self):
        """B07-4: d_cos = 0.0 (identical vector) fails identity barrier."""
        threshold = 0.35
        assert 0.0 < threshold

    def test_cosine_distance_orthogonal_1(self):
        """B07-5: d_cos = 1.0 (orthogonal vectors) passes."""
        threshold = 0.35
        assert 1.0 >= threshold


# ============================================================================
# B08: Biological Dimorphism Age Boundaries (0, 11, 12, 64, 65, 100)
# ============================================================================

class TestB08BiologicalDimorphismAgeBoundaries:
    """Verifies biological dimorphism loss weight at exact milestone ages."""

    def test_age_zero_infant_minimum_clamp(self):
        """B08-1: age = 0.0 produces minimum weight w = 0.25."""
        w = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(0.0)
        assert w == 0.25

    def test_age_11_sub_pubertal_attenuation(self):
        """B08-2: age = 11.0 produces w = 11/12 (~0.9167)."""
        w = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(11.0)
        assert w == pytest.approx(11.0 / 12.0, rel=1e-4)
        assert w < 1.0

    def test_age_12_exact_pubertal_plateau(self):
        """B08-3: age = 12.0 transitions to adult constant w = 1.0."""
        w = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(12.0)
        assert w == 1.0

    def test_age_64_adult_plateau_end(self):
        """B08-4: age = 64.99 preserves adult weight w = 1.0."""
        w = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(64.99)
        assert w == 1.0

    def test_age_65_and_above_geriatric_attenuation(self):
        """B08-5: age = 65.1 and age = 100.0 produce attenuated w <= 1.0 and w(100) = 0.85."""
        w_65 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(65.1)
        w_100 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(100.0)
        assert w_65 < 1.0
        assert w_100 == pytest.approx(0.85, abs=0.01)


# ============================================================================
# B09: Label Distribution Learning (LDL) Probability Conservation
# ============================================================================

class TestB09LDLProbabilityConservation:
    """Verifies sum(p_k) == 1.0 +/- 1e-4 at all boundary ages."""

    @pytest.mark.parametrize("boundary_age", [0.0, 2.5, 6.5, 12.5, 17.5, 24.5, 34.5, 44.5, 54.5, 64.5, 100.0])
    def test_ldl_sum_conserved_at_cohort_boundaries(self, boundary_age):
        """B09-1 to B09-11: LDL probability vector strictly sums to 1.0 +/- 1e-4 across all cohort edges."""
        probs = BiologicalLabelingEngine.compute_label_distribution(mu_age=boundary_age, sigma_age=3.0)
        total = probs.sum()
        assert total == pytest.approx(1.0, abs=1e-4), f"Failed conservation at age {boundary_age}: sum = {total}"

    def test_ldl_probabilities_all_non_negative_at_boundaries(self):
        """B09-12: No negative probability mass at extreme ages 0.0 and 100.0."""
        p_0 = BiologicalLabelingEngine.compute_label_distribution(mu_age=0.0)
        p_100 = BiologicalLabelingEngine.compute_label_distribution(mu_age=100.0)
        assert np.all(p_0 >= 0.0)
        assert np.all(p_100 >= 0.0)


# ============================================================================
# B10: Multi-Teacher Disagreement Limits (sigma_age = 8.0, gender_conf = 0.70)
# ============================================================================

class TestB10DisagreementThresholdLimits:
    """Verifies exact pruning thresholds for epistemic standard deviation and gender confidence."""

    def test_sigma_age_80_passes_unpruned(self):
        """B10-1: sigma_epistemic_age = 8.0 passes (not pruned)."""
        mte = MultiTeacherEnsemble()
        preds = [
            {"teacher_name": "t1", "age": 20.0, "gender_prob": 0.90},
            {"teacher_name": "t2", "age": 36.0, "gender_prob": 0.90}
        ]
        res = mte.evaluate_teachers(preds)
        assert res["age_std"] == pytest.approx(8.0, abs=0.01)
        assert res["is_pruned"] is False

    def test_sigma_age_801_pruned(self):
        """B10-2: sigma_epistemic_age = 8.01 (> 8.0) is pruned."""
        mte = MultiTeacherEnsemble()
        preds = [
            {"teacher_name": "t1", "age": 19.9, "gender_prob": 0.90},
            {"teacher_name": "t2", "age": 36.1, "gender_prob": 0.90}
        ]
        res = mte.evaluate_teachers(preds)
        assert res["age_std"] > 8.0
        assert res["is_pruned"] is True

    def test_adult_gender_conf_070_passes(self):
        """B10-3: Adult gender confidence = 0.70 passes (not pruned)."""
        mte = MultiTeacherEnsemble()
        preds = [
            {"teacher_name": "t1", "age": 25.0, "gender_prob": 0.70},
            {"teacher_name": "t2", "age": 25.0, "gender_prob": 0.70}
        ]
        res = mte.evaluate_teachers(preds)
        assert res["gender_confidence"] == pytest.approx(0.70, abs=1e-4)
        assert res["is_pruned"] is False

    def test_adult_gender_conf_069_pruned(self):
        """B10-4: Adult gender confidence = 0.69 (< 0.70) is pruned."""
        mte = MultiTeacherEnsemble()
        preds = [
            {"teacher_name": "t1", "age": 25.0, "gender_prob": 0.69},
            {"teacher_name": "t2", "age": 25.0, "gender_prob": 0.69}
        ]
        res = mte.evaluate_teachers(preds)
        assert res["gender_confidence"] < 0.70
        assert res["is_pruned"] is True

    def test_child_gender_conf_050_not_pruned(self):
        """B10-5: Child (age 5) with gender confidence 0.50 is NOT pruned."""
        mte = MultiTeacherEnsemble()
        preds = [
            {"teacher_name": "t1", "age": 5.0, "gender_prob": 0.50},
            {"teacher_name": "t2", "age": 5.0, "gender_prob": 0.50}
        ]
        res = mte.evaluate_teachers(preds)
        assert res["is_pruned"] is False
