"""
Tier 3 - Cross-Feature Interactions Test Suite
================================================
Pairwise combinatorial testing of interacting features per TEST_INFRA.md:
- Extreme head pose + blur filtering
- Dimorphism uncertainty + Gaussian/Laplacian LDL distributions
- pHash / embedding deduplication + dataset split assignment (train vs val)
- Multi-teacher disagreement + Alpha signal pruning
- Photometric gating + facial landmark detection
- Umeyama alignment + sharpness conservation
- Ethnic aging kinetics + demographic calibration
"""

from __future__ import annotations
import math
from typing import List, Dict, Any
import pytest
import numpy as np
import cv2
import pandas as pd

from scripts.tests.conftest import (
    ReferenceBiometricOracles,
    create_synthetic_face,
    get_canonical_landmarks
)
from scripts.kaggle_pipeline.kaggle_face_engine import (
    FacialLandmarkDetector,
    FaceGeometryAligner,
    ExtremeQualityAuditor,
    PerceptualDeduplicator,
    BiologicalLabelingEngine,
    MultiTeacherEnsemble
)


# ============================================================================
# I01: Extreme Head Pose + Blur Filtering Interactions
# ============================================================================

class TestExtremePoseAndBlurInteractions:
    """Pairwise interaction between head pose angles and sharpness metrics."""

    def test_sharp_face_with_boundary_yaw_passes(self, sharp_face, canonical_landmarks):
        """I01-1: High sharpness + within-limit yaw passes both gates."""
        auditor = ExtremeQualityAuditor(min_laplacian_var=100.0, max_abs_yaw=35.0)
        # Pass canonical landmarks (yaw ~ 0 <= 35) with sharp image
        res = auditor.audit_face(sharp_face, landmarks_5=canonical_landmarks)
        assert res["laplacian_var"] >= 100.0
        assert abs(res["yaw"]) <= 35.0
        assert not any("BLURRY" in r for r in res["reasons"])

    def test_blurry_face_with_boundary_yaw_rejected(self, blurry_face, canonical_landmarks):
        """I01-2: Low sharpness + valid yaw is rejected specifically for blur."""
        auditor = ExtremeQualityAuditor(min_laplacian_var=100.0, max_abs_yaw=35.0)
        res = auditor.audit_face(blurry_face, landmarks_5=canonical_landmarks)
        assert res["passed"] is False
        assert any("BLURRY_LAPLACIAN" in r for r in res["reasons"])

    def test_sharp_face_with_excessive_yaw_rejected(self, sharp_face, canonical_landmarks):
        """I01-3: High sharpness + excessive yaw is rejected specifically for pose."""
        auditor = ExtremeQualityAuditor(min_laplacian_var=100.0, max_abs_yaw=35.0)
        extreme_yaw_lms = canonical_landmarks.copy()
        extreme_yaw_lms[0, 0] += 35.0  # severely skew face
        res = auditor.audit_face(sharp_face, landmarks_5=extreme_yaw_lms)
        assert res["passed"] is False
        # Sharpness passed, but pose failed
        assert res["laplacian_var"] >= 100.0
        assert any("EXCESSIVE" in r for r in res["reasons"])

    def test_blurry_face_with_excessive_yaw_logs_multiple_failures(self, blurry_face, canonical_landmarks):
        """I01-4: Both blur and excessive yaw trigger multiple distinct rejection reasons."""
        auditor = ExtremeQualityAuditor(min_laplacian_var=100.0, max_abs_yaw=35.0)
        extreme_yaw_lms = canonical_landmarks.copy()
        extreme_yaw_lms[0, 0] += 35.0
        res = auditor.audit_face(blurry_face, landmarks_5=extreme_yaw_lms)
        assert res["passed"] is False
        assert len(res["reasons"]) >= 2
        assert any("BLURRY" in r for r in res["reasons"])

    def test_tenengrad_and_pitch_interaction(self, sharp_face, canonical_landmarks):
        """I01-5: Tenengrad gradient energy interacting with pitch gating."""
        auditor = ExtremeQualityAuditor(min_tenengrad=250.0, max_abs_pitch=25.0)
        res = auditor.audit_face(sharp_face, landmarks_5=canonical_landmarks)
        assert res["tenengrad"] >= 250.0


# ============================================================================
# I02: Dimorphism Uncertainty + Gaussian / Laplacian LDL Interactions
# ============================================================================

class TestDimorphismAndLDLInteractions:
    """Interactions between biological dimorphism reweighting and LDL distributions."""

    def test_prepubertal_infant_dimorphism_and_gaussian_ldl(self):
        """I02-1: Infant (age=1.0) has minimum dimorphism weight 0.25 and LDL mass at '0-2'."""
        w = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(1.0)
        ldl = BiologicalLabelingEngine.compute_label_distribution(mu_age=1.0, sigma_age=2.0)
        assert w == 0.25
        assert np.argmax(ldl) == 0  # '0-2' bin
        assert ldl[0] > 0.60

    def test_prepubertal_child_dimorphism_and_ldl(self):
        """I02-2: Child (age=8.0) has attenuated dimorphism 8/12 and LDL mass at '7-12'."""
        w = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(8.0)
        ldl = BiologicalLabelingEngine.compute_label_distribution(mu_age=8.0, sigma_age=2.5)
        assert w == pytest.approx(8.0 / 12.0, rel=1e-4)
        assert np.argmax(ldl) == 2  # '7-12' bin
        assert ldl.sum() == pytest.approx(1.0, abs=1e-4)

    def test_adult_dimorphism_and_ldl(self):
        """I02-3: Adult (age=30.0) has full weight 1.0 and Gaussian LDL peak at '25-34'."""
        w = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(30.0)
        ldl = BiologicalLabelingEngine.compute_label_distribution(mu_age=30.0, sigma_age=3.0)
        assert w == 1.0
        assert np.argmax(ldl) == 5  # '25-34' bin

    def test_elderly_dimorphism_and_laplacian_ldl(self):
        """I02-4: Elderly (age=75.0) has attenuated dimorphism and heavy-tailed Laplacian LDL."""
        w = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(75.0)
        ldl = BiologicalLabelingEngine.compute_label_distribution(
            mu_age=75.0, sigma_age=4.0, distribution_type="laplacian"
        )
        assert w < 1.0
        assert np.argmax(ldl) == 9  # '65+' bin
        assert ldl.sum() == pytest.approx(1.0, abs=1e-4)

    def test_boundary_transition_at_12_years(self):
        """I02-5: Age 12.0 exactly transitions dimorphism to 1.0 while LDL spans both bins."""
        w = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(12.0)
        ldl = BiologicalLabelingEngine.compute_label_distribution(mu_age=12.0, sigma_age=2.5)
        assert w == 1.0
        # Both '7-12' (idx 2) and '13-17' (idx 3) receive substantial probability mass
        assert ldl[2] > 0.20 and ldl[3] > 0.20


# ============================================================================
# I03: pHash Deduplication + Split Assignment Interactions (Zero Cross-Split Leakage)
# ============================================================================

class TestPHashAndSplitDeduplicationInteractions:
    """Interactions between perceptual hashing, embedding distance, and dataset partitions."""

    def test_train_val_cross_split_near_duplicate_rejected(self, sharp_face):
        """I03-1: Image in Train rejects near-duplicate candidate (d_H <= 6) from entering Val."""
        dedup = PerceptualDeduplicator(hamming_threshold=6)
        gray_train = cv2.cvtColor(sharp_face, cv2.COLOR_BGR2GRAY)
        h_train = dedup.compute_phash_uint64(gray_train)
        # Register into Train partition
        is_dup, tag = dedup.is_duplicate_and_register(h_train)
        assert is_dup is False

        # Attempt to register perturbed version (e.g. slight brightness +2) into Val partition
        gray_val_candidate = np.clip(gray_train.astype(np.int16) + 2, 0, 255).astype(np.uint8)
        h_val = dedup.compute_phash_uint64(gray_val_candidate)
        is_dup_val, tag_val = dedup.is_duplicate_and_register(h_val)

        # Must be rejected to prevent cross-partition leakage!
        assert is_dup_val is True
        assert "DUPLICATE" in tag_val

    def test_train_val_distinct_image_accepted(self, sharp_face):
        """I03-2: Distinct image (d_H > 6) is accepted into Val partition."""
        dedup = PerceptualDeduplicator(hamming_threshold=6)
        gray1 = cv2.cvtColor(sharp_face, cv2.COLOR_BGR2GRAY)
        h1 = dedup.compute_phash_uint64(gray1)
        dedup.is_duplicate_and_register(h1)

        # Distinct checkerboard pattern
        gray2 = np.zeros((160, 160), dtype=np.uint8)
        gray2[::16, :] = 255
        h2 = dedup.compute_phash_uint64(gray2)
        is_dup, tag = dedup.is_duplicate_and_register(h2)
        assert is_dup is False
        assert tag == "UNIQUE"

    def test_dual_layer_phash_distinct_but_embedding_leakage(self, sharp_face):
        """I03-3: Image with d_H > 6 but identical embedding is rejected by embedding layer."""
        dedup = PerceptualDeduplicator(hamming_threshold=6, cosine_threshold=0.35)
        emb1 = dedup.extract_embedding_512d(sharp_face)
        # Hash 100 registered with embedding emb1
        dedup.is_duplicate_and_register(hash_val=100, embedding_512d=emb1)

        # Distinct hash 9999999 (d_H > 6), but near-identical embedding
        emb2 = emb1 + np.random.normal(0, 0.02, 512).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)
        is_dup, tag = dedup.is_duplicate_and_register(hash_val=9999999, embedding_512d=emb2)
        assert is_dup is True
        assert tag == "EMBEDDING_IDENTITY_LEAKAGE"

    def test_dual_layer_both_unique_accepted(self):
        """I03-4: Candidate with both unique pHash and unique embedding is registered."""
        dedup = PerceptualDeduplicator(hamming_threshold=6, cosine_threshold=0.35)
        e1 = np.zeros(512, dtype=np.float32); e1[0] = 1.0
        e2 = np.zeros(512, dtype=np.float32); e2[50] = 1.0
        dedup.is_duplicate_and_register(hash_val=0b11111111, embedding_512d=e1)
        is_dup, tag = dedup.is_duplicate_and_register(hash_val=0b00000000, embedding_512d=e2)
        assert is_dup is False
        assert tag == "UNIQUE"

    def test_batch_cross_partition_isolation_guarantee(self):
        """I03-5: No train-val pair has Hamming distance <= 6 in cleanly split sets."""
        train_hashes = [0x0000000000000000, 0x00000000000000ff]
        val_hashes = [0x00000000ffff0000, 0xffffffffffffffff]
        for th in train_hashes:
            for vh in val_hashes:
                dist = PerceptualDeduplicator.hamming_distance(th, vh)
                assert dist > 6, f"Cross-split leakage detected: dist={dist} <= 6"


# ============================================================================
# I04: Multi-Teacher Disagreement + Alpha Signal Pruning Interactions
# ============================================================================

class TestMultiTeacherDisagreementAndAlphaPruningInteractions:
    """Interactions between multi-teacher consensus metrics and Alpha Signal extraction."""

    def test_golden_consensus_produces_high_alpha_weight(self, golden_teacher_predictions):
        """I04-1: Unanimous consensus produces is_pruned=False and high alpha_sample_weight."""
        mte = MultiTeacherEnsemble()
        res = mte.evaluate_teachers(golden_teacher_predictions)
        assert res["is_pruned"] is False
        assert res["alpha_sample_weight"] >= 0.70
        assert res["shannon_entropy"] < 0.35

    def test_polarized_age_teachers_triggers_pruning(self, polarized_teacher_predictions):
        """I04-2: Age disagreement (sigma > 8.0) triggers pruning and drops alpha weight."""
        mte = MultiTeacherEnsemble()
        res = mte.evaluate_teachers(polarized_teacher_predictions)
        assert res["is_pruned"] is True
        assert res["age_std"] > 8.0
        assert res["alpha_sample_weight"] < 0.20

    def test_adult_gender_confusion_triggers_pruning(self, ambiguous_gender_adult_predictions):
        """I04-3: Adult gender confusion (conf < 0.70) triggers pruning and high entropy."""
        mte = MultiTeacherEnsemble()
        res = mte.evaluate_teachers(ambiguous_gender_adult_predictions)
        assert res["is_pruned"] is True
        assert res["shannon_entropy"] > 0.90

    def test_prepubertal_child_gender_ambiguity_passes(self, prepubertal_child_predictions):
        """I04-4: Prepubertal child gender ambiguity is allowed under dimorphism policy."""
        mte = MultiTeacherEnsemble()
        res = mte.evaluate_teachers(prepubertal_child_predictions)
        # Because age < 12, gender confusion does not prune sample
        assert res["is_pruned"] is False
        assert res["consensus_age"] < 12.0

    def test_alpha_weight_scales_with_consensus_certainty(self):
        """I04-5: Monotonic relationship: higher agreement -> higher alpha importance."""
        mte = MultiTeacherEnsemble()
        # Perfect agreement
        t_perfect = [{"teacher_name": "t1", "age": 25.0, "gender_prob": 1.0},
                     {"teacher_name": "t2", "age": 25.0, "gender_prob": 1.0}]
        # Slight disagreement
        t_slight = [{"teacher_name": "t1", "age": 23.0, "gender_prob": 0.85},
                    {"teacher_name": "t2", "age": 27.0, "gender_prob": 0.80}]
        # Moderate disagreement
        t_mod = [{"teacher_name": "t1", "age": 20.0, "gender_prob": 0.75},
                 {"teacher_name": "t2", "age": 30.0, "gender_prob": 0.70}]

        w_perf = mte.evaluate_teachers(t_perfect)["alpha_sample_weight"]
        w_slight = mte.evaluate_teachers(t_slight)["alpha_sample_weight"]
        w_mod = mte.evaluate_teachers(t_mod)["alpha_sample_weight"]

        assert w_perf > w_slight > w_mod


# ============================================================================
# I05: Photometric Gating + Facial Landmark Detection Interactions
# ============================================================================

class TestPhotometricAndLandmarkInteractions:
    """Interactions between lighting conditions and landmark detection / auditing."""

    def test_well_lit_face_passes_both_landmarks_and_photometry(self, sharp_face):
        """I05-1: Standard lighting passes landmark detector and photometric audit."""
        det = FacialLandmarkDetector()
        auditor = ExtremeQualityAuditor()
        ok, lms, conf = det.detect_landmarks(sharp_face)
        res = auditor.audit_face(sharp_face, landmarks_5=lms)
        assert ok is True
        assert conf >= 0.50
        assert 35.0 <= res["mean_luminance"] <= 220.0

    def test_severely_dark_face_fails_photometric_gate(self, underexposed_face):
        """I05-2: Underexposed face fails photometric gate regardless of landmarks."""
        det = FacialLandmarkDetector()
        auditor = ExtremeQualityAuditor(min_luminance=35.0)
        _, lms, _ = det.detect_landmarks(underexposed_face)
        res = auditor.audit_face(underexposed_face, landmarks_5=lms)
        assert res["passed"] is False
        assert any("UNDEREXPOSED" in r for r in res["reasons"])

    def test_severely_overexposed_face_fails_photometric_gate(self, overexposed_face):
        """I05-3: Overexposed face fails photometric gate."""
        det = FacialLandmarkDetector()
        auditor = ExtremeQualityAuditor(max_luminance=220.0)
        _, lms, _ = det.detect_landmarks(overexposed_face)
        res = auditor.audit_face(overexposed_face, landmarks_5=lms)
        assert res["passed"] is False
        assert any("OVEREXPOSED" in r for r in res["reasons"])


# ============================================================================
# I06: Umeyama Similarity Transform + Sharpness Conservation Interactions
# ============================================================================

class TestUmeyamaAlignmentAndSharpnessInteractions:
    """Interactions between geometric alignment and spatial sharpness metrics."""

    def test_aligning_sharp_face_preserves_sharpness(self, sharp_face, canonical_landmarks):
        """I06-1: Umeyama Lanczos4 warp preserves high Laplacian variance (>= 100)."""
        aligner = FaceGeometryAligner(output_size=160)
        aligned, _ = aligner.align(sharp_face, canonical_landmarks)
        gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        lap_aligned = ExtremeQualityAuditor.compute_laplacian_var(gray_aligned)
        assert lap_aligned >= 100.0

    def test_aligning_blurry_face_cannot_hallucinate_sharpness(self, blurry_face, canonical_landmarks):
        """I06-2: Warping a blurry face does not produce false high sharpness (remains < 100)."""
        aligner = FaceGeometryAligner(output_size=160)
        aligned, _ = aligner.align(blurry_face, canonical_landmarks)
        gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        lap_aligned = ExtremeQualityAuditor.compute_laplacian_var(gray_aligned)
        assert lap_aligned < 100.0


# ============================================================================
# I07: Ethnic Aging Kinetics + Demographic Calibration Interactions
# ============================================================================

class TestEthnicKineticsAndDemographicCalibrationInteractions:
    """Interactions between ethnic aging curves and demographic calibration."""

    def test_east_asian_young_adult_offset_applied(self):
        """I07-1: East Asian 30-year-old gets youthful calibrated apparent age."""
        calib_age, factor = BiologicalLabelingEngine.calibrate_apparent_age(30.0, "East Asian")
        assert calib_age < 30.0
        assert factor < 1.0

    def test_east_asian_post_cliff_offset_applied(self):
        """I07-2: East Asian 55-year-old gets accelerated apparent age after cliff."""
        calib_age, factor = BiologicalLabelingEngine.calibrate_apparent_age(55.0, "East Asian")
        assert calib_age > 50.0

    def test_african_melanin_preservation_offset_applied(self):
        """I07-3: African 45-year-old gets photoprotection youthful offset."""
        calib_age, factor = BiologicalLabelingEngine.calibrate_apparent_age(45.0, "African")
        assert calib_age < 45.0
        assert factor <= 1.0
