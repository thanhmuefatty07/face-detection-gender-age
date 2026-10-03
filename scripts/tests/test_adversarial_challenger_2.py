"""
Adversarial Stress Test Suite - Challenger 2
=============================================
Milestones 2 & 3: Empirical Stress-Testing of Deduplication, Biological Morphometry, and LDL.

Mission Objectives:
1. pHash Hamming distance boundary: dH <= 6 rejected vs dH > 6 accepted.
2. 512-d ArcFace cosine distance boundary: d_cos < 0.35 duplicate vs d_cos >= 0.35 distinct.
3. Label Distribution Learning (LDL) mathematical conservation: sum(p) == 1.0 +/- 1e-4 for all ages in [0, 100] for both Gaussian and Laplacian models.
4. Sexual dimorphism weight transition across lifespan: w(0)=0.25, linear ramp to w(12)=1.0, plateau [12, 65]=1.0, and geriatric drop w(>65)=0.85 with aleatoric uncertainty.
5. Multi-Teacher disagreement pruning: sigma_epistemic > 8.0 or confidence < 0.70 correctly prunes samples.
"""

from __future__ import annotations
import math
import pytest
import numpy as np
import cv2

from scripts.kaggle_pipeline.kaggle_face_engine import (
    PerceptualDeduplicator,
    BiologicalLabelingEngine,
    MultiTeacherEnsemble
)


# ============================================================================
# Probe Suite 1: pHash Deduplication & Hamming Distance Boundary (dH <= 6 vs dH > 6)
# ============================================================================

class TestAdversarialPHashBoundary:
    """Empirical probe of 64-bit DCT pHash deduplication boundary and bit flips."""

    @pytest.mark.parametrize("bit_flip_count", [0, 1, 2, 3, 4, 5, 6])
    def test_hamming_distance_lte_6_strictly_rejected(self, bit_flip_count):
        """Verify that dH <= 6 is strictly rejected as duplicate across diverse bit masks."""
        dedup = PerceptualDeduplicator(hamming_threshold=6)
        base_hash = 0x5A5A5A5A5A5A5A5A  # 64-bit alternating pattern
        dedup.is_duplicate_and_register(base_hash)

        # Generate bit flips
        perturbed_hash = base_hash
        for bit_idx in range(bit_flip_count):
            perturbed_hash ^= (1 << (bit_idx * 7))

        dist = dedup.hamming_distance(base_hash, perturbed_hash)
        assert dist == bit_flip_count

        is_dup, tag = dedup.is_duplicate_and_register(perturbed_hash)
        assert is_dup is True, f"Failed: dH={dist} was not rejected as duplicate"
        assert tag == "PHASH_NEAR_DUPLICATE"

    @pytest.mark.parametrize("bit_flip_count", [7, 8, 12, 16, 32, 64])
    def test_hamming_distance_gt_6_strictly_accepted(self, bit_flip_count):
        """Verify that dH > 6 is accepted as distinct across diverse flip counts."""
        dedup = PerceptualDeduplicator(hamming_threshold=6)
        base_hash = 0x5A5A5A5A5A5A5A5A
        dedup.is_duplicate_and_register(base_hash)

        perturbed_hash = base_hash
        for bit_idx in range(bit_flip_count):
            perturbed_hash ^= (1 << bit_idx)

        dist = dedup.hamming_distance(base_hash, perturbed_hash)
        assert dist == bit_flip_count

        is_dup, tag = dedup.is_duplicate_and_register(perturbed_hash)
        assert is_dup is False, f"Failed: dH={dist} was incorrectly rejected as duplicate"
        assert tag == "UNIQUE"

    def test_boundary_exact_step_from_6_to_7(self):
        """Verify exact step transition from 6 (rejected) to 7 (accepted)."""
        dedup = PerceptualDeduplicator(hamming_threshold=6)
        h0 = 0x0000000000000000
        dedup.is_duplicate_and_register(h0)

        # 6-bit flip
        h_6 = 0b00111111
        is_dup_6, tag_6 = dedup.is_duplicate_and_register(h_6)
        assert is_dup_6 is True
        assert tag_6 == "PHASH_NEAR_DUPLICATE"

        # 7-bit flip
        h_7 = 0b01111111
        is_dup_7, tag_7 = dedup.is_duplicate_and_register(h_7)
        assert is_dup_7 is False
        assert tag_7 == "UNIQUE"

    def test_dual_layer_interaction_when_embeddings_provided(self):
        """
        Adversarial Probe: Dual-layer deduplicator interaction.
        Investigate whether pHash check is bypassed when distinct embedding is provided.
        """
        dedup = PerceptualDeduplicator(hamming_threshold=6, cosine_threshold=0.35)
        # Register image 1
        h1 = 0x1234567890ABCDEF
        e1 = np.zeros(512, dtype=np.float32)
        e1[0] = 1.0
        dedup.is_duplicate_and_register(h1, e1)

        # Candidate 2: Same/near-duplicate hash (dH = 2 <= 6) BUT orthogonal embedding (d_cos = 1.0 >= 0.35)
        h2 = h1 ^ 0b11  # 2 bit flips
        e2 = np.zeros(512, dtype=np.float32)
        e2[1] = 1.0
        is_dup, tag = dedup.is_duplicate_and_register(h2, e2)

        # Observe the behavior:
        # In current implementation, distinct embedding (d_cos >= 0.35) bypasses pHash near-collision
        # so is_dup is False, tag is 'UNIQUE'.
        # Document empirical observation:
        assert tag == "UNIQUE"


# ============================================================================
# Probe Suite 2: 512-d ArcFace Cosine Distance Boundary
# ============================================================================

class TestAdversarialCosineDistanceBoundary:
    """Empirical probe of 512-d embedding cosine distance gating boundary (0.35)."""

    def test_cosine_boundary_sub_epsilon_transition(self):
        """Verify sharp transition at 0.35 - eps (duplicate) vs 0.35 + eps (distinct)."""
        dedup = PerceptualDeduplicator(cosine_threshold=0.35)

        # Base vector e1 along axis 0
        e1 = np.zeros(512, dtype=np.float32)
        e1[0] = 1.0

        # We construct e_cand such that cos_sim = 1 - target_d_cos
        # e_cand = (1 - d_cos) * e1 + sqrt(1 - (1 - d_cos)^2) * e2
        def make_vector_with_distance(d_cos: float) -> np.ndarray:
            sim = 1.0 - d_cos
            perp = math.sqrt(max(0.0, 1.0 - sim * sim))
            v = np.zeros(512, dtype=np.float32)
            v[0] = sim
            v[1] = perp
            return v / np.linalg.norm(v)

        # 1. d_cos = 0.3499 (< 0.35 -> IDENTITY DUPLICATE)
        e_sub = make_vector_with_distance(0.3499)
        d_sub = dedup.cosine_distance(e1, e_sub)
        assert d_sub < 0.35

        dedup_sub = PerceptualDeduplicator(cosine_threshold=0.35)
        dedup_sub.is_duplicate_and_register(0x1111, e1)
        is_dup, tag = dedup_sub.is_duplicate_and_register(0x9999, e_sub)
        assert is_dup is True
        assert tag == "EMBEDDING_IDENTITY_LEAKAGE"

        # 2. d_cos = 0.3501 (>= 0.35 -> DISTINCT IDENTITY)
        e_sup = make_vector_with_distance(0.3501)
        d_sup = dedup.cosine_distance(e1, e_sup)
        assert d_sup >= 0.35

        dedup_sup = PerceptualDeduplicator(cosine_threshold=0.35)
        dedup_sup.is_duplicate_and_register(0x1111, e1)
        is_dup, tag = dedup_sup.is_duplicate_and_register(0x9999, e_sup)
        assert is_dup is False
        assert tag == "UNIQUE"

    def test_zero_vector_and_extreme_norm_robustness(self):
        """Verify handling of zero vectors and extreme scales without crash or NaN."""
        dedup = PerceptualDeduplicator(cosine_threshold=0.35)
        v_zero = np.zeros(512, dtype=np.float32)
        v_normal = np.random.randn(512).astype(np.float32)
        v_normal /= np.linalg.norm(v_normal)

        # Distance with zero vector defaults safely to 1.0
        d_zero = dedup.cosine_distance(v_zero, v_normal)
        assert d_zero == 1.0
        assert not math.isnan(d_zero)

        # Scaling invariance: distance(100*u, 0.01*v) == distance(u, v)
        v1 = np.random.randn(512).astype(np.float32)
        v2 = np.random.randn(512).astype(np.float32)
        d_orig = dedup.cosine_distance(v1, v2)
        d_scaled = dedup.cosine_distance(v1 * 1000.0, v2 * 0.001)
        assert d_orig == pytest.approx(d_scaled, abs=1e-5)


# ============================================================================
# Probe Suite 3: Label Distribution Learning (LDL) Mathematical Conservation
# ============================================================================

class TestAdversarialLDLMathematicalConservation:
    """
    Empirical stress-testing of LDL mathematical probability conservation:
    Verifies that sum(probs) == 1.0 within 1e-4 for all ages in [0, 100]
    for both Gaussian and Laplacian models across 1,001 continuous age points.
    """

    @pytest.mark.parametrize("distribution_type", ["gaussian", "laplacian"])
    def test_ldl_conservation_dense_grid_0_to_100(self, distribution_type):
        """Verify sum(probs) == 1.0 within 1e-4 for 1001 points in [0, 100]."""
        ages = np.linspace(0.0, 100.0, 1001)
        max_error = 0.0
        min_prob = 1.0

        for age in ages:
            dist = BiologicalLabelingEngine.compute_label_distribution(
                mu_age=float(age),
                sigma_age=3.0,
                distribution_type=distribution_type
            )
            # Check length is 10 cohorts
            assert len(dist) == 10

            # Check conservation: sum equals 1.0 within 1e-4
            s = float(dist.sum())
            err = abs(s - 1.0)
            if err > max_error:
                max_error = err
            assert err <= 1e-4, f"Failed at age {age} ({distribution_type}): sum={s}, err={err}"

            # Check non-negativity
            m = float(np.min(dist))
            if m < min_prob:
                min_prob = m
            assert m >= 0.0, f"Negative probability at age {age} ({distribution_type}): min={m}"

            # Check no NaN or Inf
            assert not np.any(np.isnan(dist))
            assert not np.any(np.isinf(dist))

        assert max_error <= 1e-4
        assert min_prob >= 0.0

    @pytest.mark.parametrize("distribution_type", ["gaussian", "laplacian"])
    @pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5, 2.5, 5.0, 8.0, 15.0])
    def test_ldl_conservation_across_varied_sigmas(self, distribution_type, sigma):
        """Verify conservation under tight and dispersed annotator variance."""
        test_ages = [0.0, 6.0, 12.0, 25.0, 50.0, 65.0, 80.0, 100.0]
        for age in test_ages:
            dist = BiologicalLabelingEngine.compute_label_distribution(
                mu_age=age,
                sigma_age=sigma,
                distribution_type=distribution_type
            )
            assert dist.sum() == pytest.approx(1.0, abs=1e-4)
            assert np.all(dist >= 0.0)

    @pytest.mark.parametrize("boundary_edge", BiologicalLabelingEngine.AGE_EDGES)
    def test_ldl_exact_cohort_edges_conservation(self, boundary_edge):
        """Verify conservation at all 11 exact cohort boundary edges."""
        for dist_type in ["gaussian", "laplacian"]:
            dist = BiologicalLabelingEngine.compute_label_distribution(
                mu_age=float(boundary_edge),
                sigma_age=2.5,
                distribution_type=dist_type
            )
            assert dist.sum() == pytest.approx(1.0, abs=1e-4)
            assert np.all(dist >= 0.0)


# ============================================================================
# Probe Suite 4: Sexual Dimorphism Weight & Aleatoric Uncertainty Trajectory
# ============================================================================

class TestAdversarialSexualDimorphismTrajectory:
    """
    Empirical probe of sexual dimorphism weight w(a) and aleatoric uncertainty sigma(a).
    Verifies:
    - w(0) = 0.25
    - prepubertal ramp to w(12) = 1.0
    - plateau [12, 65] = 1.0
    - geriatric drop w(>65) = 0.85
    - aleatoric uncertainty characteristics.
    """

    def test_dimorphism_exact_milestones(self):
        """Verify exact milestone weights across the lifespan."""
        # Milestone 1: Age 0
        w_0 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(0.0)
        assert w_0 == 0.25

        # Milestone 2: Prepubertal ramp points
        # w(a) = max(0.25, a/12)
        w_3 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(3.0)
        assert w_3 == 0.25
        w_6 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(6.0)
        assert w_6 == pytest.approx(0.50, abs=1e-4)
        w_9 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(9.0)
        assert w_9 == pytest.approx(0.75, abs=1e-4)

        # Milestone 3: Age 12 transitions to 1.0
        w_12 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(12.0)
        assert w_12 == 1.0

        # Milestone 4: Adult plateau [12, 65)
        for age in [12.0, 18.0, 25.0, 40.0, 50.0, 64.99]:
            w = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(age)
            assert w == 1.0

        # Milestone 5: Geriatric drop at age >= 65
        for age in [65.0, 65.01, 70.0, 80.0, 95.0, 100.0]:
            w = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(age)
            assert w == 0.85

    def test_aleatoric_uncertainty_behavior(self):
        """Verify aleatoric uncertainty sigma_aleatoric(a) across lifespan."""
        # Children < 12: high uncertainty
        sig_0 = BiologicalLabelingEngine.calculate_aleatoric_uncertainty(0.0)
        sig_6 = BiologicalLabelingEngine.calculate_aleatoric_uncertainty(6.0)
        sig_12 = BiologicalLabelingEngine.calculate_aleatoric_uncertainty(12.0)

        assert sig_0 == pytest.approx(0.45 * (1.0 ** 1.20) + 0.05, abs=1e-4)  # 0.50
        assert sig_0 > sig_6 > sig_12
        assert sig_12 == 0.05  # minimum baseline

        # Adults [12, 65]: constant minimum baseline 0.05
        for a in [12.0, 20.0, 35.0, 50.0, 65.0]:
            assert BiologicalLabelingEngine.calculate_aleatoric_uncertainty(a) == 0.05

        # Geriatric > 65: increasing aleatoric uncertainty
        sig_66 = BiologicalLabelingEngine.calculate_aleatoric_uncertainty(66.0)
        sig_80 = BiologicalLabelingEngine.calculate_aleatoric_uncertainty(80.0)
        sig_100 = BiologicalLabelingEngine.calculate_aleatoric_uncertainty(100.0)

        assert sig_66 > 0.05
        assert sig_80 > sig_66
        assert sig_100 == pytest.approx(0.45 * (1.0 ** 1.35) + 0.05, abs=1e-4)  # 0.50


# ============================================================================
# Probe Suite 5: Multi-Teacher Disagreement Pruning
# ============================================================================

class TestAdversarialMultiTeacherPruning:
    """
    Empirical probe of Multi-Teacher disagreement pruning:
    - sigma_epistemic > 8.0 correctly prunes samples
    - adult gender confidence < 0.70 correctly prunes samples
    - prepubertal child gender ambiguity is spared from pruning
    """

    def test_age_disagreement_threshold_boundary(self):
        """Verify strict pruning boundary at sigma_epistemic_age = 8.0."""
        mte = MultiTeacherEnsemble(max_epistemic_age_std=8.0)

        # Exactly 8.0 std: [20.0, 36.0] -> mean=28.0, std = sqrt(64) = 8.0
        preds_boundary = [
            {"teacher_name": "t1", "age": 20.0, "gender_prob": 0.95},
            {"teacher_name": "t2", "age": 36.0, "gender_prob": 0.95}
        ]
        res_boundary = mte.evaluate_teachers(preds_boundary)
        assert res_boundary["age_std"] == pytest.approx(8.0, abs=1e-3)
        assert res_boundary["is_pruned"] is False

        # 8.01 std: [19.9, 36.1] -> mean=28.0, diff=16.2, std=8.1 > 8.0
        preds_over = [
            {"teacher_name": "t1", "age": 19.9, "gender_prob": 0.95},
            {"teacher_name": "t2", "age": 36.1, "gender_prob": 0.95}
        ]
        res_over = mte.evaluate_teachers(preds_over)
        assert res_over["age_std"] > 8.0
        assert res_over["is_pruned"] is True
        assert any("AGE_DISAGREEMENT" in tag for tag in res_over["rejection_tags"])

    def test_adult_gender_confidence_boundary(self):
        """Verify strict adult gender confidence boundary at 0.70."""
        mte = MultiTeacherEnsemble(min_gender_confidence=0.70)

        # Confidence = 0.70 exact boundary passes
        preds_70 = [
            {"teacher_name": "t1", "age": 25.0, "gender_prob": 0.70},
            {"teacher_name": "t2", "age": 25.0, "gender_prob": 0.70}
        ]
        res_70 = mte.evaluate_teachers(preds_70)
        assert res_70["gender_confidence"] == pytest.approx(0.70, abs=1e-4)
        assert res_70["is_pruned"] is False

        # Confidence = 0.699 < 0.70 is pruned
        preds_69 = [
            {"teacher_name": "t1", "age": 25.0, "gender_prob": 0.699},
            {"teacher_name": "t2", "age": 25.0, "gender_prob": 0.699}
        ]
        res_69 = mte.evaluate_teachers(preds_69)
        assert res_69["gender_confidence"] < 0.70
        assert res_69["is_pruned"] is True
        assert any("GENDER_CONFUSION" in tag for tag in res_69["rejection_tags"])

    def test_prepubertal_child_gender_ambiguity_spared(self):
        """Verify that prepubertal children (age < 12) with gender conf < 0.70 are NOT pruned."""
        mte = MultiTeacherEnsemble(min_gender_confidence=0.70)
        preds_child = [
            {"teacher_name": "t1", "age": 5.0, "gender_prob": 0.52},
            {"teacher_name": "t2", "age": 5.0, "gender_prob": 0.48}
        ]
        res_child = mte.evaluate_teachers(preds_child)
        assert res_child["consensus_age"] == 5.0
        assert res_child["gender_confidence"] == pytest.approx(0.50, abs=0.01)
        assert res_child["is_pruned"] is False

    def test_high_teacher_count_ensemble_stress(self):
        """Verify consensus and pruning behavior under 10 teachers with noise."""
        mte = MultiTeacherEnsemble(max_epistemic_age_std=8.0, min_gender_confidence=0.70)
        # Consistent teachers
        consistent = [{"teacher_name": f"t_{i}", "age": 30.0 + (i % 3) - 1, "gender_prob": 0.88 + 0.01 * (i % 4)} for i in range(10)]
        res_con = mte.evaluate_teachers(consistent)
        assert res_con["teacher_count"] == 10
        assert res_con["is_pruned"] is False
        assert res_con["alpha_sample_weight"] > 0.0

        # Highly discordant teachers
        discordant = [{"teacher_name": f"t_{i}", "age": 10.0 + 8.0 * i, "gender_prob": 0.50} for i in range(10)]
        res_disc = mte.evaluate_teachers(discordant)
        assert res_disc["age_std"] > 8.0
        assert res_disc["is_pruned"] is True


# ============================================================================
# Probe Suite 6: Deep Edge Cases & Forensic Discrepancy Probes
# ============================================================================

class TestAdversarialDeepEdgeCasesAndForensicProbes:
    """
    Forensic probes verifying edge cases, boundary anomalies, and contract discrepancies.
    """

    def test_negative_age_cohort_index_anomaly(self):
        """
        Forensic Probe: Negative age in get_discrete_group_index.
        Because AGE_EDGES starts at 0.0, any age < 0 falls through the loop and
        returns len(AGE_GROUPS) - 1 (index 9, cohort '65+'), misclassifying negative
        ages as geriatric instead of infant!
        """
        idx_neg = BiologicalLabelingEngine.get_discrete_group_index(-1.0)
        group_neg, _ = BiologicalLabelingEngine.age_to_group(-1.0)
        # Confirms empirical behavior: falls through to '65+'
        assert idx_neg == 9
        assert group_neg == "65+"

    def test_phash_bypass_dead_code_vulnerability(self):
        """
        Forensic Probe: pHash duplicate bypass when embeddings are provided.
        In PerceptualDeduplicator.is_duplicate_and_register:
        Line 628 'if self.cosine_distance(...) >= self.cosine_threshold: continue'
        causes any candidate with dH <= 6 to bypass pHash duplicate detection if
        its embedding has d_cos >= 0.35.
        Consequently, two images with dH = 0 (identical pHash) are both registered as UNIQUE.
        """
        dedup = PerceptualDeduplicator(hamming_threshold=6, cosine_threshold=0.35)
        h_common = 0xA1B2C3D4E5F60718
        e1 = np.zeros(512, dtype=np.float32); e1[0] = 1.0
        e2 = np.zeros(512, dtype=np.float32); e2[1] = 1.0  # orthogonal: d_cos = 1.0 >= 0.35

        # Register image 1
        is_dup_1, tag_1 = dedup.is_duplicate_and_register(h_common, e1)
        assert is_dup_1 is False
        assert tag_1 == "UNIQUE"

        # Image 2 has identical pHash (dH = 0)
        is_dup_2, tag_2 = dedup.is_duplicate_and_register(h_common, e2)
        # Confirms empirical behavior: bypassed because d_cos >= 0.35!
        assert is_dup_2 is False
        assert tag_2 == "UNIQUE"

    def test_age_65_boundary_condition_asymmetry(self):
        """
        Forensic Probe: Boundary condition asymmetry at exact age 65.0.
        calculate_biological_sexual_dimorphism_weight uses 'elif age >= 65.0: return 0.85'
        while calculate_aleatoric_uncertainty uses 'elif age > 65.0: ...'
        At exact age 65.0:
        dimorphism weight has already dropped to 0.85,
        while aleatoric uncertainty remains at 0.05 (adult plateau baseline).
        """
        w_65 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(65.0)
        sig_65 = BiologicalLabelingEngine.calculate_aleatoric_uncertainty(65.0)
        assert w_65 == 0.85
        assert sig_65 == 0.05

    def test_mte_ble_child_pruning_contract_divergence(self):
        """
        Forensic Probe: Pruning divergence between MultiTeacherEnsemble and BiologicalLabelingEngine.
        For age < 18 with epistemic std in (5.5, 8.0]:
        MultiTeacherEnsemble.evaluate_teachers triggers CHILD_AGE_DISAGREEMENT (is_pruned=True)
        BiologicalLabelingEngine.extract_alpha_signals only checks std > 8.0 (is_pruned=False).
        """
        preds = [
            {"teacher_name": "t1", "age": 10.0, "gender_prob": 0.95},
            {"teacher_name": "t2", "age": 22.0, "gender_prob": 0.95}
        ]
        mte = MultiTeacherEnsemble()
        res_mte = mte.evaluate_teachers(preds)
        res_ble = BiologicalLabelingEngine.extract_alpha_signals(preds)

        assert res_mte["consensus_age"] == 16.0
        assert res_mte["age_std"] == pytest.approx(6.0, abs=0.01)
        # MTE prunes because mu_age < 18 and sigma > 5.5
        assert res_mte["is_pruned"] is True
        # BLE does not prune because sigma <= 8.0
        assert res_ble["is_pruned"] is False

    def test_prepubertal_ramp_piecewise_flatness(self):
        """
        Forensic Probe: Prepubertal weight flatness in [0, 3].
        calculate_biological_sexual_dimorphism_weight clamps ratio at max(0.25, age / 12.0).
        For all ages in [0, 3], w(a) is constant at 0.25, and only starts ramping at age > 3.0.
        """
        w_0 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(0.0)
        w_1 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(1.0)
        w_2 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(2.0)
        w_3 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(3.0)
        w_4 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(4.0)

        assert w_0 == w_1 == w_2 == w_3 == 0.25
        assert w_4 > 0.25
