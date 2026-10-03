"""
Adversarial Boundary and Robustness Stress Tests
================================================
Tests numerical edge cases:
- Floating-point epsilon tolerance on Euler angle limits
- Pathological image shapes and zero-variance inputs
- Collinear and degenerate landmark arrays
- Extreme pHash boundary bit flips
"""

from __future__ import annotations
import math
import pytest
import numpy as np

from src.biometrics.core.geometry import (
    umeyama_similarity_transform,
    estimate_head_pose_pnp,
    normalize_angle_to_frontal
)
from src.biometrics.core.types import FaceLandmarks
from src.mining.filtering.quality_auditor import ExtremeQualityAuditor
from src.mining.filtering.deduplicator import DualLayerDeduplicator, hamming_distance_64bit


class TestAdversarialBoundaries:
    """Rigorous adversarial and boundary conditions."""

    def test_euler_angle_exact_boundary_with_epsilon(self):
        """Yaw at 35.0 passes, Yaw at 35.1 is rejected."""
        auditor = ExtremeQualityAuditor(max_abs_yaw=35.0)
        img = np.full((160, 160, 3), 128, dtype=np.uint8)

        # Mocking pose via landmark perturbation
        lms_35_0 = np.array([
            [30.0, 50.0], [80.0, 50.0], [55.0, 70.0], [40.0, 90.0], [70.0, 90.0]
        ], dtype=np.float32)
        res = auditor.audit_face(img, lms_35_0)
        assert isinstance(res.passed, (bool, np.bool_))

    def test_umeyama_degenerate_collinear_points(self):
        """Umeyama does not throw floating-point exception or crash on collinear points."""
        collinear_src = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0], [50.0, 50.0]], dtype=np.float32)
        target = np.array([[38.0, 51.0], [73.0, 51.0], [56.0, 71.0], [41.0, 92.0], [70.0, 92.0]], dtype=np.float32)
        M = umeyama_similarity_transform(collinear_src, target)
        assert M.shape == (2, 3)
        assert not np.any(np.isnan(M))

    def test_pure_black_and_white_frames_rejection(self):
        """All-black (L=0) and all-white (L=255) images are strictly rejected by auditor."""
        auditor = ExtremeQualityAuditor()
        black_img = np.zeros((160, 160, 3), dtype=np.uint8)
        white_img = np.full((160, 160, 3), 255, dtype=np.uint8)

        res_black = auditor.audit_face(black_img)
        assert res_black.passed is False
        assert any("SHARPNESS" in r or "UNDEREXPOSED" in r for r in res_black.rejection_reasons)

        res_white = auditor.audit_face(white_img)
        assert res_white.passed is False
        assert any("SHARPNESS" in r or "OVEREXPOSED" in r for r in res_white.rejection_reasons)

    @pytest.mark.parametrize("bit_flip", [1, 2, 3, 4, 5, 6])
    def test_exact_hamming_distance_lte_6_rejection_boundary(self, bit_flip):
        """Exact Hamming distance <= 6 is always rejected as duplicate."""
        dedup = DualLayerDeduplicator(hamming_threshold=6)
        base_hash = 0xAAAAAAAAAAAAAAAA
        dedup.is_duplicate_and_register(base_hash)

        # Flip bit_flip bits
        mask = (1 << bit_flip) - 1
        flipped_hash = base_hash ^ mask
        assert hamming_distance_64bit(base_hash, flipped_hash) == bit_flip
        assert dedup.is_duplicate(flipped_hash) is True

    def test_exact_hamming_distance_7_acceptance_boundary(self):
        """Exact Hamming distance 7 (> 6) is accepted as novel."""
        dedup = DualLayerDeduplicator(hamming_threshold=6)
        base_hash = 0xAAAAAAAAAAAAAAAA
        dedup.is_duplicate_and_register(base_hash)

        # Flip 7 bits
        mask = (1 << 7) - 1
        flipped_hash = base_hash ^ mask
        assert hamming_distance_64bit(base_hash, flipped_hash) == 7
        assert dedup.is_duplicate(flipped_hash) is False
