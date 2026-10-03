"""
Unit Tests for Extreme Quality Auditor & Deduplication
======================================================
Tests:
- ExtremeQualityAuditor gates: Laplacian, Tenengrad, Euler Pose, CIE L*a*b*
- DualLayerDeduplicator: 64-bit DCT pHash Hamming distance and Cosine distance
"""

from __future__ import annotations
import pytest
import numpy as np

from src.mining.filtering.quality_auditor import ExtremeQualityAuditor
from src.mining.filtering.deduplicator import (
    DualLayerDeduplicator,
    compute_64bit_dct_phash,
    hamming_distance_64bit
)
from src.biometrics.core.geometry import ARCFACE_CANONICAL_160


class TestExtremeQualityAuditor:
    """Verifies strict adherence to extreme quality gates."""

    def test_sharp_face_passes_audit(self, synthetic_sharp_face):
        """Synthetic sharp face with valid landmarks passes all quality gates."""
        auditor = ExtremeQualityAuditor()
        result = auditor.audit_face(synthetic_sharp_face, ARCFACE_CANONICAL_160)
        assert result.passed is True
        assert result.laplacian_variance >= 100.0
        assert result.tenengrad_energy >= 250.0

    def test_blurry_face_fails_sharpness_gate(self, synthetic_blurry_face):
        """Blurry face is rejected due to low Laplacian variance."""
        auditor = ExtremeQualityAuditor()
        result = auditor.audit_face(synthetic_blurry_face, ARCFACE_CANONICAL_160)
        assert result.passed is False
        assert any("INSUFFICIENT_SHARPNESS" in r for r in result.rejection_reasons)

    def test_dark_underexposed_face_fails_photometry(self, synthetic_sharp_face):
        """Image with luminance L < 35 is rejected."""
        auditor = ExtremeQualityAuditor()
        dark_img = (synthetic_sharp_face.astype(np.float32) * 0.15).astype(np.uint8)
        result = auditor.audit_face(dark_img, ARCFACE_CANONICAL_160)
        assert result.passed is False
        assert any("UNDEREXPOSED" in r for r in result.rejection_reasons)


class TestDualLayerDeduplicator:
    """Verifies pHash Hamming and Cosine distance deduplication."""

    def test_exact_hash_identified_as_duplicate(self):
        """Identical hash is rejected as duplicate."""
        dedup = DualLayerDeduplicator(hamming_threshold=6)
        h = 0x123456789ABCDEF0
        assert dedup.is_duplicate_and_register(h) is False  # First is novel
        assert dedup.is_duplicate_and_register(h) is True   # Second is duplicate

    def test_near_duplicate_hamming_lte_6_rejected(self):
        """Hashes with Hamming distance <= 6 are rejected."""
        dedup = DualLayerDeduplicator(hamming_threshold=6)
        base = 0xFFFFFFFFFFFFFFFF
        dedup.is_duplicate_and_register(base)

        # 3 bit flips -> distance 3 <= 6 -> duplicate
        flip_3 = base ^ 0b111
        assert dedup.is_duplicate(flip_3) is True

    def test_distinct_sample_hamming_gt_6_accepted(self):
        """Hashes with Hamming distance > 6 are accepted."""
        dedup = DualLayerDeduplicator(hamming_threshold=6)
        base = 0xFFFFFFFFFFFFFFFF
        dedup.is_duplicate_and_register(base)

        # 10 bit flips -> distance 10 > 6 -> accepted
        flip_10 = base ^ 0b1111111111
        assert dedup.is_duplicate(flip_10) is False
