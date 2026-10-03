"""
Integration Tests for Unified Biometric Pipeline
================================================
Verifies end-to-end lifecycle:
Detection -> Umeyama 160x160 Alignment -> Quality Audit -> Biometric Estimation.
"""

from __future__ import annotations
import pytest
import numpy as np

from src.biometrics.pipeline import BiometricPipeline


class TestBiometricPipelineIntegration:
    """Verifies complete end-to-end processing pipeline."""

    def test_pipeline_processes_sharp_face_successfully(self, synthetic_sharp_face):
        """Pipeline detects, aligns, audits, and estimates biometrics for a sharp face."""
        pipeline = BiometricPipeline(detector_engine="yunet", enable_quality_audit=True)
        results = pipeline.process_frame(synthetic_sharp_face)

        assert len(results) >= 1
        det, audit, pred = results[0]

        # Detection assertions
        assert det.landmarks is not None
        assert det.aligned_crop is not None
        assert det.aligned_crop.shape == (160, 160, 3)

        # Audit assertions
        assert audit is not None
        assert audit.passed is True

        # Biometric prediction assertions
        assert pred is not None
        assert 0.0 <= pred.expected_age <= 100.0
        assert pred.gender in ["Female", "Male"]
        assert 0.0 <= pred.female_probability <= 1.0
        assert 0.25 <= pred.dimorphism_weight <= 1.0
        assert len(pred.age_ldl_distribution) == 101
        assert np.isclose(np.sum(pred.age_ldl_distribution), 1.0, atol=1e-4)

    def test_pipeline_filters_out_blurry_face_when_auditing_enabled(self, synthetic_blurry_face):
        """Blurry face is rejected during audit, skipping downstream estimation."""
        pipeline = BiometricPipeline(detector_engine="yunet", enable_quality_audit=True)
        results = pipeline.process_frame(synthetic_blurry_face)

        assert len(results) >= 1
        det, audit, pred = results[0]
        assert audit is not None
        assert audit.passed is False
        assert pred is None  # Skipped because audit failed
