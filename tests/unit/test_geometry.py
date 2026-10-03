"""
Unit Tests for Geometric Alignment & Anthropometry
==================================================
Tests:
- Umeyama SVD similarity transform mathematical closed-form invariance
- solvePnP Euler angles extraction & frontal angle normalization
"""

from __future__ import annotations
import math
import pytest
import numpy as np

from src.biometrics.core.geometry import (
    UmeyamaAligner,
    umeyama_similarity_transform,
    normalize_angle_to_frontal,
    estimate_head_pose_pnp,
    ARCFACE_CANONICAL_160,
    ARCFACE_CANONICAL_112,
)
from src.biometrics.core.types import FaceLandmarks


class TestUmeyamaSimilarityTransform:
    """Verifies Umeyama 5-point alignment numerical precision and transformation invariance."""

    def test_identity_alignment_preserves_canonical_points(self):
        """When source points equal canonical points, transform should be identity."""
        src = ARCFACE_CANONICAL_160.copy()
        dst = ARCFACE_CANONICAL_160.copy()
        M = umeyama_similarity_transform(src, dst)
        assert M.shape == (2, 3)
        # Scale = 1.0, Rotation = 0, Translation = 0
        assert np.allclose(M[:2, :2], np.eye(2), atol=1e-4)
        assert np.allclose(M[:2, 2], np.zeros(2), atol=1e-4)

    def test_pure_translation_invariance(self):
        """Umeyama correctly recovers pure 2D translation without scaling or rotation."""
        src = ARCFACE_CANONICAL_160.copy()
        tx, ty = 25.0, -18.0
        src_shifted = src + np.array([tx, ty], dtype=np.float32)

        M = umeyama_similarity_transform(src_shifted, src)
        # M[:2, 2] should cancel out translation
        warped_pts = (M[:2, :2] @ src_shifted.T).T + M[:2, 2]
        assert np.allclose(warped_pts, src, atol=1e-4)

    def test_pure_rotation_invariance(self):
        """Umeyama correctly recovers rotation around center."""
        src = ARCFACE_CANONICAL_160.copy()
        theta = math.radians(30.0)
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)

        center = np.mean(src, axis=0)
        src_rotated = ((R @ (src - center).T).T) + center

        M = umeyama_similarity_transform(src_rotated, src)
        warped_pts = (M[:2, :2] @ src_rotated.T).T + M[:2, 2]
        assert np.allclose(warped_pts, src, atol=1e-4)

    def test_aligner_warps_image_to_target_dimension(self, synthetic_sharp_face):
        """UmeyamaAligner produces exact (160, 160, 3) image shape."""
        aligner = UmeyamaAligner(target_size=160)
        lms = FaceLandmarks(points=ARCFACE_CANONICAL_160.copy())
        aligned = aligner.align(synthetic_sharp_face, lms)
        assert aligned.shape == (160, 160, 3)


class TestAnthropometricHeadPoseEstimation:
    """Verifies solvePnP Euler angles and frontal normalization."""

    def test_frontal_angle_normalization(self):
        """Angle normalization brings angles into [-90, 90]."""
        assert normalize_angle_to_frontal(0.0) == 0.0
        assert normalize_angle_to_frontal(30.0) == 30.0
        assert normalize_angle_to_frontal(-25.0) == -25.0
        assert normalize_angle_to_frontal(179.0) == -1.0
        assert normalize_angle_to_frontal(-178.0) == 2.0

    def test_canonical_face_produces_low_euler_angles(self):
        """Canonical ArcFace points should yield near-zero frontal Euler angles."""
        yaw, pitch, roll = estimate_head_pose_pnp(ARCFACE_CANONICAL_160, 160, 160)
        assert abs(yaw) <= 25.0
        assert abs(pitch) <= 25.0
        assert abs(roll) <= 20.0
