"""
Core biometric data structures and geometric alignment tools.
"""

from .types import BoundingBox, FaceLandmarks, FaceDetection, BiometricPrediction, FaceAuditResult
from .geometry import (
    UmeyamaAligner,
    ARCFACE_CANONICAL_112,
    ARCFACE_CANONICAL_160,
    estimate_head_pose_pnp,
    normalize_angle_to_frontal,
)

__all__ = [
    "BoundingBox",
    "FaceLandmarks",
    "FaceDetection",
    "BiometricPrediction",
    "FaceAuditResult",
    "UmeyamaAligner",
    "ARCFACE_CANONICAL_112",
    "ARCFACE_CANONICAL_160",
    "estimate_head_pose_pnp",
    "normalize_angle_to_frontal",
]
