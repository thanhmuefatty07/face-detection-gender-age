"""
Biometric Data Structures and Types
==================================
Defines immutable, strongly-typed representations for facial bounding boxes,
landmarks, audit results, and multi-task biometric predictions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Represents a facial bounding box in pixel coordinates."""
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0

    @property
    def x1(self) -> float:
        return self.x

    @property
    def y1(self) -> float:
        return self.y

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def aspect_ratio(self) -> float:
        return self.height / max(self.width, 1e-6)

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.width, self.height)


@dataclass(frozen=True)
class FaceLandmarks:
    """
    Standard 5-point facial landmarks:
    Order: [Left Eye, Right Eye, Nose Tip, Left Mouth Corner, Right Mouth Corner]
    """
    points: np.ndarray  # Shape: (5, 2), dtype: float32

    def __post_init__(self):
        if not isinstance(self.points, np.ndarray) or self.points.shape != (5, 2):
            raise ValueError(f"FaceLandmarks points must be a numpy array of shape (5, 2), got {getattr(self.points, 'shape', None)}")

    @property
    def left_eye(self) -> Tuple[float, float]:
        return (float(self.points[0, 0]), float(self.points[0, 1]))

    @property
    def right_eye(self) -> Tuple[float, float]:
        return (float(self.points[1, 0]), float(self.points[1, 1]))

    @property
    def nose_tip(self) -> Tuple[float, float]:
        return (float(self.points[2, 0]), float(self.points[2, 1]))

    @property
    def left_mouth(self) -> Tuple[float, float]:
        return (float(self.points[3, 0]), float(self.points[3, 1]))

    @property
    def right_mouth(self) -> Tuple[float, float]:
        return (float(self.points[4, 0]), float(self.points[4, 1]))

    @property
    def interocular_distance(self) -> float:
        """Euclidean distance between left and right eye centers."""
        return float(np.linalg.norm(self.points[0] - self.points[1]))


@dataclass
class FaceDetection:
    """Represents a detected face instance with bounding box and landmarks."""
    bbox: BoundingBox
    landmarks: Optional[FaceLandmarks] = None
    aligned_crop: Optional[np.ndarray] = None  # Aligned 160x160 RGB/BGR crop
    detector_name: str = "unknown"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FaceAuditResult:
    """Extreme Quality Audit metrics for a face candidate."""
    passed: bool
    rejection_reasons: List[str] = field(default_factory=list)
    laplacian_variance: float = 0.0
    tenengrad_energy: float = 0.0
    mean_luminance: float = 0.0
    shadow_clip_fraction: float = 0.0
    highlight_clip_fraction: float = 0.0
    cheek_occlusion_delta_e: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass
class BiometricPrediction:
    """Complete multi-task biometric inference result."""
    # Age estimation
    expected_age: float
    age_ldl_distribution: np.ndarray  # 101-bin probability simplex
    age_cohort_name: str
    age_cohort_index: int
    apparent_age_std: float

    # Gender estimation
    gender: str                        # "Female" or "Male"
    female_probability: float
    dimorphism_weight: float           # w_dimorphism in [0.25, 1.0]

    # Model confidence & epistemic signals
    aleatoric_uncertainty: float
    alpha_sample_weight: float
    quality_score: float

    # Meta
    aligned_face_phash: Optional[int] = None
    embedding_512d: Optional[np.ndarray] = None
