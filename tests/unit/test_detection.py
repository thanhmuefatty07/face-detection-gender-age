"""
Unit Tests for Dual-Engine Face Detectors
=========================================
Tests:
- YuNet native detector initialization and morphological feature extraction
- SCRFD detector interface compliance
- Extraction of valid 5-point facial landmarks and bounding box
"""

from __future__ import annotations
import pytest
import numpy as np

from src.biometrics.detection.yunet_detector import YuNetFaceDetector
from src.biometrics.detection.scrfd_detector import SCRFDFaceDetector
from src.biometrics.core.types import FaceDetection, FaceLandmarks


class TestFaceDetectorEngines:
    """Tests face detection engines on synthetic images."""

    def test_yunet_detector_extracts_detection_and_landmarks(self, synthetic_sharp_face):
        """YuNet detector returns valid FaceDetection with 5 landmarks."""
        detector = YuNetFaceDetector(confidence_threshold=0.5)
        detections = detector.detect(synthetic_sharp_face)
        assert len(detections) >= 1
        det = detections[0]
        assert det.landmarks is not None
        assert det.landmarks.points.shape == (5, 2)
        assert det.bbox.width > 0 and det.bbox.height > 0

    def test_scrfd_detector_interface_compliance(self, synthetic_sharp_face):
        """SCRFD detector conforms to FaceDetectorBase and returns valid format."""
        detector = SCRFDFaceDetector(confidence_threshold=0.5)
        assert detector.get_backend_name() == "scrfd_insightface"
        detections = detector.detect(synthetic_sharp_face)
        assert len(detections) >= 1
        det = detections[0]
        assert det.landmarks is not None
        assert det.landmarks.points.shape == (5, 2)

    def test_detector_handles_empty_image_gracefully(self):
        """Detectors return empty list without crashing on empty inputs."""
        detector = YuNetFaceDetector()
        assert detector.detect(np.array([])) == []
        assert detector.detect(None) == []
        empty_img = np.zeros((10, 10, 3), dtype=np.uint8)
        assert detector.detect(empty_img) == []
