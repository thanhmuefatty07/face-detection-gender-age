"""
Abstract Face Detector Interface
=================================
Defines the unified contract for face detection and 5-point landmark extraction.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np

from ..core.types import FaceDetection


class FaceDetectorBase(ABC):
    """Abstract Base Class for all face detection backends."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detects faces in BGR/RGB image and extracts 5-point facial landmarks.

        Args:
            image: Input image (H, W, 3) as numpy array.

        Returns:
            List of FaceDetection instances.
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Returns unique identifier for detector backend."""
        pass
