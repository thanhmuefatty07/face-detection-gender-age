"""
Pytest Test Fixtures and Synthetic Biometric Data Generators
============================================================
Provides standardized fixtures for unit, integration, and adversarial testing.
"""

from __future__ import annotations
import math
import pytest
import numpy as np
import cv2

from src.biometrics.core.types import BoundingBox, FaceLandmarks, FaceDetection


@pytest.fixture
def canonical_arcface_landmarks():
    """Returns canonical ArcFace 160x160 5-point landmarks."""
    from src.biometrics.core.geometry import ARCFACE_CANONICAL_160
    return ARCFACE_CANONICAL_160.copy()


@pytest.fixture
def synthetic_sharp_face():
    """
    Generates a synthetic sharp face crop (160, 160, 3)
    with high-frequency gradient features ensuring Laplacian >= 100.
    """
    img = np.full((160, 160, 3), 130, dtype=np.uint8)
    # Add high-frequency facial texture (eyes, eyebrows, mouth, nose)
    cv2.circle(img, (55, 65), 10, (20, 20, 20), -1)  # Left pupil
    cv2.circle(img, (105, 65), 10, (20, 20, 20), -1) # Right pupil
    cv2.ellipse(img, (80, 120), (22, 6), 0, 0, 360, (50, 40, 180), -1) # Lips
    cv2.line(img, (80, 80), (80, 100), (40, 40, 40), 2) # Nose bridge

    # High-frequency texture noise
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 18, (160, 160, 3)).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_img


@pytest.fixture
def synthetic_blurry_face():
    """Generates a severely blurred face failing Laplacian sharpness gate."""
    img = np.full((160, 160, 3), 128, dtype=np.uint8)
    blurred = cv2.GaussianBlur(img, (31, 31), 10.0)
    return blurred
