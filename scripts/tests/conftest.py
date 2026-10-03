"""
E2E Test Infrastructure - Fixtures, Oracles, and Synthetic Generators
======================================================================
Defines authoritative mathematical models, synthetic face image generators,
and shared pytest fixtures for Tiers 1-4 biometric test suites.
"""

from __future__ import annotations
import math
import sys
import importlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
import pytest
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# AUTHORITATIVE MATHEMATICAL ORACLES (From ORIGINAL_REQUEST.md & survey_specs)
# ============================================================================

class ReferenceBiometricOracles:
    """Authoritative mathematical reference models derived from project specifications."""

    AGE_GROUPS = ['0-2', '3-6', '7-12', '13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    AGE_EDGES = [0.0, 2.5, 6.5, 12.5, 17.5, 24.5, 34.5, 44.5, 54.5, 64.5, 100.0]

    @classmethod
    def age_to_group_index(cls, age: float) -> int:
        for i in range(len(cls.AGE_EDGES) - 1):
            if cls.AGE_EDGES[i] <= age < cls.AGE_EDGES[i + 1]:
                return i
        return len(cls.AGE_GROUPS) - 1

    @classmethod
    def dimorphism_weight(cls, age: float) -> float:
        """w_dimorphism(a) bounded in [0.25, 1.0]."""
        if age < 12.0:
            return max(0.25, float(age / 12.0))
        elif age > 65.0:
            return 0.85
        return 1.0

    @classmethod
    def aleatoric_uncertainty(
        cls,
        age: float,
        sigma_0: float = 0.05,
        sigma_max: float = 0.45,
        gamma_1: float = 1.20,
        gamma_2: float = 1.35
    ) -> float:
        """Heteroscedastic aleatoric uncertainty curve across lifespan."""
        if age < 12.0:
            return float(sigma_max * ((1.0 - age / 12.0) ** gamma_1) + sigma_0)
        elif age > 65.0:
            frac = min(1.0, (age - 65.0) / 35.0)
            return float(sigma_max * (frac ** gamma_2) + sigma_0)
        return float(sigma_0)

    @classmethod
    def east_asian_apparent_age(cls, age: float, delta_youth: float = 4.5, kappa_cliff: float = 1.30) -> float:
        """Non-linear East Asian aging kinetics (youthful plateau -> cliff at 48)."""
        if age < 48.0:
            return float(age - delta_youth * (age / 48.0))
        return float((48.0 - delta_youth) + kappa_cliff * ((age - 48.0) ** 1.35))

    @classmethod
    def caucasian_photoaging_rate(cls, age: float, k_c: float = 1.0) -> float:
        """Linear-to-early rhytid aging kinetics for Caucasian skin."""
        return float(k_c * age)

    @classmethod
    def gaussian_ldl(cls, mu_age: float, sigma_age: float = 3.0) -> np.ndarray:
        """Gaussian CDF integration over 10 age cohorts using math.erf."""
        probs = np.zeros(len(cls.AGE_GROUPS), dtype=np.float64)
        for i in range(len(cls.AGE_GROUPS)):
            low = cls.AGE_EDGES[i]
            high = cls.AGE_EDGES[i + 1]
            z_low = (low - mu_age) / (sigma_age * math.sqrt(2))
            z_high = (high - mu_age) / (sigma_age * math.sqrt(2))
            p = 0.5 * (math.erf(z_high) - math.erf(z_low))
            probs[i] = max(0.0, p)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs[cls.age_to_group_index(mu_age)] = 1.0
        return probs

    @classmethod
    def laplacian_ldl(cls, mu_age: float, b_scale: float = 2.1213) -> np.ndarray:
        """Laplacian CDF integration over 10 age cohorts."""
        probs = np.zeros(len(cls.AGE_GROUPS), dtype=np.float64)
        def cdf(x: float) -> float:
            if x <= mu_age:
                return 0.5 * math.exp((x - mu_age) / b_scale)
            else:
                return 1.0 - 0.5 * math.exp(-(x - mu_age) / b_scale)

        for i in range(len(cls.AGE_GROUPS)):
            low = cls.AGE_EDGES[i]
            high = cls.AGE_EDGES[i + 1]
            p = cdf(high) - cdf(low)
            probs[i] = max(0.0, p)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs[cls.age_to_group_index(mu_age)] = 1.0
        return probs

    @classmethod
    def shannon_entropy(cls, prob: float) -> float:
        """Shannon entropy H(p) = -p log2(p) - (1-p) log2(1-p)."""
        p = max(1e-7, min(1.0 - 1e-7, prob))
        return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))

    @classmethod
    def emd_distance(cls, p: np.ndarray, q: np.ndarray) -> float:
        """1D Earth Mover's Distance / Wasserstein-1 between discrete CDFs."""
        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)
        return float(np.mean(np.abs(cdf_p - cdf_q)))

    @classmethod
    def kl_divergence(cls, p: np.ndarray, q: np.ndarray, eps: float = 1e-7) -> float:
        """Kullback-Leibler divergence D_KL(P || Q)."""
        p_safe = np.clip(p, eps, 1.0)
        q_safe = np.clip(q, eps, 1.0)
        return float(np.sum(p_safe * np.log(p_safe / q_safe)))

    @classmethod
    def hamming_distance(cls, h1: int, h2: int) -> int:
        return bin(h1 ^ h2).count('1')

    @classmethod
    def cosine_distance(cls, v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-9 or norm2 < 1e-9:
            return 1.0
        return float(1.0 - np.dot(v1, v2) / (norm1 * norm2))


# ============================================================================
# SYNTHETIC FACE IMAGE AND GEOMETRY GENERATORS
# ============================================================================

def create_synthetic_face(
    size: int = 160,
    blur_ksize: int = 0,
    luminance_offset: int = 0,
    noise_sigma: float = 0.0,
    clip_extremes: bool = False
) -> np.ndarray:
    """
    Synthesizes a realistic test face image containing crisp facial features
    (head oval, eyes with pupils, eyebrows, nose bridge, mouth with lips).
    Generates genuine high Laplacian variance and Tenengrad energy when not blurred.
    """
    img = np.full((size, size, 3), 128, dtype=np.uint8)

    # 1. Skin tone base oval
    skin_color = (160, 190, 220)  # BGR skin tone
    center = (size // 2, size // 2)
    axes = (int(size * 0.35), int(size * 0.45))
    cv2.ellipse(img, center, axes, 0, 0, 360, skin_color, -1)

    # 2. Forehead shadow & hair contour (adds natural high-frequency texture)
    cv2.ellipse(img, (center[0], int(size * 0.25)), (int(size * 0.36), int(size * 0.2)), 0, 180, 360, (40, 30, 20), -1)

    # 3. Left Eye (Sclera + dark iris + bright pupil reflection)
    le_x, le_y = int(size * 0.35), int(size * 0.45)
    cv2.ellipse(img, (le_x, le_y), (int(size * 0.08), int(size * 0.04)), 0, 0, 360, (240, 240, 240), -1)
    cv2.circle(img, (le_x, le_y), int(size * 0.035), (20, 20, 20), -1)
    cv2.circle(img, (le_x - 1, le_y - 1), int(size * 0.01), (255, 255, 255), -1)

    # Eyebrow
    cv2.line(img, (le_x - 15, le_y - 12), (le_x + 15, le_y - 10), (30, 20, 15), 3)

    # 4. Right Eye
    re_x, re_y = int(size * 0.65), int(size * 0.45)
    cv2.ellipse(img, (re_x, re_y), (int(size * 0.08), int(size * 0.04)), 0, 0, 360, (240, 240, 240), -1)
    cv2.circle(img, (re_x, re_y), int(size * 0.035), (20, 20, 20), -1)
    cv2.circle(img, (re_x - 1, re_y - 1), int(size * 0.01), (255, 255, 255), -1)

    # Eyebrow
    cv2.line(img, (re_x - 15, re_y - 10), (re_x + 15, re_y - 12), (30, 20, 15), 3)

    # 5. Nose tip & nostrils
    nose_x, nose_y = size // 2, int(size * 0.62)
    cv2.circle(img, (nose_x, nose_y), int(size * 0.04), (130, 160, 190), -1)
    cv2.circle(img, (nose_x - 5, nose_y + 3), 3, (60, 70, 80), -1)
    cv2.circle(img, (nose_x + 5, nose_y + 3), 3, (60, 70, 80), -1)

    # 6. Mouth / Lips
    mouth_x, mouth_y = size // 2, int(size * 0.80)
    cv2.ellipse(img, (mouth_x, mouth_y), (int(size * 0.16), int(size * 0.05)), 0, 0, 360, (90, 90, 180), -1)
    cv2.line(img, (mouth_x - int(size * 0.14), mouth_y), (mouth_x + int(size * 0.14), mouth_y), (50, 40, 80), 2)

    # Apply luminance adjustment
    if luminance_offset != 0:
        img = np.clip(img.astype(np.int16) + luminance_offset, 0, 255).astype(np.uint8)

    # Apply Gaussian noise
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Apply extreme sensor clipping if requested
    if clip_extremes:
        # Blow out top 10% and crush bottom 10%
        img[0:int(size * 0.15), :] = 255
        img[-int(size * 0.15):, :] = 0

    # Apply Gaussian blur if requested
    if blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        # Moderate contrast to realistically simulate soft focus
        img = np.clip((img.astype(np.float32) - 128.0) * 0.35 + 128.0, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    return img


def get_canonical_landmarks(size: int = 160) -> np.ndarray:
    """Standard ArcFace 160x160 canonical landmarks."""
    ref_112 = np.array([
        [38.2946, 51.6963],  # Left eye
        [73.5318, 51.5014],  # Right eye
        [56.0252, 71.7366],  # Nose tip
        [41.5493, 92.3655],  # Left mouth
        [70.7299, 92.2041]   # Right mouth
    ], dtype=np.float32)
    return ref_112 * (size / 112.0)


# ============================================================================
# DYNAMIC IMPORTER HELPER
# ============================================================================

def resolve_feature(module_rel_path: str, item_name: str) -> Optional[Any]:
    """
    Attempts to import an item from scripts.kaggle_pipeline.
    Returns the resolved class/function, or None if not yet implemented by Worker.
    """
    try:
        mod = importlib.import_module(f"scripts.kaggle_pipeline.{module_rel_path}")
        return getattr(mod, item_name, None)
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None


# ============================================================================
# SHARED PYTEST FIXTURES
# ============================================================================

@pytest.fixture
def oracles() -> type[ReferenceBiometricOracles]:
    """Provides access to reference mathematical oracles."""
    return ReferenceBiometricOracles


@pytest.fixture
def sharp_face() -> np.ndarray:
    """Fixture producing a crisp, sharp synthetic face image (Laplacian > 150, Tenengrad > 300)."""
    return create_synthetic_face(size=160, blur_ksize=0, luminance_offset=0)


@pytest.fixture
def blurry_face() -> np.ndarray:
    """Fixture producing a severely blurred face (Laplacian < 50, Tenengrad < 100)."""
    return create_synthetic_face(size=160, blur_ksize=27, luminance_offset=0)


@pytest.fixture
def underexposed_face() -> np.ndarray:
    """Fixture producing an underexposed dark face (mean L* < 30)."""
    return create_synthetic_face(size=160, blur_ksize=0, luminance_offset=-135)


@pytest.fixture
def overexposed_face() -> np.ndarray:
    """Fixture producing an overexposed blown-out face (mean L* > 225)."""
    return create_synthetic_face(size=160, blur_ksize=0, luminance_offset=110)


@pytest.fixture
def clipped_face() -> np.ndarray:
    """Fixture producing a face with sensor clipping fraction > 15%."""
    return create_synthetic_face(size=160, clip_extremes=True)


@pytest.fixture
def canonical_landmarks() -> np.ndarray:
    """Canonical ArcFace 5-point landmark coordinates for 160x160 face."""
    return get_canonical_landmarks(size=160)


@pytest.fixture
def golden_teacher_predictions() -> List[Dict[str, Any]]:
    """Ensemble predictions with strong consensus (no disagreement)."""
    return [
        {"teacher": "insightface", "age": 28.0, "gender_prob": 0.96},
        {"teacher": "mivolo", "age": 27.5, "gender_prob": 0.94},
        {"teacher": "fairface", "age": 28.5, "gender_prob": 0.95},
        {"teacher": "clip", "age": 28.0, "gender_prob": 0.92}
    ]


@pytest.fixture
def polarized_teacher_predictions() -> List[Dict[str, Any]]:
    """Ensemble predictions with severe age disagreement (sigma > 8.0)."""
    return [
        {"teacher": "insightface", "age": 20.0, "gender_prob": 0.95},
        {"teacher": "mivolo", "age": 42.0, "gender_prob": 0.85},
        {"teacher": "fairface", "age": 55.0, "gender_prob": 0.90},
        {"teacher": "clip", "age": 30.0, "gender_prob": 0.88}
    ]


@pytest.fixture
def ambiguous_gender_adult_predictions() -> List[Dict[str, Any]]:
    """Adult predictions with high gender entropy (prob ~ 0.50)."""
    return [
        {"teacher": "insightface", "age": 30.0, "gender_prob": 0.52},
        {"teacher": "mivolo", "age": 31.0, "gender_prob": 0.48},
        {"teacher": "fairface", "age": 29.5, "gender_prob": 0.50},
        {"teacher": "clip", "age": 30.5, "gender_prob": 0.51}
    ]


@pytest.fixture
def prepubertal_child_predictions() -> List[Dict[str, Any]]:
    """Prepubertal child predictions (age < 12) with moderate gender ambiguity."""
    return [
        {"teacher": "insightface", "age": 6.0, "gender_prob": 0.55},
        {"teacher": "mivolo", "age": 6.5, "gender_prob": 0.58},
        {"teacher": "fairface", "age": 5.8, "gender_prob": 0.52},
        {"teacher": "clip", "age": 6.2, "gender_prob": 0.60}
    ]


@pytest.fixture
def tmp_dataset_root(tmp_path: Path) -> Path:
    """Creates a temporary mock dataset folder hierarchy representing Kaggle mounts."""
    root = tmp_path / "kaggle_input"
    root.mkdir(parents=True, exist_ok=True)
    for ds in ["fairface", "celeba", "afad", "appa_real"]:
        ds_dir = root / ds
        ds_dir.mkdir(parents=True, exist_ok=True)
        # Create a sample image in each
        sample_img = create_synthetic_face(160)
        cv2.imwrite(str(ds_dir / "sample_01.jpg"), sample_img)
    return root
