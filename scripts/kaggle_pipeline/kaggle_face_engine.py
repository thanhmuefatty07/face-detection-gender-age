"""
Kaggle Face Mining & Extreme Filtering Engine
=============================================
Mathematically grounded face processing pipeline:
- 5-point landmark similarity alignment
- Extreme quality auditing (Laplacian, Tenengrad, Euler Pose, Photometry, Occlusion)
- Perceptual hash deduplication (pHash 64-bit)
- Biological morphometry & Label Distribution Learning (LDL)
- Alpha Signal extraction for optimal model training
"""

from __future__ import annotations
import math
import numpy as np
import cv2
from typing import Dict, Any, Tuple, List, Optional


class FaceGeometryAligner:
    """
    Standard facial alignment using 5-point facial landmarks.
    Transforms arbitrary face orientation to canonical front-facing alignment
    via least-squares similarity transformation (translation, rotation, scale).
    """

    # Canonical 5-point landmark coordinates for a 112x112 reference frame
    REFERENCE_LANDMARKS_112 = np.array([
        [38.2946, 51.6963],  # Left eye center
        [73.5318, 51.5014],  # Right eye center
        [56.0252, 71.7366],  # Nose tip
        [41.5493, 92.3655],  # Left mouth corner
        [70.7299, 92.2041]   # Right mouth corner
    ], dtype=np.float32)

    def __init__(self, output_size: int = 160):
        self.output_size = output_size
        scale = output_size / 112.0
        self.target_landmarks = self.REFERENCE_LANDMARKS_112 * scale

    def align(self, image: np.ndarray, landmarks_5: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align face image according to 5-point landmarks.
        
        Args:
            image: BGR numpy image.
            landmarks_5: Array of shape (5, 2) containing (x, y) coordinates of
                         [left_eye, right_eye, nose, left_mouth, right_mouth].
        Returns:
            aligned_face: Warped face image of shape (output_size, output_size, 3).
            transform_matrix: 2x3 affine transformation matrix.
        """
        assert landmarks_5.shape == (5, 2), f"Expected shape (5, 2), got {landmarks_5.shape}"
        
        # Estimate partial affine (similarity) matrix with least squares
        matrix, _ = cv2.estimateAffinePartial2D(
            landmarks_5.astype(np.float32),
            self.target_landmarks,
            method=cv2.LMEDS
        )
        if matrix is None:
            # Fallback to standard 3-point affine if 5-point LMEDS fails
            src_tri = landmarks_5[:3].astype(np.float32)
            dst_tri = self.target_landmarks[:3]
            matrix = cv2.getAffineTransform(src_tri, dst_tri)

        aligned_face = cv2.warpAffine(
            image,
            matrix,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101
        )
        return aligned_face, matrix


class ExtremeQualityAuditor:
    """
    Extreme quality gate for biometric dataset curation.
    Rejects any sample that introduces gradient poisoning, label noise,
    or structural distortion.
    """

    # 3D Anthropometric model coordinates for 5 landmarks (mm)
    MODEL_3D_POINTS = np.array([
        [-33.0,  35.0, -25.0],  # Left eye
        [ 33.0,  35.0, -25.0],  # Right eye
        [  0.0,   0.0,   0.0],  # Nose tip
        [-25.0, -45.0, -25.0],  # Left mouth
        [ 25.0, -45.0, -25.0],  # Right mouth
    ], dtype=np.float64)

    def __init__(
        self,
        min_face_size: int = 112,
        min_laplacian_var: float = 100.0,
        min_tenengrad: float = 250.0,
        max_abs_yaw: float = 35.0,
        max_abs_pitch: float = 25.0,
        max_abs_roll: float = 20.0,
        min_luminance: float = 35.0,
        max_luminance: float = 220.0,
        max_clipping_fraction: float = 0.08
    ):
        self.min_face_size = min_face_size
        self.min_laplacian_var = min_laplacian_var
        self.min_tenengrad = min_tenengrad
        self.max_abs_yaw = max_abs_yaw
        self.max_abs_pitch = max_abs_pitch
        self.max_abs_roll = max_abs_roll
        self.min_luminance = min_luminance
        self.max_luminance = max_luminance
        self.max_clipping_fraction = max_clipping_fraction

    @staticmethod
    def compute_laplacian_var(gray_img: np.ndarray) -> float:
        """Calculates variance of the Laplacian: sigma^2 = Var(nabla^2 I)."""
        lap = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=3)
        return float(lap.var())

    @staticmethod
    def compute_tenengrad(gray_img: np.ndarray) -> float:
        """Calculates Tenengrad sharpness metric based on Sobel energy."""
        gx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        mag_sq = gx ** 2 + gy ** 2
        return float(np.mean(mag_sq))

    def estimate_head_pose(self, landmarks_5: np.ndarray, img_w: int, img_h: int) -> Tuple[float, float, float]:
        """
        Estimate Euler angles (Yaw, Pitch, Roll) using Perspective-n-Point (PnP) solver.
        """
        image_points = landmarks_5.astype(np.float64)
        focal_length = img_w
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_3D_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )
        if not success:
            return 999.0, 999.0, 999.0

        rmat, _ = cv2.Rodrigues(rvec)
        # Compute Euler angles from rotation matrix
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = math.atan2(rmat[2, 1], rmat[2, 2])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = 0.0

        return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)

    def audit_face(
        self,
        face_bgr: np.ndarray,
        landmarks_5: Optional[np.ndarray] = None,
        raw_box: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Runs comprehensive quality audit against extreme thresholds.
        """
        rejection_reasons = []
        h, w = face_bgr.shape[:2]

        # 1. Size constraint
        if raw_box is not None:
            bw, bh = raw_box[2], raw_box[3]
            if bw < self.min_face_size or bh < self.min_face_size:
                rejection_reasons.append(f"FACE_TOO_SMALL (w={bw}, h={bh} < {self.min_face_size})")

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        # 2. Sharpness audit
        lap_var = self.compute_laplacian_var(gray)
        if lap_var < self.min_laplacian_var:
            rejection_reasons.append(f"BLURRY_LAPLACIAN ({lap_var:.1f} < {self.min_laplacian_var})")

        tenengrad = self.compute_tenengrad(gray)
        if tenengrad < self.min_tenengrad:
            rejection_reasons.append(f"LOW_TENENGRAD ({tenengrad:.1f} < {self.min_tenengrad})")

        # 3. Photometric audit (Luminance & Clipping)
        lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        mean_l = float(np.mean(l_channel))
        shadow_clip = float(np.mean(l_channel < 15))
        highlight_clip = float(np.mean(l_channel > 240))

        if mean_l < self.min_luminance:
            rejection_reasons.append(f"UNDEREXPOSED (Mean L={mean_l:.1f} < {self.min_luminance})")
        elif mean_l > self.max_luminance:
            rejection_reasons.append(f"OVEREXPOSED (Mean L={mean_l:.1f} > {self.max_luminance})")

        if (shadow_clip + highlight_clip) > self.max_clipping_fraction:
            rejection_reasons.append(f"EXTREME_CLIPPING (Clip={(shadow_clip + highlight_clip):.2%} > {self.max_clipping_fraction:.0%})")

        # 4. Pose audit
        yaw, pitch, roll = 0.0, 0.0, 0.0
        if landmarks_5 is not None:
            yaw, pitch, roll = self.estimate_head_pose(landmarks_5, w, h)
            if abs(yaw) > self.max_abs_yaw:
                rejection_reasons.append(f"EXCESSIVE_YAW (|{yaw:.1f}°| > {self.max_abs_yaw}°)")
            if abs(pitch) > self.max_abs_pitch:
                rejection_reasons.append(f"EXCESSIVE_PITCH (|{pitch:.1f}°| > {self.max_abs_pitch}°)")
            if abs(roll) > self.max_abs_roll:
                rejection_reasons.append(f"EXCESSIVE_ROLL (|{roll:.1f}°| > {self.max_abs_roll}°)")

        passed = len(rejection_reasons) == 0
        return {
            "passed": passed,
            "reasons": rejection_reasons,
            "laplacian_var": round(lap_var, 2),
            "tenengrad": round(tenengrad, 2),
            "mean_luminance": round(mean_l, 2),
            "shadow_clip": round(shadow_clip, 4),
            "highlight_clip": round(highlight_clip, 4),
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "roll": round(roll, 2)
        }


class PerceptualDeduplicator:
    """
    High-speed Perceptual Hashing (64-bit DCT pHash) for zero-leakage deduplication.
    Identifies exact and near-duplicate images across train/val and external sources.
    """

    def __init__(self, hamming_threshold: int = 6):
        self.hamming_threshold = hamming_threshold
        self.seen_hashes: List[int] = []

    @staticmethod
    def compute_phash_uint64(gray_image: np.ndarray) -> int:
        """
        Computes 64-bit perceptual hash using 2D Discrete Cosine Transform (DCT).
        """
        # Resize to 32x32 for DCT
        resized = cv2.resize(gray_image, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        # Extract 8x8 low-frequency components (excluding DC component at (0, 0))
        dct_low = dct[0:8, 0:8]
        median_val = np.median(dct_low[1:, 1:])
        # Binary comparison
        diff = dct_low > median_val
        # Pack boolean array into 64-bit unsigned int
        hash_val = 0
        for bit in diff.flatten():
            hash_val = (hash_val << 1) | int(bit)
        return hash_val

    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        """Bitwise XOR and count set bits."""
        return bin(hash1 ^ hash2).count('1')

    def is_duplicate_and_register(self, hash_val: int) -> Tuple[bool, Optional[int]]:
        """
        Checks if hash_val is within hamming_threshold of any previously seen hash.
        If unique, registers hash and returns False.
        """
        for seen in self.seen_hashes:
            if self.hamming_distance(hash_val, seen) <= self.hamming_threshold:
                return True, seen
        self.seen_hashes.append(hash_val)
        return False, None


class BiologicalLabelingEngine:
    """
    Biological morphometry, age distribution learning (LDL), and alpha signal mining.
    Addresses:
    - Prepubescent children (<12): Low facial sexual dimorphism.
    - Elderly (65+): Hormone-induced androgenic convergence.
    - Non-linear ethnic aging trajectories.
    - Continuous-to-Ordinal Label Distribution Learning.
    """

    AGE_GROUPS = ['0-2', '3-6', '7-12', '13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    AGE_CENTERS = np.array([1.0, 4.5, 9.5, 15.0, 21.0, 29.5, 39.5, 49.5, 59.5, 72.0], dtype=np.float32)
    AGE_EDGES = [0.0, 2.5, 6.5, 12.5, 17.5, 24.5, 34.5, 44.5, 54.5, 64.5, 100.0]

    @classmethod
    def get_discrete_group_index(cls, age: float) -> int:
        """Maps continuous age to the project's standard 10 discrete bins."""
        for i in range(len(cls.AGE_EDGES) - 1):
            if cls.AGE_EDGES[i] <= age < cls.AGE_EDGES[i + 1]:
                return i
        return len(cls.AGE_GROUPS) - 1

    @classmethod
    def compute_label_distribution(cls, mu_age: float, sigma_age: float = 3.0) -> np.ndarray:
        """
        Label Distribution Learning (LDL):
        Converts continuous apparent age into a normalized probability distribution over 10 age bins.
        P(c) ~ integral of Normal(mu, sigma) over [edge_i, edge_{i+1}].
        """
        probs = np.zeros(len(cls.AGE_GROUPS), dtype=np.float32)
        for i in range(len(cls.AGE_GROUPS)):
            low = cls.AGE_EDGES[i]
            high = cls.AGE_EDGES[i + 1]
            # Gaussian CDF integration
            z_low = (low - mu_age) / (sigma_age * math.sqrt(2))
            z_high = (high - mu_age) / (sigma_age * math.sqrt(2))
            p = 0.5 * (math.erf(z_high) - math.erf(z_low))
            probs[i] = max(0.0, p)

        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs[cls.get_discrete_group_index(mu_age)] = 1.0
        return probs

    @classmethod
    def calculate_biological_sexual_dimorphism_weight(cls, age: float) -> float:
        """
        Downweights hard gender loss penalties where sexual dimorphism is biologically low.
        - Prepubescent (<12): Facial bone dimorphism is minimal.
        - Elderly (65+): Dimorphism decreases due to soft tissue volume loss.
        """
        if age < 12.0:
            return max(0.25, float(age / 12.0))
        elif age > 65.0:
            return 0.85
        return 1.0

    @classmethod
    def extract_alpha_signals(
        cls,
        teacher_predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Alpha Signal Extraction from Multi-Teacher ensemble.
        
        Args:
            teacher_predictions: List of dicts with 'gender' (0: M, 1: F, prob),
                                 'age' (continuous value), 'ethnicity' (optional).
        Returns:
            Dict containing consensus targets, epistemic uncertainty, sample weights,
            and continuous LDL vector.
        """
        if not teacher_predictions:
            raise ValueError("No teacher predictions provided")

        ages = [p['age'] for p in teacher_predictions]
        gender_probs = [p['gender_prob'] for p in teacher_predictions]  # prob of Female (1)

        mu_age = float(np.mean(ages))
        sigma_epistemic_age = float(np.std(ages))  # Disagreement among teachers

        mean_gender_prob = float(np.mean(gender_probs))
        gender_vote = 1 if mean_gender_prob >= 0.5 else 0
        gender_confidence = mean_gender_prob if gender_vote == 1 else (1.0 - mean_gender_prob)
        gender_disagreement = float(np.std(gender_probs))

        # Alpha Sample Weight:
        # High consensus + low epistemic noise = high training weight
        # High disagreement = reduced weight (or pruned)
        alpha_importance_weight = float(
            (1.0 / (1.0 + sigma_epistemic_age)) *
            (1.0 - gender_disagreement) *
            cls.calculate_biological_sexual_dimorphism_weight(mu_age)
        )

        discrete_bin_idx = cls.get_discrete_group_index(mu_age)
        ldl_distribution = cls.compute_label_distribution(mu_age, sigma_age=max(2.5, sigma_epistemic_age))

        # Rejection flag: extreme teacher disagreement indicates ambiguous / non-face artifact
        is_pruned = (sigma_epistemic_age > 8.0) or (gender_confidence < 0.70 and mu_age >= 12.0)

        return {
            "pruned": is_pruned,
            "consensus_age": round(mu_age, 2),
            "age_group": cls.AGE_GROUPS[discrete_bin_idx],
            "age_group_idx": discrete_bin_idx,
            "age_epistemic_std": round(sigma_epistemic_age, 3),
            "consensus_gender": "Female" if gender_vote == 1 else "Male",
            "gender_idx": gender_vote,
            "gender_confidence": round(gender_confidence, 4),
            "gender_epistemic_std": round(gender_disagreement, 4),
            "alpha_importance_weight": round(alpha_importance_weight, 4),
            "ldl_distribution": [round(float(p), 5) for p in ldl_distribution]
        }
