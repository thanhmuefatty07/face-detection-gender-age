"""
Extreme Quality Auditor
=======================
Enforces highest biometric acceptance gates:
- Laplacian variance >= 100.0 (micro-texture sharpness)
- Tenengrad Sobel energy >= 250.0 (high-frequency edge gradient)
- CIE L*a*b* Photometric Gate: Mean L in [35.0, 220.0]
- Sensor Clipping: Shadow (L < 8) + Highlight (L > 240) <= 8.0%
- Bilateral Cheek Delta E <= 18.0 (occlusion & non-uniform lighting detection)
- Anthropometric solvePnP Head Pose: |Yaw| <= 35.0°, |Pitch| <= 25.0°, |Roll| <= 20.0°
  with IEEE 754 float epsilon protection (eps = 1e-4).
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

from ...biometrics.core.geometry import estimate_head_pose_pnp
from ...biometrics.core.types import FaceAuditResult


class ExtremeQualityAuditor:
    """Rigorous optical, geometric, and photometric face quality auditor."""

    def __init__(
        self,
        min_laplacian_var: float = 100.0,
        min_tenengrad: float = 250.0,
        max_abs_yaw: float = 35.0,
        max_abs_pitch: float = 25.0,
        max_abs_roll: float = 20.0,
        min_mean_luminance: float = 35.0,
        max_mean_luminance: float = 220.0,
        max_clip_fraction: float = 0.08,
        max_cheek_delta_e: float = 18.0
    ):
        self.min_laplacian_var = min_laplacian_var
        self.min_tenengrad = min_tenengrad
        self.max_abs_yaw = max_abs_yaw
        self.max_abs_pitch = max_abs_pitch
        self.max_abs_roll = max_abs_roll
        self.min_mean_luminance = min_mean_luminance
        self.max_mean_luminance = max_mean_luminance
        self.max_clip_fraction = max_clip_fraction
        self.max_cheek_delta_e = max_cheek_delta_e

    def audit_face(
        self,
        face_bgr: np.ndarray,
        landmarks_5: Optional[np.ndarray] = None
    ) -> FaceAuditResult:
        """
        Audits a face crop against all extreme criteria.
        """
        rejection_reasons: List[str] = []
        if face_bgr is None or face_bgr.size == 0:
            return FaceAuditResult(passed=False, rejection_reasons=["EMPTY_IMAGE"])

        h, w = face_bgr.shape[:2]
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if len(face_bgr.shape) == 3 else face_bgr

        # 1. Laplacian sharpness
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        lap_var = float(lap.var())
        if lap_var < self.min_laplacian_var:
            rejection_reasons.append(f"INSUFFICIENT_SHARPNESS_LAPLACIAN ({lap_var:.1f} < {self.min_laplacian_var})")

        # 2. Tenengrad edge gradient
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.mean(gx**2 + gy**2))
        if tenengrad < self.min_tenengrad:
            rejection_reasons.append(f"INSUFFICIENT_EDGE_ENERGY_TENENGRAD ({tenengrad:.1f} < {self.min_tenengrad})")

        # 3. CIE L*a*b* Photometry & Sensor Clipping
        if len(face_bgr.shape) == 3:
            lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
        else:
            l_channel = gray

        mean_l = float(np.mean(l_channel))
        if mean_l < self.min_mean_luminance:
            rejection_reasons.append(f"UNDEREXPOSED_LUMINANCE ({mean_l:.1f} < {self.min_mean_luminance})")
        elif mean_l > self.max_mean_luminance:
            rejection_reasons.append(f"OVEREXPOSED_LUMINANCE ({mean_l:.1f} > {self.max_mean_luminance})")

        # Shadows L < 8 (prevent black hair false rejections) & Highlights L > 240
        shadow_clip = float(np.mean(l_channel < 8))
        highlight_clip = float(np.mean(l_channel > 240))
        total_clip = shadow_clip + highlight_clip
        if total_clip > self.max_clip_fraction:
            rejection_reasons.append(f"SENSOR_CLIPPING_EXCEEDED ({total_clip*100:.1f}% > {self.max_clip_fraction*100}%)")

        # 4. Bilateral Cheek Occlusion
        cheek_delta_e = 0.0
        if len(face_bgr.shape) == 3:
            l_cheek = lab[int(0.55*h):int(0.75*h), int(0.15*w):int(0.35*w)]
            r_cheek = lab[int(0.55*h):int(0.75*h), int(0.65*w):int(0.85*w)]
            if l_cheek.size > 0 and r_cheek.size > 0:
                l_mean = np.mean(l_cheek, axis=(0, 1)).astype(np.float64)
                r_mean = np.mean(r_cheek, axis=(0, 1)).astype(np.float64)
                cheek_delta_e = float(np.linalg.norm(l_mean - r_mean))
                if cheek_delta_e > self.max_cheek_delta_e:
                    rejection_reasons.append(f"BILATERAL_CHEEK_ASYMMETRY_OCCLUSION (ΔE={cheek_delta_e:.1f} > {self.max_cheek_delta_e})")

        # 5. Anthropometric Head Pose solvePnP
        yaw, pitch, roll = 0.0, 0.0, 0.0
        if landmarks_5 is not None:
            yaw, pitch, roll = estimate_head_pose_pnp(landmarks_5, w, h)
            eps = 1e-4
            if abs(yaw) - self.max_abs_yaw > eps:
                rejection_reasons.append(f"EXCESSIVE_YAW (|{yaw:.1f}°| > {self.max_abs_yaw}°)")
            if abs(pitch) - self.max_abs_pitch > eps:
                rejection_reasons.append(f"EXCESSIVE_PITCH (|{pitch:.1f}°| > {self.max_abs_pitch}°)")
            if abs(roll) - self.max_abs_roll > eps:
                rejection_reasons.append(f"EXCESSIVE_ROLL (|{roll:.1f}°| > {self.max_abs_roll}°)")

        passed = len(rejection_reasons) == 0
        return FaceAuditResult(
            passed=passed,
            rejection_reasons=rejection_reasons,
            laplacian_variance=round(lap_var, 2),
            tenengrad_energy=round(tenengrad, 2),
            mean_luminance=round(mean_l, 2),
            shadow_clip_fraction=round(shadow_clip, 4),
            highlight_clip_fraction=round(highlight_clip, 4),
            cheek_occlusion_delta_e=round(cheek_delta_e, 2),
            yaw=round(yaw, 2),
            pitch=round(pitch, 2),
            roll=round(roll, 2)
        )
