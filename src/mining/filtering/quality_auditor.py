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
        max_cheek_delta_e: float = 18.0,
        min_shannon_entropy: float = 3.8,
        max_flat_gradient_ratio: float = 0.88,
        max_top8_bin_concentration: float = 0.85,
        min_cheek_microtexture_std: float = 1.15,
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
        self.min_shannon_entropy = min_shannon_entropy
        self.max_flat_gradient_ratio = max_flat_gradient_ratio
        self.max_top8_bin_concentration = max_top8_bin_concentration
        self.min_cheek_microtexture_std = min_cheek_microtexture_std

    def audit_face(
        self,
        face_bgr: np.ndarray,
        landmarks_5: Optional[np.ndarray] = None,
        raw_width: Optional[int] = None,
        raw_height: Optional[int] = None
    ) -> FaceAuditResult:
        """
        Audits a face crop against all extreme criteria, including non-real/synthetic
        cartoon, sketch, posterization, CGI/airbrushed plastic skin, and flat-render rejection gates.
        """
        rejection_reasons: List[str] = []
        if face_bgr is None or face_bgr.size == 0:
            return FaceAuditResult(passed=False, rejection_reasons=["EMPTY_IMAGE"])

        h, w = face_bgr.shape[:2]
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if len(face_bgr.shape) == 3 else face_bgr

        # 1. Laplacian sharpness
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        lap_var = float(lap.var())
        lap_var_r = round(lap_var, 2)
        if lap_var < self.min_laplacian_var or lap_var_r < self.min_laplacian_var:
            rejection_reasons.append(f"INSUFFICIENT_SHARPNESS_LAPLACIAN ({lap_var:.1f} < {self.min_laplacian_var})")

        # 2. Tenengrad edge gradient
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.mean(gx**2 + gy**2))
        tenengrad_r = round(tenengrad, 2)
        if tenengrad < self.min_tenengrad or tenengrad_r < self.min_tenengrad:
            rejection_reasons.append(f"INSUFFICIENT_EDGE_ENERGY_TENENGRAD ({tenengrad:.1f} < {self.min_tenengrad})")

        # 3. CIE L*a*b* Photometry & Sensor Clipping
        if len(face_bgr.shape) == 3:
            lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
        else:
            l_channel = gray

        mean_l = float(np.mean(l_channel))
        mean_l_r = round(mean_l, 2)
        if mean_l < self.min_mean_luminance or mean_l_r < self.min_mean_luminance:
            rejection_reasons.append(f"UNDEREXPOSED_LUMINANCE ({mean_l:.1f} < {self.min_mean_luminance})")
        elif mean_l > self.max_mean_luminance or mean_l_r > self.max_mean_luminance:
            rejection_reasons.append(f"OVEREXPOSED_LUMINANCE ({mean_l:.1f} > {self.max_mean_luminance})")

        # Shadows L < 8 (prevent black hair false rejections) & Highlights L > 240
        shadow_clip = float(np.mean(l_channel < 8))
        highlight_clip = float(np.mean(l_channel > 240))
        shadow_clip_r = round(shadow_clip, 4)
        highlight_clip_r = round(highlight_clip, 4)
        total_clip = shadow_clip + highlight_clip
        total_clip_r = float(np.float64(shadow_clip_r) + np.float64(highlight_clip_r))
        if (
            total_clip > self.max_clip_fraction
            or total_clip_r > self.max_clip_fraction
            or shadow_clip_r > self.max_clip_fraction
            or highlight_clip_r > self.max_clip_fraction
        ):
            rejection_reasons.append(f"SENSOR_CLIPPING_EXCEEDED ({max(total_clip, total_clip_r)*100:.2f}% > {self.max_clip_fraction*100}%)")

        # 4. Bilateral Cheek Occlusion
        cheek_delta_e = 0.0
        if len(face_bgr.shape) == 3:
            l_cheek = lab[int(0.55*h):int(0.75*h), int(0.15*w):int(0.35*w)]
            r_cheek = lab[int(0.55*h):int(0.75*h), int(0.65*w):int(0.85*w)]
            if l_cheek.size > 0 and r_cheek.size > 0:
                l_mean = np.mean(l_cheek, axis=(0, 1)).astype(np.float64)
                r_mean = np.mean(r_cheek, axis=(0, 1)).astype(np.float64)
                cheek_delta_e = float(np.linalg.norm(l_mean - r_mean))
                if cheek_delta_e > self.max_cheek_delta_e or round(cheek_delta_e, 2) > self.max_cheek_delta_e:
                    rejection_reasons.append(f"BILATERAL_CHEEK_ASYMMETRY_OCCLUSION (ΔE={cheek_delta_e:.1f} > {self.max_cheek_delta_e})")

        # 5. Anthropometric Head Pose solvePnP
        yaw, pitch, roll = 0.0, 0.0, 0.0
        if landmarks_5 is not None:
            pose_w = raw_width if raw_width is not None else w
            pose_h = raw_height if raw_height is not None else h
            yaw, pitch, roll = estimate_head_pose_pnp(landmarks_5, pose_w, pose_h)
            yaw_r, pitch_r, roll_r = round(yaw, 2), round(pitch, 2), round(roll, 2)
            eps = 1e-4
            if abs(yaw) - self.max_abs_yaw > eps or abs(yaw_r) > self.max_abs_yaw:
                rejection_reasons.append(f"EXCESSIVE_YAW (|{yaw:.1f}°| > {self.max_abs_yaw}°)")
            if abs(pitch) - self.max_abs_pitch > eps or abs(pitch_r) > self.max_abs_pitch:
                rejection_reasons.append(f"EXCESSIVE_PITCH (|{pitch:.1f}°| > {self.max_abs_pitch}°)")
            if abs(roll) - self.max_abs_roll > eps or abs(roll_r) > self.max_abs_roll:
                rejection_reasons.append(f"EXCESSIVE_ROLL (|{roll:.1f}°| > {self.max_abs_roll}°)")

        # 6. Information-Theoretic Shannon Luminance Entropy & Non-Real Image Rejection Gates
        shannon_ent = self.compute_shannon_luminance_entropy(gray)
        michelson_c = self.compute_michelson_contrast(gray)
        flat_grad_ratio = self.compute_flat_gradient_ratio(gx, gy, gray_img=gray)
        top8_conc = self.compute_top_bin_concentration(gray, top_k=8)
        cheek_micro_std = self.compute_cheek_microtexture_std(gray)
        if len(face_bgr.shape) == 3:
            chroma_std = float(np.hypot(np.std(lab[:, :, 1].astype(np.float32)), np.std(lab[:, :, 2].astype(np.float32))))
        else:
            chroma_std = 0.0

        if shannon_ent < 1.5 or (
            shannon_ent < self.min_shannon_entropy
            and (flat_grad_ratio > self.max_flat_gradient_ratio or chroma_std < 2.0)
        ):
            rejection_reasons.append(
                f"INSUFFICIENT_TONAL_ENTROPY_NON_REAL ({shannon_ent:.2f} < {self.min_shannon_entropy})"
            )
        if flat_grad_ratio > self.max_flat_gradient_ratio:
            rejection_reasons.append(
                f"SYNTHETIC_FLAT_SHADING_OR_CARTOON ({flat_grad_ratio:.3f} > {self.max_flat_gradient_ratio})"
            )
        if top8_conc > self.max_top8_bin_concentration and (
            flat_grad_ratio > self.max_flat_gradient_ratio or chroma_std < 2.0
        ):
            rejection_reasons.append(
                f"POSTERIZED_OR_SKETCH_HISTOGRAM ({top8_conc:.3f} > {self.max_top8_bin_concentration})"
            )
        if cheek_micro_std < self.min_cheek_microtexture_std:
            rejection_reasons.append(
                f"SYNTHETIC_PLASTIC_SKIN_MICROTEXTURE_NON_REAL ({cheek_micro_std:.3f} < {self.min_cheek_microtexture_std})"
            )

        skin_ita_deg, skin_group = (
            self.compute_skin_tone_ita(lab)
            if len(face_bgr.shape) == 3
            else (28.0, "Type_III_Intermediate")
        )

        passed = len(rejection_reasons) == 0
        return FaceAuditResult(
            passed=passed,
            rejection_reasons=rejection_reasons,
            laplacian_variance=lap_var_r,
            tenengrad_energy=tenengrad_r,
            mean_luminance=mean_l_r,
            shadow_clip_fraction=shadow_clip_r,
            highlight_clip_fraction=highlight_clip_r,
            cheek_occlusion_delta_e=round(cheek_delta_e, 2),
            yaw=round(yaw, 2),
            pitch=round(pitch, 2),
            roll=round(roll, 2),
            shannon_entropy=round(shannon_ent, 3),
            michelson_contrast=round(michelson_c, 4),
            cheek_microtexture_std=round(cheek_micro_std, 4),
            skin_tone_ita_deg=round(skin_ita_deg, 2),
            skin_tone_group=skin_group,
        )

    @staticmethod
    def compute_skin_tone_ita(lab_img: np.ndarray) -> Tuple[float, str]:
        """
        Computes Chardon et al. (1991) dermatological Individual Typology Angle (ITA°)
        in CIE L*a*b* space on bilateral cheek ROIs with physiological melanin chrominance
        regularization for achromatic/grayscale or cool-cast images:
            ITA° = arctan((L*_CIE - 50) / b*_eff) * (180 / pi)
        and maps ITA° to the 6 Fitzpatrick/Chardon skin-tone categories.
        """
        if lab_img is None or lab_img.size == 0 or lab_img.ndim != 3:
            return 28.0, "Type_III_Intermediate"
        h, w = lab_img.shape[:2]
        y0, y1 = int(0.50 * h), max(int(0.50 * h) + 1, int(0.75 * h))
        lx0, lx1 = int(0.18 * w), max(int(0.18 * w) + 1, int(0.40 * w))
        rx0, rx1 = int(0.60 * w), max(int(0.60 * w) + 1, int(0.82 * w))
        l_patch = lab_img[y0:y1, lx0:lx1].astype(np.float64)
        r_patch = lab_img[y0:y1, rx0:rx1].astype(np.float64)
        if l_patch.size > 0 and r_patch.size > 0:
            l_cv = 0.5 * (float(np.median(l_patch[:, :, 0])) + float(np.median(r_patch[:, :, 0])))
            a_cv = 0.5 * (float(np.median(l_patch[:, :, 1])) + float(np.median(r_patch[:, :, 1])))
            b_cv = 0.5 * (float(np.median(l_patch[:, :, 2])) + float(np.median(r_patch[:, :, 2])))
        else:
            l_cv = float(np.median(lab_img[:, :, 0]))
            a_cv = float(np.median(lab_img[:, :, 1]))
            b_cv = float(np.median(lab_img[:, :, 2]))
        l_cie = l_cv * (100.0 / 255.0)
        a_cie = a_cv - 128.0
        b_cie = b_cv - 128.0
        chroma_ab = math.hypot(a_cie, b_cie)
        b_phys = 12.0 + 10.0 * math.exp(-((l_cie - 55.0) ** 2) / (2.0 * 18.0 * 18.0))
        alpha = min(1.0, max(0.0, (chroma_ab - 1.5) / 6.0))
        b_eff = max(8.0, alpha * max(b_cie, 8.0) + (1.0 - alpha) * b_phys)
        ita_deg = math.degrees(math.atan((l_cie - 50.0) / b_eff))
        if ita_deg > 55.0:
            group = "Type_I_Very_Light"
        elif ita_deg > 41.0:
            group = "Type_II_Light"
        elif ita_deg > 28.0:
            group = "Type_III_Intermediate"
        elif ita_deg > 10.0:
            group = "Type_IV_Tan"
        elif ita_deg > -30.0:
            group = "Type_V_Brown"
        else:
            group = "Type_VI_Dark"
        return float(ita_deg), group

    @staticmethod
    def compute_shannon_luminance_entropy(gray_img: np.ndarray) -> float:
        """
        Computes 8-bit Shannon luminance entropy H(I) = -sum p(k) log2 p(k) in [0, 8] bits.
        Measures tonal information content and dynamic range utilization.
        """
        if gray_img is None or gray_img.size == 0:
            return 0.0
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).ravel()
        p = hist / max(float(gray_img.size), 1.0)
        p_pos = p[p > 0]
        return float(-np.sum(p_pos * np.log2(p_pos)))

    @staticmethod
    def compute_michelson_contrast(gray_img: np.ndarray) -> float:
        """
        Computes robust percentile Michelson contrast (P95 - P5) / (P95 + P5 + eps).
        """
        if gray_img is None or gray_img.size == 0:
            return 0.0
        p5, p95 = np.percentile(gray_img.astype(np.float32), [5.0, 95.0])
        denom = float(p95 + p5)
        if denom <= 1e-6:
            return 0.0
        return float(max(0.0, min(1.0, (p95 - p5) / denom)))

    @staticmethod
    def compute_flat_gradient_ratio(
        gx: np.ndarray, gy: np.ndarray, gray_img: Optional[np.ndarray] = None
    ) -> float:
        """
        Computes fraction of interior facial pixels with near-zero Sobel gradient (|Gx| + |Gy| <= 1.0).
        Real optical photographs have continuous sensor shot noise and skin pore micro-texture,
        whereas vector cartoons, silhouettes, and flat-shaded CGI (including grain-injected flat renders)
        have large zero-gradient regions on the raw or 5x5 denoised image.
        """
        if gx is None or gy is None or gx.size == 0:
            return 1.0
        h, w = gx.shape[:2]
        y0, y1 = int(0.15 * h), max(int(0.15 * h) + 1, int(0.85 * h))
        x0, x1 = int(0.15 * w), max(int(0.15 * w) + 1, int(0.85 * w))
        mag_l1 = np.abs(gx[y0:y1, x0:x1]) + np.abs(gy[y0:y1, x0:x1])
        raw_flat = float(np.mean(mag_l1 <= 1.0))
        if gray_img is not None and h >= 32 and w >= 32 and raw_flat <= 0.88:
            denoised = cv2.GaussianBlur(gray_img, (5, 5), 1.2)
            hist_d = cv2.calcHist([denoised], [0], None, [256], [0, 256]).ravel()
            tot_d = float(hist_d.sum())
            top8_d = float(np.sort(hist_d)[-8:].sum() / tot_d) if tot_d > 0.0 else 1.0
            if top8_d > 0.80:
                dgx = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
                dgy = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
                dmag = np.abs(dgx[y0:y1, x0:x1]) + np.abs(dgy[y0:y1, x0:x1])
                denoised_flat = float(np.mean(dmag <= 2.5))
                if denoised_flat > 0.78:
                    return max(raw_flat, min(0.99, denoised_flat + 0.12))
        return raw_flat

    @staticmethod
    def compute_top_bin_concentration(gray_img: np.ndarray, top_k: int = 8) -> float:
        """
        Computes fraction of pixels concentrated in the top-k most frequent 8-bit luminance bins.
        Rejects binary sketches, line drawings, and heavily posterized/color-quantized avatars.
        """
        if gray_img is None or gray_img.size == 0:
            return 1.0
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).ravel()
        p = hist / max(float(gray_img.size), 1.0)
        top_probs = np.partition(p, -min(top_k, len(p)))[-min(top_k, len(p)):]
        return float(np.sum(top_probs))

    @staticmethod
    def compute_cheek_microtexture_std(gray_img: np.ndarray) -> float:
        """
        Computes high-frequency skin micro-texture standard deviation on bilateral cheek ROIs
        via 3x3 Gaussian high-pass residual: std(I - G_3x3 * I).
        Real optical human skin photographs exhibit continuous sensor shot noise and biological
        pore micro-texture on the cheeks, whereas CGI/3D-rendered/Diffusion/airbrushed faces
        with sharp drawn eyes/hair exhibit poreless plastic cheek surfaces.
        """
        if gray_img is None or gray_img.size == 0:
            return 0.0
        gray_f = gray_img.astype(np.float32)
        blurred = cv2.GaussianBlur(gray_f, (3, 3), 0)
        high_pass = gray_f - blurred
        h, w = gray_img.shape[:2]
        y0, y1 = int(0.50 * h), max(int(0.50 * h) + 1, int(0.75 * h))
        lx0, lx1 = int(0.18 * w), max(int(0.18 * w) + 1, int(0.40 * w))
        rx0, rx1 = int(0.60 * w), max(int(0.60 * w) + 1, int(0.82 * w))
        l_roi = high_pass[y0:y1, lx0:lx1]
        r_roi = high_pass[y0:y1, rx0:rx1]
        if l_roi.size == 0 or r_roi.size == 0:
            return float(np.std(high_pass))
        return float(0.5 * (np.std(l_roi) + np.std(r_roi)))
