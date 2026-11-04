"""
Physical Environmental Augmentation Transforms
==============================================
Realistic physical transformations simulating challenging edge & real-world conditions:
1. Overexposure: Gamma compression (gamma in [0.4, 0.7]), specular highlight flares with Gaussian falloff, CLAHE.
2. Underexposure & Backlight: Gamma expansion (gamma in [1.8, 2.5]), Poisson-Gaussian noise, directional gradients.
3. Motion Blur: Point Spread Function (PSF) kernel (3-11 px) with arbitrary trajectory angles.
4. Cutout Occlusion: Lower-face surgical mask and eye-region sunglasses occlusion.
"""

from __future__ import annotations
import math
import random
from typing import Optional, Union, Tuple
import cv2
import numpy as np


def apply_overexposure_transform(
    image: np.ndarray,
    gamma: float = 0.5,
    flare_strength: float = 0.6,
    flare_center: Optional[Tuple[int, int]] = None,
    is_bgr: bool = False,
) -> np.ndarray:
    """
    Simulates direct sunlight, harsh specular reflections, and glare.
    - Gamma compression compresses dynamic range towards peak luminance.
    - Specular highlight flare creates localized saturated bloom.
    - CLAHE balances local contrast across shadows and highlights.
    
    Args:
        image: (H, W, 3) uint8 image (RGB or BGR)
        gamma: compression exponent in [0.4, 0.7]
        flare_strength: flare peak intensity scaling [0.0, 1.0]
        flare_center: optional (x, y) coordinate of flare center
        is_bgr: True if image is in BGR channel order, False if RGB
    Returns:
        Augmented (H, W, 3) uint8 image
    """
    H, W = image.shape[:2]

    # 1. Gamma compression
    gamma = float(np.clip(gamma, 0.2, 0.9))
    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
    img_gamma = cv2.LUT(image, lut)

    # 2. Specular highlight flare with Gaussian falloff
    if flare_center is None:
        x0 = int(np.random.randint(int(W * 0.2), int(W * 0.8)))
        y0 = int(np.random.randint(int(H * 0.1), int(H * 0.6)))
    else:
        x0, y0 = flare_center

    sigma = float(np.random.uniform(15.0, 35.0))
    Y, X = np.ogrid[:H, :W]
    dist_sq = (X - x0) ** 2 + (Y - y0) ** 2
    flare = np.exp(-dist_sq / (2.0 * sigma ** 2)) * (255.0 * flare_strength)
    flare_3ch = np.dstack([flare, flare, flare])

    img_flare = np.clip(img_gamma.astype(np.float32) + flare_3ch, 0.0, 255.0).astype(np.uint8)

    # 3. CLAHE equalization on luminance channel
    code_to_lab = cv2.COLOR_BGR2LAB if is_bgr else cv2.COLOR_RGB2LAB
    code_from_lab = cv2.COLOR_LAB2BGR if is_bgr else cv2.COLOR_LAB2RGB
    lab = cv2.cvtColor(img_flare, code_to_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, code_from_lab)

    return out


def apply_underexposure_backlight_transform(
    image: np.ndarray,
    gamma: float = 2.2,
    noise_sigma: float = 12.0,
    gradient_dir: str = "horizontal",
) -> np.ndarray:
    """
    Simulates low-light, night-time capture, strong backlighting, and sensor noise.
    - Gamma expansion darkens midtones and shadows.
    - Directional gradient models uneven illumination / backlight silhouettes.
    - Heteroscedastic Poisson-Gaussian noise models optical sensor noise.
    
    Args:
        image: (H, W, 3) uint8 image
        gamma: expansion exponent in [1.8, 2.5]
        noise_sigma: sensor read noise standard deviation
        gradient_dir: 'horizontal', 'vertical', or 'none'
    Returns:
        Augmented (H, W, 3) uint8 image
    """
    H, W = image.shape[:2]

    # 1. Gamma expansion
    gamma = float(np.clip(gamma, 1.2, 3.5))
    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
    img_dark = cv2.LUT(image, lut).astype(np.float32)

    # 2. Directional lighting gradient
    if gradient_dir == "horizontal":
        ramp = np.linspace(0.4, 1.0, W, dtype=np.float32).reshape(1, W, 1)
    elif gradient_dir == "vertical":
        ramp = np.linspace(1.2, 0.5, H, dtype=np.float32).reshape(H, 1, 1)
    else:
        ramp = 1.0
    img_grad = img_dark * ramp

    # 3. Heteroscedastic Poisson-Gaussian sensor noise
    # Shot noise: variance proportional to signal intensity
    shot_noise = np.random.normal(0, 1.0, img_grad.shape).astype(np.float32) * np.sqrt(
        np.maximum(img_grad, 0.0) * 0.4
    )
    # Read noise: additive Gaussian noise
    read_noise = np.random.normal(0, noise_sigma, img_grad.shape).astype(np.float32)

    noisy = img_grad + shot_noise + read_noise
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def apply_motion_blur_transform(
    image: np.ndarray,
    kernel_size: int = 7,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """
    Simulates rapid subject movement or camera jitter using a linear Point Spread Function (PSF).
    
    Args:
        image: (H, W, 3) uint8 image
        kernel_size: PSF length in pixels [3, 11]
        angle_deg: direction of motion trajectory in degrees
    Returns:
        Blurred (H, W, 3) uint8 image
    """
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1

    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0

    # Rotate kernel by angle_deg
    rot_mat = cv2.getRotationMatrix2D((k / 2.0 - 0.5, k / 2.0 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (k, k))

    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum
    else:
        kernel[k // 2, k // 2] = 1.0

    return cv2.filter2D(image, -1, kernel)


def apply_cutout_occlusion_transform(
    image: np.ndarray,
    occlusion_type: str = "mask",
) -> np.ndarray:
    """
    Simulates realistic facial occlusions (medical mouth masks, sunglasses).
    
    Args:
        image: (H, W, 3) uint8 image (assumes 160x160 aligned face)
        occlusion_type: 'mask', 'sunglasses', or 'random'
    Returns:
        Occluded (H, W, 3) uint8 image
    """
    out = image.copy()
    H, W = out.shape[:2]

    target_type = occlusion_type
    if target_type == "random":
        target_type = random.choice(["mask", "sunglasses"])

    if target_type == "mask":
        # Medical / surgical mask covering mouth and chin (lower 40% of aligned face)
        mask_color = [140, 180, 215]  # Surgical light blue
        poly = np.array([
            [int(W * 0.15), int(H * 0.58)],
            [int(W * 0.85), int(H * 0.58)],
            [int(W * 0.82), int(H * 0.96)],
            [int(W * 0.18), int(H * 0.96)],
        ], dtype=np.int32)
        cv2.fillPoly(out, [poly], mask_color)
        # Mask folds / contours for realistic texture
        cv2.line(out, (int(W * 0.18), int(H * 0.70)), (int(W * 0.82), int(H * 0.70)), (120, 160, 195), 1)
        cv2.line(out, (int(W * 0.19), int(H * 0.82)), (int(W * 0.81), int(H * 0.82)), (120, 160, 195), 1)

    elif target_type == "sunglasses":
        # Dark sunglasses covering eye band
        glass_color = [20, 20, 25]  # Tinted dark lens
        left_center = (int(W * 0.35), int(H * 0.42))
        right_center = (int(W * 0.65), int(H * 0.42))
        axes = (int(W * 0.18), int(H * 0.12))

        cv2.ellipse(out, left_center, axes, 0, 0, 360, glass_color, -1)
        cv2.ellipse(out, right_center, axes, 0, 0, 360, glass_color, -1)
        # Bridge connecting lenses
        cv2.line(out, left_center, right_center, (30, 30, 35), 3)

    else:
        # Generic box cutout
        ch = np.random.randint(int(H * 0.2), int(H * 0.4))
        cw = np.random.randint(int(W * 0.2), int(W * 0.4))
        cy = np.random.randint(0, H - ch)
        cx = np.random.randint(0, W - cw)
        out[cy:cy + ch, cx:cx + cw] = np.random.randint(0, 255, (ch, cw, 3), dtype=np.uint8)

    return out


def apply_raking_directional_lighting_transform(
    image: np.ndarray,
    azimuth_deg: float = 75.0,
    elevation_deg: float = 15.0,
    ambient: float = 0.18,
    direct_strength: float = 1.25,
    color_temp_shift: float = 0.0,
) -> np.ndarray:
    """
    Simulates extreme 3D directional / raking illumination (side-lighting, harsh top-down
    streetlights, bottom-up horror/flashlight angles) and Planckian color temperature shift
    using a canonical 3D craniofacial surface normal model on the aligned 160x160 crop.

    Mathematical Formulation:
    1. Canonical 3D facial surface height map z(u, v) on u, v in [-1, 1]:
       Combines a general ellipsoidal cranial dome + prominent nasal bridge ridge +
       concave orbital eye sockets.
    2. Surface normal field n(u, v) = normalize(-dz/du, -dz/dv, 1).
    3. Incident light vector l(az, el) = (sin(az)cos(el), -sin(el), cos(az)cos(el)).
    4. Lambertian irradiance + cast nasal shadow:
       I_out(u, v) = I_in(u, v) * (ambient + direct_strength * max(0, n(u, v) . l) * shadow(u, v))
    5. Planckian chromaticity shift: scales (R, G, B) channels for warm (<0) or cool (>0) light.
    """
    H, W = image.shape[:2]
    v = np.linspace(-1.0, 1.0, H, dtype=np.float32).reshape(H, 1)
    u = np.linspace(-1.0, 1.0, W, dtype=np.float32).reshape(1, W)

    # Canonical craniofacial outward surface normals on 160x160 aligned convex face dome
    # For convex height z(u,v) = exp(-0.35*(u^2+v^2)), outward normal (-dz/du, -dz/dv, 1) has positive u, v signs
    nx = 0.85 * u * np.exp(-0.35 * (u ** 2 + v ** 2))
    ny = 0.75 * v * np.exp(-0.35 * (u ** 2 + v ** 2))

    # Nasal ridge normal contribution (centered at u=0, v in [-0.25, 0.25])
    nose_env = np.exp(-(u ** 2) / (2.0 * 0.14 ** 2) - ((v - 0.02) ** 2) / (2.0 * 0.24 ** 2))
    nx = nx + (u / (0.14 ** 2)) * 0.12 * nose_env

    # Orbital socket concavity around (u = +-0.32, v = -0.16)
    left_eye = np.exp(-((u + 0.32) ** 2 + (v + 0.16) ** 2) / (2.0 * 0.15 ** 2))
    right_eye = np.exp(-((u - 0.32) ** 2 + (v + 0.16) ** 2) / (2.0 * 0.15 ** 2))
    ny = ny - ((v + 0.16) / (0.15 ** 2)) * 0.05 * (left_eye + right_eye)

    nz = np.ones_like(nx)
    norm_len = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2 + 1e-8)
    nx /= norm_len
    ny /= norm_len
    nz /= norm_len

    # Light source unit vector from azimuth and elevation
    az_rad = math.radians(float(azimuth_deg))
    el_rad = math.radians(float(elevation_deg))
    lx = math.sin(az_rad) * math.cos(el_rad)
    ly = -math.sin(el_rad)
    lz = math.cos(az_rad) * math.cos(el_rad)

    cos_theta = np.maximum(0.0, nx * lx + ny * ly + nz * lz)

    # Cast nasal shadow on the contralateral cheek under extreme raking azimuth (|az| > 45 deg)
    shadow_mask = np.ones((H, W), dtype=np.float32)
    if abs(azimuth_deg) > 40.0:
        shadow_sign = -1.0 if azimuth_deg > 0 else 1.0
        # Contralateral side relative to light azimuth
        contra_u = np.maximum(0.0, shadow_sign * u)
        cast_len = min(0.45, 0.12 + 0.30 * (abs(azimuth_deg) / 90.0))
        nose_shadow = np.exp(
            -((contra_u - 0.18) ** 2) / (2.0 * (cast_len * 0.5) ** 2)
            - ((v - 0.04) ** 2) / (2.0 * 0.22 ** 2)
        ) * (contra_u > 0.02).astype(np.float32)
        shadow_mask = 1.0 - 0.65 * nose_shadow

    shading = float(np.clip(ambient, 0.02, 0.8)) + float(direct_strength) * cos_theta * shadow_mask
    shading_3ch = shading[:, :, np.newaxis]

    # Planckian illuminant chromaticity tint (color_temp_shift in [-0.35, 0.35])
    shift = float(np.clip(color_temp_shift, -0.4, 0.4))
    channel_gains = np.array(
        [1.0 - 0.55 * shift, 1.0 - 0.05 * abs(shift), 1.0 + 0.55 * shift],
        dtype=np.float32,
    ).reshape(1, 1, 3)

    out = image.astype(np.float32) * shading_3ch * channel_gains
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def apply_perspective_pose_foreshortening_transform(
    image: np.ndarray,
    yaw_deg: float = 20.0,
    pitch_deg: float = 12.0,
    roll_deg: float = 5.0,
) -> np.ndarray:
    """
    Simulates 3D out-of-plane head pose foreshortening (yaw, pitch, roll camera perspective)
    via a planar homography projection on the 160x160 canonical face plane.
    """
    H, W = image.shape[:2]
    src_pts = np.array(
        [[0.0, 0.0], [W - 1.0, 0.0], [W - 1.0, H - 1.0], [0.0, H - 1.0]],
        dtype=np.float32,
    )

    # Normalized perspective foreshortening offsets bounded to preserve face topology
    yaw_frac = float(np.clip(yaw_deg / 90.0, -0.35, 0.35)) * 0.22 * W
    pitch_frac = float(np.clip(pitch_deg / 90.0, -0.35, 0.35)) * 0.18 * H

    dst_pts = src_pts.copy()
    if yaw_frac >= 0:
        # Right hemiface foreshortened inward
        dst_pts[1, 0] -= yaw_frac
        dst_pts[2, 0] -= yaw_frac
        dst_pts[1, 1] += 0.35 * yaw_frac
        dst_pts[2, 1] -= 0.35 * yaw_frac
    else:
        # Left hemiface foreshortened inward
        ay = abs(yaw_frac)
        dst_pts[0, 0] += ay
        dst_pts[3, 0] += ay
        dst_pts[0, 1] += 0.35 * ay
        dst_pts[3, 1] -= 0.35 * ay

    if pitch_frac >= 0:
        # Top foreshortened
        dst_pts[0, 0] += 0.30 * pitch_frac
        dst_pts[1, 0] -= 0.30 * pitch_frac
        dst_pts[0, 1] += pitch_frac
        dst_pts[1, 1] += pitch_frac
    else:
        ap = abs(pitch_frac)
        dst_pts[3, 0] += 0.30 * ap
        dst_pts[2, 0] -= 0.30 * ap
        dst_pts[3, 1] -= ap
        dst_pts[2, 1] -= ap

    H_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        image,
        H_mat,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    if abs(roll_deg) > 0.5:
        rot_mat = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), float(roll_deg), 1.0)
        warped = cv2.warpAffine(
            warped,
            rot_mat,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    return warped


def apply_isp_degradation_and_compression_transform(
    image: np.ndarray,
    downscale_factor: float = 0.45,
    jpeg_quality: int = 22,
    chroma_noise_sigma: float = 6.0,
) -> np.ndarray:
    """
    Simulates edge-camera ISP degradation: distant low-resolution capture (downscale + upscale),
    chroma subsampling noise, and aggressive JPEG DCT block quantization artifacts.
    """
    H, W = image.shape[:2]
    scale = float(np.clip(downscale_factor, 0.20, 1.0))
    small_w = max(16, int(round(W * scale)))
    small_h = max(16, int(round(H * scale)))

    # 1. Optical resolution loss via area downsampling + bilinear upsampling
    down = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (W, H), interpolation=cv2.INTER_LINEAR)

    # 2. Sensor chroma noise
    if chroma_noise_sigma > 0.0:
        noise = np.random.normal(0.0, float(chroma_noise_sigma), up.shape).astype(np.float32)
        up = np.clip(up.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

    # 3. JPEG DCT 8x8 quantization artifacts
    q = int(np.clip(jpeg_quality, 8, 95))
    ok, enc = cv2.imencode(".jpg", up, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if dec is not None and dec.shape == image.shape:
            return dec
    return up


class PhysicalEnvironmentalAugmenter:
    """
    Configurable physical environmental augmentation pipeline for training data.
    Simulates overexposure, underexposure, 3D raking directional lighting, perspective
    pose foreshortening, motion blur, ISP/JPEG degradation, and occlusion stochastically.
    """

    def __init__(
        self,
        p_overexposed: float = 0.25,
        p_underexposed: float = 0.25,
        p_blur: float = 0.25,
        p_cutout: float = 0.25,
        p_raking_light: float = 0.25,
        p_pose_warp: float = 0.20,
        p_isp_degrade: float = 0.20,
    ):
        self.p_overexposed = p_overexposed
        self.p_underexposed = p_underexposed
        self.p_blur = p_blur
        self.p_cutout = p_cutout
        self.p_raking_light = p_raking_light
        self.p_pose_warp = p_pose_warp
        self.p_isp_degrade = p_isp_degrade

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Applies stochastic environmental physical augmentations.
        
        Args:
            image: (160, 160, 3) uint8 image
        Returns:
            Transformed (160, 160, 3) uint8 image
        """
        out = image.copy()

        # 1. 3D Pose perspective foreshortening
        if random.random() < self.p_pose_warp:
            yaw = random.uniform(-24.0, 24.0)
            pitch = random.uniform(-15.0, 15.0)
            roll = random.uniform(-8.0, 8.0)
            out = apply_perspective_pose_foreshortening_transform(
                out, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll
            )

        # 2. Photometric / Lighting transform
        r = random.random()
        if r < self.p_overexposed:
            gamma = random.uniform(0.4, 0.7)
            out = apply_overexposure_transform(out, gamma=gamma)
        elif r < self.p_overexposed + self.p_underexposed:
            gamma = random.uniform(1.8, 2.5)
            out = apply_underexposure_backlight_transform(out, gamma=gamma)
        elif r < self.p_overexposed + self.p_underexposed + self.p_raking_light:
            az = random.choice([random.uniform(-85.0, -45.0), random.uniform(45.0, 85.0)])
            el = random.uniform(-35.0, 45.0)
            amb = random.uniform(0.10, 0.28)
            ct = random.uniform(-0.28, 0.28)
            out = apply_raking_directional_lighting_transform(
                out, azimuth_deg=az, elevation_deg=el, ambient=amb, color_temp_shift=ct
            )

        # 3. Motion blur
        if random.random() < self.p_blur:
            k = random.choice([3, 5, 7, 9, 11])
            angle = random.uniform(0.0, 180.0)
            out = apply_motion_blur_transform(out, kernel_size=k, angle_deg=angle)

        # 4. ISP / JPEG compression degradation
        if random.random() < self.p_isp_degrade:
            scale = random.uniform(0.38, 0.70)
            jq = random.randint(15, 40)
            out = apply_isp_degradation_and_compression_transform(
                out, downscale_factor=scale, jpeg_quality=jq, chroma_noise_sigma=random.uniform(2.0, 7.0)
            )

        # 5. Occlusion
        if random.random() < self.p_cutout:
            occ_type = random.choice(["mask", "sunglasses"])
            out = apply_cutout_occlusion_transform(out, occlusion_type=occ_type)

        return out


class ExtremeNoiseStressBenchmark:
    """
    Deterministic multi-regime stress-test benchmark for evaluating biometric model
    accuracy and smoothness under extreme environmental corruptions:
      1. 'clean': Uncorrupted canonical aligned face
      2. 'raking_directional_light': Harsh 80-deg azimuth raking light + cast shadow + warm shift
      3. 'extreme_overexposure_glare': Gamma 0.42 + intense specular highlight bloom
      4. 'extreme_lowlight_sensor_noise': Gamma 2.45 + backlight silhouette + Poisson-Gaussian noise
      5. 'motion_blur_psf': 9px directional motion blur + 18-deg yaw perspective foreshortening
      6. 'partial_occlusion': Lower-face surgical mask or sunglasses occlusion
      7. 'extreme_compound_storm': Combined raking light + motion blur + ISP/JPEG compression
    """

    REGIMES = (
        "clean",
        "raking_directional_light",
        "extreme_overexposure_glare",
        "extreme_lowlight_sensor_noise",
        "motion_blur_psf",
        "partial_occlusion",
        "extreme_compound_storm",
    )

    @classmethod
    def apply_regime_uint8(cls, image_rgb: np.ndarray, regime: str, seed: int = 42) -> np.ndarray:
        """Applies a named deterministic corruption regime to a (H, W, 3) uint8 RGB image."""
        if regime == "clean":
            return image_rgb.copy()

        H, W = image_rgb.shape[:2]
        rng_state = np.random.get_state()
        py_state = random.getstate()
        try:
            np.random.seed(seed)
            random.seed(seed)

            if regime == "raking_directional_light":
                az = 78.0 if (seed % 2 == 0) else -78.0
                el = 22.0 if (seed % 3 != 0) else -20.0
                return apply_raking_directional_lighting_transform(
                    image_rgb,
                    azimuth_deg=az,
                    elevation_deg=el,
                    ambient=0.14,
                    direct_strength=1.35,
                    color_temp_shift=0.22 if (seed % 2 == 0) else -0.22,
                )
            if regime == "extreme_overexposure_glare":
                return apply_overexposure_transform(
                    image_rgb,
                    gamma=0.45,
                    flare_strength=0.65,
                    flare_center=(int(W * 0.55), int(H * 0.35)),
                    is_bgr=False,
                )
            if regime == "extreme_lowlight_sensor_noise":
                return apply_underexposure_backlight_transform(
                    image_rgb,
                    gamma=2.35,
                    noise_sigma=14.0,
                    gradient_dir="horizontal",
                )
            if regime == "motion_blur_psf":
                warped = apply_perspective_pose_foreshortening_transform(
                    image_rgb, yaw_deg=18.0, pitch_deg=10.0, roll_deg=4.0
                )
                return apply_motion_blur_transform(warped, kernel_size=9, angle_deg=35.0)
            if regime == "partial_occlusion":
                occ = "mask" if (seed % 2 == 0) else "sunglasses"
                return apply_cutout_occlusion_transform(image_rgb, occlusion_type=occ)
            if regime == "extreme_compound_storm":
                lit = apply_raking_directional_lighting_transform(
                    image_rgb,
                    azimuth_deg=-72.0,
                    elevation_deg=18.0,
                    ambient=0.16,
                    direct_strength=1.28,
                    color_temp_shift=-0.18,
                )
                blurred = apply_motion_blur_transform(lit, kernel_size=7, angle_deg=25.0)
                return apply_isp_degradation_and_compression_transform(
                    blurred, downscale_factor=0.48, jpeg_quality=24, chroma_noise_sigma=5.0
                )
            raise ValueError(f"Unknown corruption regime: {regime}")
        finally:
            np.random.set_state(rng_state)
            random.setstate(py_state)

