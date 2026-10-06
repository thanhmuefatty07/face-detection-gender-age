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
    lab = cv2.cvtColor(img_flare, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

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


class PhysicalEnvironmentalAugmenter:
    """
    Configurable physical environmental augmentation pipeline for training data.
    Simulates overexposure, underexposure, motion blur, and occlusion stochastically.
    """

    def __init__(
        self,
        p_overexposed: float = 0.25,
        p_underexposed: float = 0.25,
        p_blur: float = 0.25,
        p_cutout: float = 0.25,
    ):
        self.p_overexposed = p_overexposed
        self.p_underexposed = p_underexposed
        self.p_blur = p_blur
        self.p_cutout = p_cutout

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Applies stochastic environmental physical augmentations.
        
        Args:
            image: (160, 160, 3) uint8 image
        Returns:
            Transformed (160, 160, 3) uint8 image
        """
        out = image.copy()

        # Lighting transform (mutually exclusive or sequential)
        r = random.random()
        if r < self.p_overexposed:
            gamma = random.uniform(0.4, 0.7)
            out = apply_overexposure_transform(out, gamma=gamma)
        elif r < self.p_overexposed + self.p_underexposed:
            gamma = random.uniform(1.8, 2.5)
            out = apply_underexposure_backlight_transform(out, gamma=gamma)

        # Motion blur
        if random.random() < self.p_blur:
            k = random.choice([3, 5, 7, 9, 11])
            angle = random.uniform(0.0, 180.0)
            out = apply_motion_blur_transform(out, kernel_size=k, angle_deg=angle)

        # Occlusion
        if random.random() < self.p_cutout:
            occ_type = random.choice(["mask", "sunglasses"])
            out = apply_cutout_occlusion_transform(out, occlusion_type=occ_type)

        return out
