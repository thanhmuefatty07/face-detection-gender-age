"""
Unit Tests for Physical Environmental Augmentations
===================================================
Tests physical transforms for 160x160 aligned face crops:
- Overexposure (Gamma compression + Specular highlight flares + CLAHE)
- Underexposure & Backlight (Gamma expansion + Poisson-Gaussian noise + Directional gradient)
- Motion Blur (Point Spread Function kernel 3-11 px)
- Cutout Occlusion (Mouth mask, Sunglasses)
- PhysicalEnvironmentalAugmenter pipeline
"""

from __future__ import annotations
import cv2
import numpy as np
import pytest

from src.mining.augmentation.environmental_transforms import (
    apply_overexposure_transform,
    apply_underexposure_backlight_transform,
    apply_motion_blur_transform,
    apply_cutout_occlusion_transform,
    PhysicalEnvironmentalAugmenter,
)


@pytest.fixture
def base_face_image():
    """Returns a synthetic 160x160x3 sharp face image."""
    img = np.full((160, 160, 3), 120, dtype=np.uint8)
    # Eyes
    cv2.circle(img, (55, 65), 10, (20, 20, 20), -1)
    cv2.circle(img, (105, 65), 10, (20, 20, 20), -1)
    # Nose
    cv2.line(img, (80, 75), (80, 100), (40, 40, 40), 2)
    # Mouth
    cv2.ellipse(img, (80, 125), (20, 6), 0, 0, 360, (50, 40, 180), -1)
    return img


class TestEnvironmentalTransforms:
    def test_overexposure_transform(self, base_face_image):
        augmented = apply_overexposure_transform(base_face_image, gamma=0.5, flare_strength=0.7)
        assert augmented.shape == (160, 160, 3)
        assert augmented.dtype == np.uint8
        # Overexposed image should have higher average intensity
        assert augmented.mean() > base_face_image.mean()
        # Flare should create saturated highlights (255)
        assert (augmented >= 250).sum() > 0

    def test_overexposure_transform_bgr(self, base_face_image):
        augmented_bgr = apply_overexposure_transform(base_face_image, gamma=0.5, flare_strength=0.7, is_bgr=True)
        assert augmented_bgr.shape == (160, 160, 3)
        assert augmented_bgr.dtype == np.uint8
        assert augmented_bgr.mean() > base_face_image.mean()

    def test_underexposure_backlight_transform(self, base_face_image):
        augmented = apply_underexposure_backlight_transform(
            base_face_image, gamma=2.2, noise_sigma=10.0
        )
        assert augmented.shape == (160, 160, 3)
        assert augmented.dtype == np.uint8
        # Underexposed image should have lower average intensity
        assert augmented.mean() < base_face_image.mean()

    @pytest.mark.parametrize("kernel_size", [3, 7, 11])
    def test_motion_blur_transform(self, base_face_image, kernel_size):
        blurred = apply_motion_blur_transform(base_face_image, kernel_size=kernel_size, angle_deg=45.0)
        assert blurred.shape == (160, 160, 3)
        assert blurred.dtype == np.uint8
        # Laplacian variance measures sharpness - blur must reduce sharpness
        lap_orig = cv2.Laplacian(base_face_image, cv2.CV_64F).var()
        lap_blur = cv2.Laplacian(blurred, cv2.CV_64F).var()
        assert lap_blur < lap_orig

    def test_cutout_mask_occlusion(self, base_face_image):
        occluded = apply_cutout_occlusion_transform(base_face_image, occlusion_type="mask")
        assert occluded.shape == (160, 160, 3)
        assert occluded.dtype == np.uint8
        # Lower face (mouth region: y ~ 110-130, x ~ 70-90) should be modified
        mouth_region_orig = base_face_image[110:130, 70:90]
        mouth_region_occ = occluded[110:130, 70:90]
        assert not np.array_equal(mouth_region_orig, mouth_region_occ)
        # Upper face (eyes: y ~ 65, x ~ 55) should remain intact
        assert np.array_equal(base_face_image[60:70, 50:60], occluded[60:70, 50:60])

    def test_cutout_sunglasses_occlusion(self, base_face_image):
        occluded = apply_cutout_occlusion_transform(base_face_image, occlusion_type="sunglasses")
        assert occluded.shape == (160, 160, 3)
        assert occluded.dtype == np.uint8
        # Eye region should be modified
        eyes_orig = base_face_image[60:75, 45:115]
        eyes_occ = occluded[60:75, 45:115]
        assert not np.array_equal(eyes_orig, eyes_occ)

    def test_physical_augmenter_pipeline(self, base_face_image):
        augmenter = PhysicalEnvironmentalAugmenter(p_overexposed=0.5, p_underexposed=0.5, p_blur=0.5, p_cutout=0.5)
        out = augmenter(base_face_image)
        assert out.shape == (160, 160, 3)
        assert out.dtype == np.uint8

    def test_raking_directional_lighting_and_planckian_shift(self, base_face_image):
        from src.mining.augmentation.environmental_transforms import apply_raking_directional_lighting_transform

        # Positive azimuth (+80 deg) illuminates right hemiface and casts shadow on left hemiface
        lit_right = apply_raking_directional_lighting_transform(
            base_face_image, azimuth_deg=80.0, elevation_deg=15.0, ambient=0.12, direct_strength=1.35, color_temp_shift=0.25
        )
        assert lit_right.shape == (160, 160, 3)
        assert lit_right.dtype == np.uint8
        # Right half (x > 85) should have higher mean brightness than shadowed left half (x < 75)
        assert lit_right[:, 90:145, :].mean() > lit_right[:, 15:70, :].mean()

    def test_perspective_pose_foreshortening_and_isp_degradation(self, base_face_image):
        from src.mining.augmentation.environmental_transforms import (
            apply_perspective_pose_foreshortening_transform,
            apply_isp_degradation_and_compression_transform,
        )

        warped = apply_perspective_pose_foreshortening_transform(
            base_face_image, yaw_deg=22.0, pitch_deg=-12.0, roll_deg=6.0
        )
        assert warped.shape == (160, 160, 3)
        assert warped.dtype == np.uint8
        assert not np.array_equal(warped, base_face_image)

        degraded = apply_isp_degradation_and_compression_transform(
            base_face_image, downscale_factor=0.45, jpeg_quality=20, chroma_noise_sigma=5.0
        )
        assert degraded.shape == (160, 160, 3)
        assert degraded.dtype == np.uint8
        assert not np.array_equal(degraded, base_face_image)

    def test_extreme_noise_stress_benchmark_all_7_regimes(self, base_face_image):
        from src.mining.augmentation.environmental_transforms import ExtremeNoiseStressBenchmark

        assert len(ExtremeNoiseStressBenchmark.REGIMES) == 7
        clean_out = ExtremeNoiseStressBenchmark.apply_regime_uint8(base_face_image, "clean", seed=123)
        assert np.array_equal(clean_out, base_face_image)

        for regime in ExtremeNoiseStressBenchmark.REGIMES[1:]:
            out1 = ExtremeNoiseStressBenchmark.apply_regime_uint8(base_face_image, regime, seed=123)
            out2 = ExtremeNoiseStressBenchmark.apply_regime_uint8(base_face_image, regime, seed=123)
            assert out1.shape == (160, 160, 3)
            assert out1.dtype == np.uint8
            assert np.array_equal(out1, out2)
            assert not np.array_equal(out1, base_face_image)

