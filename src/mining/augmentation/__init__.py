"""
Mining Augmentation Package
===========================
Physical environmental transforms and robust dataset augmentations.
"""

from .environmental_transforms import (
    apply_overexposure_transform,
    apply_underexposure_backlight_transform,
    apply_motion_blur_transform,
    apply_cutout_occlusion_transform,
    PhysicalEnvironmentalAugmenter,
)

__all__ = [
    "apply_overexposure_transform",
    "apply_underexposure_backlight_transform",
    "apply_motion_blur_transform",
    "apply_cutout_occlusion_transform",
    "PhysicalEnvironmentalAugmenter",
]
