"""
Biometric Estimation Module: Continuous LDL, Sexual Dimorphism & Backbones
"""

from .dimorphism import (
    calculate_sexual_dimorphism_weight,
    calculate_ethnic_aging_offset,
    AGE_COHORTS,
    get_cohort_for_age,
)
from .ldl_engine import (
    ContinuousLDLEngine,
    compute_gaussian_ldl,
    compute_laplacian_ldl,
    wasserstein_emd_loss,
)
from .backbone import (
    BiometricMultiTaskModel,
    ConvNeXtV2BiometricHead,
    MobileNetV4BiometricHead,
)

__all__ = [
    "calculate_sexual_dimorphism_weight",
    "calculate_ethnic_aging_offset",
    "AGE_COHORTS",
    "get_cohort_for_age",
    "ContinuousLDLEngine",
    "compute_gaussian_ldl",
    "compute_laplacian_ldl",
    "wasserstein_emd_loss",
    "BiometricMultiTaskModel",
    "ConvNeXtV2BiometricHead",
    "MobileNetV4BiometricHead",
]
