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
    get_adaptive_sigma,
    compute_gaussian_ldl,
    compute_boundary_calibrated_ldl,
    compute_bc_ldl,
    compute_laplacian_ldl,
    wasserstein_emd_loss,
)
from .backbone import (
    BiometricMultiTaskModel,
    ConvNeXtV2BiometricModel,
    MultiScaleBiometricFusion,
    CoordinateAttention,
    GlobalResponseNorm1d,
    GlobalResponseNorm2d,
    PhotometricRetinexStem,
    ConvNeXtV2BiometricHead,
    MobileNetV4BiometricHead,
)
from .dataset import (
    FaceGoldenDataset,
)
from .losses import (
    cramer_distance_loss,
    soft_kl_divergence_loss,
    soft_l1_mae_loss,
    interval_censored_ordinal_loss,
    fine_interval_ldl_loss,
    ldl_tikhonov_smoothness_loss,
    photometric_consistency_loss,
    masked_dimorphic_bce_loss,
    aleatoric_heteroscedastic_loss,
    CoupledHeteroscedasticLaplaceLoss,
    SupervisedContrastiveLoss,
    UnifiedBiometricLoss,
)

__all__ = [
    "calculate_sexual_dimorphism_weight",
    "calculate_ethnic_aging_offset",
    "AGE_COHORTS",
    "get_cohort_for_age",
    "ContinuousLDLEngine",
    "get_adaptive_sigma",
    "compute_gaussian_ldl",
    "compute_boundary_calibrated_ldl",
    "compute_bc_ldl",
    "compute_laplacian_ldl",
    "wasserstein_emd_loss",
    "BiometricMultiTaskModel",
    "ConvNeXtV2BiometricModel",
    "MultiScaleBiometricFusion",
    "CoordinateAttention",
    "GlobalResponseNorm1d",
    "GlobalResponseNorm2d",
    "PhotometricRetinexStem",
    "ConvNeXtV2BiometricHead",
    "MobileNetV4BiometricHead",
    "FaceGoldenDataset",
    "cramer_distance_loss",
    "soft_kl_divergence_loss",
    "soft_l1_mae_loss",
    "interval_censored_ordinal_loss",
    "fine_interval_ldl_loss",
    "ldl_tikhonov_smoothness_loss",
    "photometric_consistency_loss",
    "masked_dimorphic_bce_loss",
    "aleatoric_heteroscedastic_loss",
    "CoupledHeteroscedasticLaplaceLoss",
    "SupervisedContrastiveLoss",
    "UnifiedBiometricLoss",
]

