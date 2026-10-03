"""
Continuous Label Distribution Learning (LDL) & Earth Mover's Distance (EMD)
============================================================================
Formulates age as a continuous probability distribution across 101 discrete bins [0, 100].
Penalizes probability divergence via Wasserstein-1 / EMD loss to respect chronological distance.
"""

from __future__ import annotations
import math
from typing import Tuple
import numpy as np


def compute_gaussian_ldl(mean_age: float, sigma: float = 2.5, num_bins: int = 101) -> np.ndarray:
    """
    Computes a continuous Gaussian label distribution over [0, num_bins - 1].
    Guarantees sum(p) == 1.0 (simplex conservation).
    """
    sigma = max(sigma, 0.5)
    bins = np.arange(num_bins, dtype=np.float64)

    # CDF-based integration using math.erf for exact normalization
    sqrt2 = math.sqrt(2.0)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)((bins + 0.5 - mean_age) / (sigma * sqrt2)))
    cdf_prev = 0.5 * (1.0 + np.vectorize(math.erf)((bins - 0.5 - mean_age) / (sigma * sqrt2)))

    probs = cdf - cdf_prev
    probs = np.maximum(probs, 1e-12)
    probs /= np.sum(probs)
    return probs.astype(np.float32)


def compute_laplacian_ldl(mean_age: float, scale_b: float = 2.0, num_bins: int = 101) -> np.ndarray:
    """
    Computes a continuous Laplacian label distribution over [0, num_bins - 1].
    Captures heavier biological tail variance for geriatric populations.
    """
    scale_b = max(scale_b, 0.5)
    bins = np.arange(num_bins, dtype=np.float64)
    probs = np.exp(-np.abs(bins - mean_age) / scale_b)
    probs = np.maximum(probs, 1e-12)
    probs /= np.sum(probs)
    return probs.astype(np.float32)


def wasserstein_emd_loss(pred_dist: np.ndarray, target_dist: np.ndarray) -> float:
    """
    Computes 1D Earth Mover's Distance (Wasserstein-1) between predicted and target distributions.
    W_1(P, Q) = sum | CDF_P(k) - CDF_Q(k) | / K
    """
    cdf_pred = np.cumsum(pred_dist)
    cdf_target = np.cumsum(target_dist)
    return float(np.mean(np.abs(cdf_pred - cdf_target)))


class ContinuousLDLEngine:
    """
    Inference and conversion engine for Continuous Label Distribution Learning.
    """

    def __init__(self, num_bins: int = 101):
        self.num_bins = num_bins
        self.bin_indices = np.arange(num_bins, dtype=np.float32)

    def expected_age(self, dist: np.ndarray) -> float:
        """Computes expected age mu = sum(i * p_i)."""
        dist_norm = dist / np.sum(dist)
        return float(np.sum(self.bin_indices * dist_norm))

    def variance(self, dist: np.ndarray) -> float:
        """Computes apparent age distribution variance sigma^2 = sum((i - mu)^2 * p_i)."""
        mu = self.expected_age(dist)
        dist_norm = dist / np.sum(dist)
        return float(np.sum(((self.bin_indices - mu) ** 2) * dist_norm))

    def std_dev(self, dist: np.ndarray) -> float:
        """Computes standard deviation."""
        return float(math.sqrt(max(self.variance(dist), 1e-6)))

    def integrate_to_10_cohorts(self, dist: np.ndarray) -> np.ndarray:
        """
        Integrates 101 continuous bins into the standard 10 biometric cohorts:
        ['0-2', '3-6', '7-12', '13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        """
        from .dimorphism import AGE_COHORTS
        cohort_probs = np.zeros(len(AGE_COHORTS), dtype=np.float32)
        for idx, (_, low, high) in enumerate(AGE_COHORTS):
            high_bound = min(high, self.num_bins - 1)
            cohort_probs[idx] = np.sum(dist[low:high_bound + 1])
        cohort_probs /= np.sum(cohort_probs)
        return cohort_probs
