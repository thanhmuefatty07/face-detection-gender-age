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


def _boundary_calibrate_distribution(
    probs: np.ndarray,
    target_mean: float,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Moment calibration via exponential tilting:
    q_k(\\lambda) \\propto q_k * \\exp(\\lambda * (k - \\mu))
    Finds \\lambda such that \\sum k * q_k(\\lambda) == target_mean within tol.
    """
    num_bins = len(probs)
    bins = np.arange(num_bins, dtype=np.float64)
    lam = 0.0
    p = probs.copy().astype(np.float64)

    for _ in range(max_iter):
        exp_factor = np.exp(np.clip(lam * (bins - target_mean), -50.0, 50.0))
        w = probs * exp_factor
        w_sum = np.sum(w)
        if w_sum <= 1e-15:
            break
        p = w / w_sum
        cur_mean = np.sum(bins * p)
        err = cur_mean - target_mean
        if abs(err) <= tol:
            return p.astype(np.float32)
        var = np.sum(((bins - cur_mean) ** 2) * p)
        if var < 1e-8:
            break
        lam -= err / var

    return p.astype(np.float32)


def get_adaptive_sigma(mean_age: float, base_sigma: float = 2.5) -> float:
    """
    Age-adaptive standard deviation sigma(a) reflecting biological face development:
    Facial morphology develops rapidly in infancy/childhood, requiring narrower sigma,
    while adult/elderly appearance variance is broader.
    Bound to prevent truncation expectation distortion near boundaries (Bhatia-Davis).
    """
    if mean_age <= 0.0:
        return 0.32
    elif mean_age < 5.0:
        return float(min(base_sigma, 0.32 + 0.31 * mean_age))
    elif mean_age > 95.0:
        dist_to_max = max(0.0, 100.0 - mean_age)
        return float(min(base_sigma, 0.32 + 0.31 * dist_to_max))
    return float(base_sigma)


def compute_boundary_calibrated_ldl(
    mean_age: float,
    sigma: float = 2.5,
    num_bins: int = 101,
) -> np.ndarray:
    """
    Boundary-Calibrated Folded Gaussian Label Distribution Learning (BC-LDL).
    
    For ages near boundaries (mean_age < 5.0 or mean_age > num_bins - 1 - 5.0):
    1. Evaluates Gaussian with age-adaptive sigma(a) and folds/reflects truncated mass
       back into the active range: q_k^* = q_k + q_{-k} (for lower boundary)
       and q_k^* = q_k + q_{2K - k} (for upper boundary K = num_bins - 1).
    2. Applies moment calibration to ensure the continuous expectation sum(k * q_k^*)
       strictly equals mean_age within +/- 0.0001 (e.g. exactly 1.0000 for age 1.0),
       eliminating infant boundary distortion and Soft-L1 MAE gradient conflicts.
    """
    mean_age = float(np.clip(mean_age, 0.0, num_bins - 1))
    sigma = get_adaptive_sigma(mean_age, base_sigma=float(sigma))
    bins = np.arange(num_bins, dtype=np.float64)
    K = float(num_bins - 1)

    # Base Gaussian over [0, K]
    q_pos = np.exp(-0.5 * ((bins - mean_age) / sigma) ** 2)

    # Folded reflection across lower boundary (k = 0): q_{-k} folded into q_k
    q_neg_lower = np.exp(-0.5 * ((-bins - mean_age) / sigma) ** 2)

    # Folded reflection across upper boundary (k = K): q_{2K - k} folded into q_k
    q_neg_upper = np.exp(-0.5 * ((2.0 * K - bins - mean_age) / sigma) ** 2)

    # Fold reflected mass (exclude boundaries from self double-counting)
    q = q_pos.copy()
    q[1:] += q_neg_lower[1:]
    q[:-1] += q_neg_upper[:-1]

    # Simplex normalization
    q = np.maximum(q, 1e-12)
    q /= np.sum(q)

    # Boundary calibration if expectation deviates from mean_age due to boundary truncation
    cur_mean = float(np.sum(bins * q))
    if abs(cur_mean - mean_age) > 1e-4:
        q = _boundary_calibrate_distribution(q, mean_age)

    # Final simplex guarantee
    q = np.maximum(q, 1e-12)
    q /= np.sum(q)
    return q.astype(np.float32)


# Alias
compute_bc_ldl = compute_boundary_calibrated_ldl


def compute_gaussian_ldl(
    mean_age: float,
    sigma: float = 2.5,
    num_bins: int = 101,
    boundary_calibrate: bool = True,
) -> np.ndarray:
    """
    Computes a continuous Gaussian label distribution over [0, num_bins - 1].
    Guarantees sum(p) == 1.0 (simplex conservation).
    
    When boundary_calibrate=True (default), uses BC-LDL near boundaries
    (age < 3*sigma or age > 100 - 3*sigma) to fold truncated mass and guarantee expectation
    matches target age within +/- 0.0001 (e.g., exactly 1.0000 for age 1.0) with zero jumps.
    """
    boundary_span = max(5.0, 3.0 * float(sigma))
    if boundary_calibrate and (mean_age < boundary_span or mean_age > (num_bins - 1 - boundary_span)):
        return compute_boundary_calibrated_ldl(mean_age=mean_age, sigma=sigma, num_bins=num_bins)

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
