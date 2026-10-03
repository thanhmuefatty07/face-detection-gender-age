"""
Unit Tests for Continuous LDL & Sexual Dimorphism
=================================================
Tests:
- Continuous Gaussian and Laplacian LDL simplex conservation (sum(p) == 1.0)
- Expected age expectation mu = sum(i * p_i) matches ground truth
- Wasserstein-1 / EMD loss behavior
- Sexual dimorphism transitions across age spectrum
"""

from __future__ import annotations
import math
import pytest
import numpy as np

from src.biometrics.estimation.ldl_engine import (
    ContinuousLDLEngine,
    compute_gaussian_ldl,
    compute_laplacian_ldl,
    wasserstein_emd_loss
)
from src.biometrics.estimation.dimorphism import (
    calculate_sexual_dimorphism_weight,
    calculate_ethnic_aging_offset,
    get_cohort_for_age,
)


class TestContinuousLDLEngine:
    """Verifies LDL mathematical properties."""

    @pytest.mark.parametrize("age", [0.0, 5.0, 12.0, 25.0, 50.0, 75.0, 99.0])
    def test_gaussian_ldl_conserves_simplex_probability(self, age):
        """Sum of probabilities must equal 1.0 +/- 1e-4 for any continuous age."""
        dist = compute_gaussian_ldl(mean_age=age, sigma=2.5, num_bins=101)
        assert len(dist) == 101
        assert np.isclose(np.sum(dist), 1.0, atol=1e-4)
        assert np.all(dist >= 0.0)

    @pytest.mark.parametrize("age", [10.0, 30.0, 60.0, 80.0])
    def test_laplacian_ldl_conserves_simplex_probability(self, age):
        """Laplacian distribution sums to 1.0 +/- 1e-4."""
        dist = compute_laplacian_ldl(mean_age=age, scale_b=2.0, num_bins=101)
        assert np.isclose(np.sum(dist), 1.0, atol=1e-4)

    def test_expected_age_recovers_ground_truth(self):
        """Expected age mu matches injected ground truth within +/- 0.5 years for interior ages."""
        engine = ContinuousLDLEngine(num_bins=101)
        for target_age in [20.0, 35.0, 50.0, 65.0]:
            dist = compute_gaussian_ldl(mean_age=target_age, sigma=2.0)
            mu = engine.expected_age(dist)
            assert abs(mu - target_age) < 0.2

    def test_wasserstein_emd_loss_properties(self):
        """Identical distributions have zero EMD; separated distributions have positive EMD."""
        d1 = compute_gaussian_ldl(mean_age=30.0, sigma=2.0)
        d2 = compute_gaussian_ldl(mean_age=30.0, sigma=2.0)
        d3 = compute_gaussian_ldl(mean_age=40.0, sigma=2.0)

        assert np.isclose(wasserstein_emd_loss(d1, d2), 0.0, atol=1e-5)
        loss_sep = wasserstein_emd_loss(d1, d3)
        assert loss_sep > 0.0


class TestSexualDimorphismAndEthnicKinetics:
    """Verifies biological dimorphism attenuation and ethnic aging kinetics."""

    def test_prepubertal_children_dimorphism_attenuation(self):
        """Infants have w=0.25, linear ramp to 1.0 at age 12."""
        assert calculate_sexual_dimorphism_weight(0.0) == 0.25
        assert calculate_sexual_dimorphism_weight(6.0) == 0.625
        assert calculate_sexual_dimorphism_weight(12.0) == 1.0

    def test_adult_dimorphism_plateau(self):
        """Adults [12, 65] have full dimorphism weight w=1.0."""
        for a in [15.0, 25.0, 40.0, 65.0]:
            assert calculate_sexual_dimorphism_weight(a) == 1.0

    def test_geriatric_dimorphism_attenuation(self):
        """Geriatric faces >65 have w=0.85."""
        assert calculate_sexual_dimorphism_weight(70.0) == 0.85
        assert calculate_sexual_dimorphism_weight(85.0) == 0.85

    def test_cohort_categorization(self):
        """Verifies correct age cohort naming."""
        assert get_cohort_for_age(1.0)[0] == '0-2'
        assert get_cohort_for_age(15.0)[0] == '13-17'
        assert get_cohort_for_age(28.0)[0] == '25-34'
        assert get_cohort_for_age(75.0)[0] == '65+'
