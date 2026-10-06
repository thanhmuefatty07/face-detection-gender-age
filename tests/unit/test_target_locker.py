"""
Unit Tests for Target Locking (Epistemic-Kalman & Leaky Bayesian Gender Accumulator)
====================================================================================
Tests zero-flicker age smoothing and anti-flip gender locking:
1. Epistemic-Kalman age smoothing with dynamic R = sigma^2
2. Bounded Leaky Bayesian gender accumulator with gamma=0.95 clamped [-6.0, 6.0]
3. Multi-target lock manager (TargetLocker)
"""

from __future__ import annotations
import pytest
import numpy as np

from src.biometrics.tracking.target_locker import (
    EpistemicAgeKalmanFilter,
    BoundedLeakyGenderAccumulator,
    TargetLocker,
)


class TestEpistemicAgeKalmanFilter:
    def test_filter_smooths_noisy_measurements(self):
        kf = EpistemicAgeKalmanFilter(initial_age=30.0, initial_std=1.0)
        
        # Feed high-uncertainty noisy measurements (e.g. sudden spike to 50 with sigma=10)
        smoothed_age = kf.update(measured_age=50.0, aleatoric_std=10.0)
        # Because sigma=10 (R=100), the filter should reject the large spike
        assert abs(smoothed_age - 30.0) < 3.0

        # Feed confident measurement (e.g. 31 with sigma=0.5 -> R=0.25)
        smoothed_age = kf.update(measured_age=31.0, aleatoric_std=0.5)
        assert abs(smoothed_age - 31.0) < 1.0

    def test_zero_flicker_stability(self):
        """Simulate 20 frames of slight age fluctuations around 25.0 years."""
        kf = EpistemicAgeKalmanFilter(initial_age=25.0, initial_std=1.0)
        outputs = []
        rng = np.random.RandomState(42)
        for _ in range(20):
            noisy_meas = 25.0 + rng.normal(0, 1.5)
            out = kf.update(noisy_meas, aleatoric_std=1.5)
            outputs.append(out)

        # Standard deviation of smoothed output should be significantly lower than raw noise
        assert np.std(outputs) < 0.6


class TestBoundedLeakyGenderAccumulator:
    def test_accumulation_and_clamping(self):
        acc = BoundedLeakyGenderAccumulator(decay=0.95, min_bound=-6.0, max_bound=6.0)

        # Feed strong Female logits (+3.0) repeatedly
        for _ in range(10):
            acc.update(gender_logit=3.0)

        assert acc.log_odds <= 6.0
        assert acc.log_odds > 5.5
        assert acc.probability > 0.99
        assert acc.label == "Female"

    def test_anti_flip_noise_resistance(self):
        """A well-established Female subject should NOT flip from a single corrupted/noisy frame."""
        acc = BoundedLeakyGenderAccumulator(decay=0.95, min_bound=-6.0, max_bound=6.0)

        # Establish strong Female confidence
        for _ in range(5):
            acc.update(gender_logit=2.5)

        assert acc.label == "Female"

        # One sudden noisy corrupted frame predicting Male (-3.0)
        acc.update(gender_logit=-3.0)

        # Must still maintain Female label!
        assert acc.label == "Female"
        assert acc.log_odds > 0.0

    def test_leaky_decay_on_neutral_input(self):
        acc = BoundedLeakyGenderAccumulator(decay=0.95, min_bound=-6.0, max_bound=6.0)
        acc.update(gender_logit=4.0)
        val_before = acc.log_odds

        # Neutral evidence (logit = 0.0) decays state by 0.95
        acc.update(gender_logit=0.0)
        assert pytest.approx(acc.log_odds, rel=1e-3) == val_before * 0.95


class TestTargetLocker:
    def test_multi_target_locking(self):
        locker = TargetLocker()

        # Target 1: 25yo Female
        # Target 2: 50yo Male
        for _ in range(5):
            res1 = locker.update(track_id=1, age_pred=25.0, aleatoric_std=1.0, gender_logit=2.0)
            res2 = locker.update(track_id=2, age_pred=50.0, aleatoric_std=1.0, gender_logit=-2.0)

        assert pytest.approx(res1["locked_age"], abs=1.0) == 25.0
        assert res1["locked_gender"] == "Female"
        assert res1["gender_prob"] > 0.8

        assert pytest.approx(res2["locked_age"], abs=1.0) == 50.0
        assert res2["locked_gender"] == "Male"
        assert res2["gender_prob"] < 0.2

    def test_prune_inactive_targets(self):
        locker = TargetLocker()
        locker.update(track_id=1, age_pred=25.0, aleatoric_std=1.0, gender_logit=2.0)
        locker.update(track_id=2, age_pred=30.0, aleatoric_std=1.0, gender_logit=2.0)

        assert len(locker.active_track_ids) == 2
        locker.prune_inactive(active_track_ids=[1])
        assert len(locker.active_track_ids) == 1
        assert 1 in locker.active_track_ids
        assert 2 not in locker.active_track_ids

    def test_first_frame_initialization_no_doubling(self):
        """Verifies that the first observation is not double-counted on creation."""
        locker = TargetLocker()
        res = locker.update(track_id=1, age_pred=35.0, aleatoric_std=1.5, gender_logit=2.0)
        assert pytest.approx(res["log_odds"], abs=1e-3) == 2.0
        assert pytest.approx(res["locked_age"], abs=1e-3) == 35.0

    def test_coasting_without_measurement_does_not_pollute(self):
        """Verifies that occluded frames preserve biometric estimate without pulling towards 25.0."""
        locker = TargetLocker()
        locker.update(track_id=1, age_pred=65.0, aleatoric_std=1.0, gender_logit=3.0)

        # Coast for 5 frames without measurement
        for _ in range(5):
            res = locker.coast(track_id=1)

        # Age must remain at 65.0, NOT pulled towards 25.0!
        assert pytest.approx(res["locked_age"], abs=1e-3) == 65.0
        assert res["locked_gender"] == "Female"
        # Gender log-odds leaked naturally: 3.0 * (0.95^5) ~ 2.32
        assert pytest.approx(res["log_odds"], rel=1e-2) == 3.0 * (0.95 ** 5)
