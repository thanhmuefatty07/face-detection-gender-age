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

    def test_huber_kalman_outlier_gating(self):
        """Verifies Huber-Kalman innovation gating suppresses an extreme 50-year spike even when aleatoric_std is small."""
        kf = EpistemicAgeKalmanFilter(initial_age=25.0, initial_std=1.0, huber_delta=5.0)
        # Without Huber gating, measured_age=85.0 with aleatoric_std=1.0 would pull age to ~55.0!
        smoothed = kf.update(measured_age=85.0, aleatoric_std=1.0)
        assert abs(smoothed - 25.0) < 6.0

    def test_23_bracket_hysteresis_deadband_prevents_boundary_oscillation(self):
        """Verifies that slight age oscillations across the 25.5 boundary ('22-25' vs '26-30') do not flicker brackets."""
        locker = TargetLocker(interval_hysteresis_margin=0.35)
        res0 = locker.update(track_id=7, age_pred=25.3, aleatoric_std=0.2, gender_logit=1.0)
        assert res0["locked_age_interval"] == "22-25"

        # Slightly cross 25.5 to 25.65 (within 25.5 + 0.35 = 25.85 hysteresis retention band)
        res1 = locker.update(track_id=7, age_pred=25.7, aleatoric_std=0.1, gender_logit=1.0)
        assert 25.5 < res1["locked_age"] <= 25.85
        assert res1["locked_age_interval"] == "22-25"

        # Penetrate well past 25.85 (e.g. 27.5) -> transitions cleanly to '26-30'
        res2 = locker.update(track_id=7, age_pred=28.0, aleatoric_std=0.1, gender_logit=1.0)
        assert res2["locked_age_interval"] == "26-30"

    def test_first_real_measurement_after_unmeasured_creation_anchors_immediately(self):
        """Verifies a track created with has_measurement=False anchors to its first real measurement instead of rejecting it via Huber gating."""
        locker = TargetLocker()
        # Frame 1: Track created before first crop inference completes
        res_unmeasured = locker.update(track_id=99, has_measurement=False)
        assert res_unmeasured["locked_age"] == 25.0

        # Frame 2: First real measurement arrives for a 72yo subject
        res_measured = locker.update(
            track_id=99,
            age_pred=72.0,
            aleatoric_std=1.0,
            gender_logit=2.5,
            has_measurement=True,
        )
        assert pytest.approx(res_measured["locked_age"], abs=1e-3) == 72.0
        assert res_measured["locked_age_interval"] == "71-75"
        assert res_measured["locked_gender"] == "Female"
        assert pytest.approx(res_measured["log_odds"], abs=1e-3) == 2.5

    def test_nan_and_inf_measurements_coast_safely_without_poisoning(self):
        """Verifies NaN or Inf inputs do not poison Kalman state or crash interval lookup."""
        locker = TargetLocker()
        locker.update(track_id=5, age_pred=42.0, aleatoric_std=1.0, gender_logit=-2.0)

        res_nan = locker.update(
            track_id=5,
            age_pred=float("nan"),
            aleatoric_std=float("inf"),
            gender_logit=float("nan"),
            has_measurement=True,
        )
        assert pytest.approx(res_nan["locked_age"], abs=1e-3) == 42.0
        assert res_nan["locked_age_interval"] == "41-45"
        assert res_nan["locked_gender"] == "Male"

    def test_high_uncertainty_ood_crop_attenuates_gender_flip(self):
        """Verifies high aleatoric/total uncertainty (>2.0) attenuates gender logit updates to prevent OOD flips."""
        locker = TargetLocker()
        for _ in range(2):
            res = locker.update(track_id=11, age_pred=30.0, aleatoric_std=1.0, gender_logit=1.5)
        assert res["locked_gender"] == "Female"

        # Two consecutive high-uncertainty OOD / occluded crops (std=10.0) with strong opposite logit (-4.0)
        for _ in range(2):
            res = locker.update(track_id=11, age_pred=75.0, aleatoric_std=10.0, gender_logit=-4.0)
        assert res["locked_gender"] == "Female"
        assert res["log_odds"] > 1.5


