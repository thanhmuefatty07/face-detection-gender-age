"""
Target Locker: Epistemic-Kalman Age Smoothing & Bounded Leaky Gender Accumulator
================================================================================
Eliminates age jitter and gender label flipping across consecutive video frames:
1. Epistemic-Kalman Filter for Continuous Age:
   Dynamically weights measurements using measurement variance R = sigma_aleatoric^2.
   When the model reports high uncertainty (turning face, occlusion, motion blur),
   Kalman Gain K approaches 0, filtering out jitter and holding steady estimates.
2. Bounded Leaky Bayesian Accumulator for Gender:
   Accumulates log-odds with leak factor gamma = 0.95 and saturation bounds [-6.0, 6.0].
   Prevents one-frame noise spikes from flipping classification decisions while maintaining
   responsiveness to true state transitions.
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Set, Optional
import numpy as np


class EpistemicAgeKalmanFilter:
    """
    Epistemic-Kalman filter for continuous age estimation.
    Measurement variance R is modulated by model's predicted aleatoric variance sigma^2,
    with Huber-Kalman innovation gating to suppress extreme environmental outlier spikes.
    """

    def __init__(
        self,
        initial_age: float = 25.0,
        initial_std: float = 1.0,
        process_noise_q: float = 0.05,
        huber_delta: float = 6.0,
    ):
        safe_age = float(initial_age) if math.isfinite(float(initial_age)) else 25.0
        safe_std = float(initial_std) if (math.isfinite(float(initial_std)) and float(initial_std) > 0.0) else 1.0
        self.age = float(np.clip(safe_age, 0.0, 100.0))
        self.var = float(max(1e-4, safe_std ** 2))
        self.q = float(process_noise_q)
        self.huber_delta = float(huber_delta)

    def predict(self) -> float:
        """Coasting step when no measurement is received."""
        self.var = self.var + self.q
        return self.age

    def update(self, measured_age: float, aleatoric_std: float) -> float:
        """
        Kalman predict and update step with heteroscedastic measurement noise
        and Huber innovation gating.
        
        Args:
            measured_age: instantaneous age prediction from network
            aleatoric_std: predicted standard deviation sigma
        Returns:
            Smoothed age estimate
        """
        if not math.isfinite(float(measured_age)) or not math.isfinite(float(aleatoric_std)):
            return self.predict()

        # Predict step
        p_pred = self.var + self.q

        # Measurement variance R = sigma^2
        r = float(max(1e-4, float(aleatoric_std) ** 2))
        clamped_meas = float(np.clip(float(measured_age), 0.0, 100.0))
        innovation = clamped_meas - self.age

        # Huber-Kalman robust innovation gating against extreme multi-frame spikes
        innov_std = math.sqrt(p_pred + r)
        threshold = self.huber_delta * innov_std
        if self.huber_delta > 0.0 and abs(innovation) > threshold:
            scale = abs(innovation) / max(threshold, 1e-6)
            r = r * (scale ** 2)

        # Kalman gain
        k = p_pred / (p_pred + r)

        # Update step
        self.age = float(np.clip(self.age + k * innovation, 0.0, 100.0))
        self.var = (1.0 - k) * p_pred

        return self.age


class BoundedLeakyGenderAccumulator:
    """
    Bounded Leaky Bayesian Accumulator for gender classification decisions.
    
    Update equation:
    L_t = clip(gamma * L_{t-1} + g_t, min_bound, max_bound)
    where g_t is the predicted gender logit (positive = Female, negative = Male).
    """

    def __init__(
        self,
        decay: float = 0.95,
        min_bound: float = -6.0,
        max_bound: float = 6.0,
        initial_logit: float = 0.0,
    ):
        self.decay = float(decay)
        self.min_bound = float(min_bound)
        self.max_bound = float(max_bound)
        safe_logit = float(initial_logit) if math.isfinite(float(initial_logit)) else 0.0
        self.log_odds = float(np.clip(safe_logit, self.min_bound, self.max_bound))

    def predict(self) -> float:
        """Leaky decay step when no new evidence is observed."""
        self.log_odds = self.decay * self.log_odds
        self.log_odds = float(np.clip(self.log_odds, self.min_bound, self.max_bound))
        return self.log_odds

    def update(self, gender_logit: float) -> float:
        """
        Updates log-odds with new evidence.
        
        Args:
            gender_logit: unnormalized logit (positive = Female)
        Returns:
            Updated log-odds
        """
        if not math.isfinite(float(gender_logit)):
            return self.predict()
        self.log_odds = self.decay * self.log_odds + float(gender_logit)
        self.log_odds = float(np.clip(self.log_odds, self.min_bound, self.max_bound))
        return self.log_odds

    @property
    def probability(self) -> float:
        """Posterior probability P(Female) from sigmoid of log-odds."""
        return float(1.0 / (1.0 + math.exp(-self.log_odds)))

    @property
    def label(self) -> str:
        """Discrete binary decision."""
        return "Female" if self.log_odds > 0.0 else "Male"


class TargetLocker:
    """
    Manages continuous biometric state locking for active video tracks.
    Maintains persistent Kalman filters, Bayesian accumulators, and 23-bracket
    hysteresis deadbands per track_id.
    """

    def __init__(
        self,
        gender_decay: float = 0.95,
        gender_bounds: float = 6.0,
        age_process_noise: float = 0.05,
        interval_hysteresis_margin: float = 0.35,
    ):
        self.gender_decay = gender_decay
        self.gender_bounds = gender_bounds
        self.age_process_noise = age_process_noise
        self.interval_hysteresis_margin = float(interval_hysteresis_margin)

        self.age_filters: Dict[int, EpistemicAgeKalmanFilter] = {}
        self.gender_accumulators: Dict[int, BoundedLeakyGenderAccumulator] = {}
        self.locked_intervals: Dict[int, int] = {}
        self._measured_tracks: Set[int] = set()

    @property
    def active_track_ids(self) -> Set[int]:
        """Returns the set of currently locked track IDs."""
        return set(self.age_filters.keys())

    def update(
        self,
        track_id: int,
        age_pred: float = 25.0,
        aleatoric_std: float = 1.0,
        gender_logit: float = 0.0,
        has_measurement: bool = True,
    ) -> Dict[str, Any]:
        """
        Updates continuous biometric target lock for a specified track.
        
        Args:
            track_id: unique persistent tracker identity ID
            age_pred: raw instantaneous expected age
            aleatoric_std: raw predicted uncertainty std
            gender_logit: raw predicted gender logit
            has_measurement: whether a real detection was matched to this track in current frame
        Returns:
            Dictionary with locked biometric attributes:
            - 'locked_age': continuous smoothed age (zero flicker)
            - 'locked_age_interval': 3-5 year hysteresis-stabilized bracket string
            - 'locked_age_interval_idx': 3-5 year bracket index [0..22]
            - 'locked_age_bounds': (low, high) integer bounds
            - 'locked_gender': stable gender string ('Male' or 'Female')
            - 'gender_prob': posterior confidence in Female [0.0, 1.0]
            - 'log_odds': current log-odds accumulator value
        """
        from ..estimation.intervals import FINE_AGE_INTERVALS, get_fine_interval_for_age

        valid_measurement = bool(
            has_measurement
            and math.isfinite(float(age_pred))
            and math.isfinite(float(aleatoric_std))
            and math.isfinite(float(gender_logit))
        )
        is_new_track = track_id not in self.age_filters
        is_first_measurement = valid_measurement and (track_id not in self._measured_tracks)
        safe_std = max(1e-4, float(aleatoric_std)) if valid_measurement else 1.0
        g_weight = min(1.0, (2.0 / safe_std) ** 2) if valid_measurement else 1.0

        if is_new_track or is_first_measurement:
            self.age_filters[track_id] = EpistemicAgeKalmanFilter(
                initial_age=age_pred if valid_measurement else 25.0,
                initial_std=aleatoric_std if valid_measurement else 1.0,
                process_noise_q=self.age_process_noise,
            )
            self.gender_accumulators[track_id] = BoundedLeakyGenderAccumulator(
                decay=self.gender_decay,
                min_bound=-self.gender_bounds,
                max_bound=self.gender_bounds,
                initial_logit=(float(gender_logit) * g_weight) if valid_measurement else 0.0,
            )
            if valid_measurement:
                self._measured_tracks.add(track_id)
            locked_age = self.age_filters[track_id].age
            log_odds = self.gender_accumulators[track_id].log_odds
        else:
            if valid_measurement:
                locked_age = self.age_filters[track_id].update(age_pred, aleatoric_std)
                log_odds = self.gender_accumulators[track_id].update(float(gender_logit) * g_weight)
            else:
                locked_age = self.age_filters[track_id].predict()
                log_odds = self.gender_accumulators[track_id].predict()

        gender_prob = self.gender_accumulators[track_id].probability
        gender_label = self.gender_accumulators[track_id].label
        iv_name, iv_idx, iv_low, iv_high = get_fine_interval_for_age(locked_age)

        # Apply 23-bracket hysteresis deadband to prevent boundary oscillation
        if not (is_new_track or is_first_measurement) and track_id in self.locked_intervals and self.interval_hysteresis_margin > 0.0:
            prev_idx = self.locked_intervals[track_id]
            if prev_idx != iv_idx and 0 <= prev_idx < len(FINE_AGE_INTERVALS):
                prev_name, prev_low, prev_high = FINE_AGE_INTERVALS[prev_idx]
                # Retention band around previous interval [prev_low - 0.5 - margin, prev_high + 0.5 + margin]
                if (prev_low - 0.5 - self.interval_hysteresis_margin) <= locked_age <= (prev_high + 0.5 + self.interval_hysteresis_margin):
                    iv_name, iv_idx, iv_low, iv_high = prev_name, prev_idx, prev_low, prev_high

        self.locked_intervals[track_id] = iv_idx

        return {
            "locked_age": locked_age,
            "locked_age_interval": iv_name,
            "locked_age_interval_idx": iv_idx,
            "locked_age_bounds": (iv_low, iv_high),
            "locked_gender": gender_label,
            "gender_prob": gender_prob,
            "log_odds": log_odds,
        }

    def coast(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Advances track without measurement (during temporary occlusion)."""
        if track_id not in self.age_filters:
            return None
        return self.update(track_id=track_id, has_measurement=False)

    def prune_inactive(self, active_track_ids: List[int]) -> None:
        """Removes state trackers for tracks that are no longer active."""
        active_set = set(active_track_ids)
        to_remove = [tid for tid in self.age_filters if tid not in active_set]
        for tid in to_remove:
            self.age_filters.pop(tid, None)
            self.gender_accumulators.pop(tid, None)
            self.locked_intervals.pop(tid, None)
            self._measured_tracks.discard(tid)

