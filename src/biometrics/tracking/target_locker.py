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
        process_noise_q: float = 0.01,
        huber_delta: float = 2.5,
    ):
        safe_age = float(initial_age) if math.isfinite(float(initial_age)) else 25.0
        safe_std = float(initial_std) if (math.isfinite(float(initial_std)) and float(initial_std) > 0.0) else 1.0
        self.age = float(np.clip(safe_age, 0.0, 100.0))
        self.var = float(max(1e-4, safe_std ** 2))
        self.q = float(process_noise_q)
        self.huber_delta = float(huber_delta)

    def predict(self) -> float:
        """Coasting step when no measurement is received."""
        self.var = min(4.0, self.var + self.q)
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

        # Huber-Kalman robust innovation gating against multi-frame spikes
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
        decay: float = 0.98,
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
        """Holds log-odds steady during coasting frames when no new measurement is observed."""
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
    Acquires an immediate one-shot lock on the first valid measurement and holds
    the locked gender and 3-5 year age interval steady without post-lock jumping.
    """

    def __init__(
        self,
        gender_decay: float = 0.98,
        gender_bounds: float = 6.0,
        age_process_noise: float = 0.005,
        interval_hysteresis_margin: float = 2.5,
        lock_on_first_shot: bool = True,
    ):
        self.gender_decay = gender_decay
        self.gender_bounds = gender_bounds
        self.age_process_noise = age_process_noise
        self.interval_hysteresis_margin = float(interval_hysteresis_margin)
        self.lock_on_first_shot = bool(lock_on_first_shot)

        self.age_filters: Dict[int, EpistemicAgeKalmanFilter] = {}
        self.gender_accumulators: Dict[int, BoundedLeakyGenderAccumulator] = {}
        self.locked_intervals: Dict[int, int] = {}
        self.locked_genders: Dict[int, str] = {}
        self._initial_stds: Dict[int, float] = {}
        self._candidate_counts: Dict[int, Tuple[int, int]] = {}
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
        Locks immediately on the first valid measurement ("một phát ăn ngay") and
        prevents post-lock interval/gender jumping across subsequent frames.
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
            anchor_std = max(0.15, min(0.50, 0.20 * safe_std)) if valid_measurement else 1.0
            self.age_filters[track_id] = EpistemicAgeKalmanFilter(
                initial_age=age_pred if valid_measurement else 25.0,
                initial_std=anchor_std,
                process_noise_q=self.age_process_noise,
            )
            if valid_measurement:
                raw_g = float(gender_logit) * g_weight
                if abs(raw_g) > 1e-5 and abs(raw_g) < 3.0:
                    init_logit = math.copysign(max(3.0, abs(raw_g)), raw_g)
                else:
                    init_logit = raw_g
            else:
                init_logit = 0.0

            self.gender_accumulators[track_id] = BoundedLeakyGenderAccumulator(
                decay=self.gender_decay,
                min_bound=-self.gender_bounds,
                max_bound=self.gender_bounds,
                initial_logit=init_logit,
            )
            locked_age = self.age_filters[track_id].age
            log_odds = self.gender_accumulators[track_id].log_odds
            iv_name, iv_idx, iv_low, iv_high = get_fine_interval_for_age(locked_age)
            gender_label = self.gender_accumulators[track_id].label
            if valid_measurement:
                self._measured_tracks.add(track_id)
                self.locked_intervals[track_id] = iv_idx
                self.locked_genders[track_id] = gender_label
                self._initial_stds[track_id] = safe_std
                self._candidate_counts.pop(track_id, None)
            elif track_id not in self.locked_intervals:
                self.locked_intervals[track_id] = iv_idx
                self.locked_genders[track_id] = gender_label
        else:
            if valid_measurement:
                locked_age = self.age_filters[track_id].update(age_pred, max(safe_std * 2.5, 2.0))
                log_odds = self.gender_accumulators[track_id].update(float(gender_logit) * g_weight * 0.25)
            else:
                locked_age = self.age_filters[track_id].predict()
                log_odds = self.gender_accumulators[track_id].predict()

            iv_name, iv_idx, iv_low, iv_high = get_fine_interval_for_age(locked_age)
            gender_label = self.gender_accumulators[track_id].label

            if track_id in self.locked_intervals:
                prev_idx = self.locked_intervals[track_id]
                prev_name, prev_low, prev_high = FINE_AGE_INTERVALS[prev_idx]
                init_std = self._initial_stds.get(track_id, 1.0)

                if self.lock_on_first_shot and init_std <= 5.0:
                    # Keep the first-shot locked interval and gender rock-solid
                    iv_name, iv_idx, iv_low, iv_high = prev_name, prev_idx, prev_low, prev_high
                    gender_label = self.locked_genders.get(track_id, gender_label)
                    locked_age = float(np.clip(locked_age, float(iv_low), float(iv_high)))
                    self.age_filters[track_id].age = locked_age
                elif prev_idx != iv_idx:
                    in_deadband = (
                        (prev_low - 0.5 - self.interval_hysteresis_margin)
                        <= locked_age
                        <= (prev_high + 0.5 + self.interval_hysteresis_margin)
                    )
                    if in_deadband:
                        iv_name, iv_idx, iv_low, iv_high = prev_name, prev_idx, prev_low, prev_high
                        self._candidate_counts.pop(track_id, None)
                    else:
                        cand_idx, cand_cnt = self._candidate_counts.get(track_id, (iv_idx, 0))
                        cand_cnt = (cand_cnt + 1) if (cand_idx == iv_idx and valid_measurement) else 1
                        self._candidate_counts[track_id] = (iv_idx, cand_cnt)
                        if cand_cnt < 5:
                            iv_name, iv_idx, iv_low, iv_high = prev_name, prev_idx, prev_low, prev_high
                        else:
                            self.locked_intervals[track_id] = iv_idx
                            self._initial_stds[track_id] = min(init_std, safe_std)
                            self._candidate_counts.pop(track_id, None)

        gender_prob = self.gender_accumulators[track_id].probability
        self.locked_intervals[track_id] = iv_idx
        self.locked_genders[track_id] = gender_label

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
            self.locked_genders.pop(tid, None)
            self._initial_stds.pop(tid, None)
            self._candidate_counts.pop(tid, None)
            self._measured_tracks.discard(tid)


