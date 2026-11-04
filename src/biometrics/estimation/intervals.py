"""
Fine-Grained 3-5 Year Age Interval Architecture & Box-Convolved LDL
===================================================================
Replaces brittle point-age assumptions on noisy public datasets with a
biologically and statistically rigorous 23-bracket (3-5 year) age interval
system across the human lifespan [0, 100]:

1. Childhood & Adolescence (0-17 yrs): 3-year intervals ('0-2' .. '15-17')
   capturing rapid craniofacial osteological and dental development.
2. Young Adulthood (18-25 yrs): 4-year intervals ('18-21', '22-25')
   capturing post-pubertal skeletal maturation.
3. Adulthood, Middle Age & Senescence (26-100 yrs): 5-year intervals
   ('26-30' .. '96-100') capturing dermal elastosis and soft-tissue remodeling.

Every interval strictly spans 3 to 5 integer years (3 <= high - low + 1 <= 5),
partitioning [0, 100] with zero gaps and zero overlaps.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ldl_engine import compute_gaussian_ldl, get_adaptive_sigma


# 23 contiguous 3-5 year intervals partitioning [0, 100]
FINE_AGE_INTERVALS: List[Tuple[str, int, int]] = [
    # Childhood & Adolescence: 3-year intervals (width = 3 integer years)
    ("0-2", 0, 2),
    ("3-5", 3, 5),
    ("6-8", 6, 8),
    ("9-11", 9, 11),
    ("12-14", 12, 14),
    ("15-17", 15, 17),
    # Young Adult: 4-year intervals (width = 4 integer years)
    ("18-21", 18, 21),
    ("22-25", 22, 25),
    # Adult, Middle Age, Senior & Geriatric: 5-year intervals (width = 5 integer years)
    ("26-30", 26, 30),
    ("31-35", 31, 35),
    ("36-40", 36, 40),
    ("41-45", 41, 45),
    ("46-50", 46, 50),
    ("51-55", 51, 55),
    ("56-60", 56, 60),
    ("61-65", 61, 65),
    ("66-70", 66, 70),
    ("71-75", 71, 75),
    ("76-80", 76, 80),
    ("81-85", 81, 85),
    ("86-90", 86, 90),
    ("91-95", 91, 95),
    ("96-100", 96, 100),
]

NUM_FINE_AGE_INTERVALS: int = len(FINE_AGE_INTERVALS)

FINE_INTERVAL_NAME_TO_IDX: Dict[str, int] = {
    name: idx for idx, (name, _, _) in enumerate(FINE_AGE_INTERVALS)
}

FINE_INTERVAL_CENTERS: np.ndarray = np.array(
    [0.5 * (low + high) for _, low, high in FINE_AGE_INTERVALS],
    dtype=np.float32,
)

# Common Adience benchmark typo corrections documented in Levi & Hassner (CVPR 2015)
ADIENCE_TYPO_MAP: Dict[Tuple[int, int], Tuple[int, int]] = {
    (8, 23): (8, 12),
    (38, 42): (38, 43),
    (38, 48): (38, 43),
    (27, 32): (25, 32),
}


@dataclass(frozen=True)
class AgeIntervalSpec:
    """Structured descriptor for a 3-5 year age interval."""
    index: int
    name: str
    low: int
    high: int

    @property
    def span_years(self) -> int:
        return self.high - self.low + 1

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.low + self.high)


def get_fine_interval_by_index(idx: int) -> Tuple[str, int, int]:
    """Returns (name, low, high) for a valid interval index in [0, 22]."""
    clamped = max(0, min(NUM_FINE_AGE_INTERVALS - 1, int(idx)))
    return FINE_AGE_INTERVALS[clamped]


def get_fine_interval_for_age(age: float) -> Tuple[str, int, int, int]:
    """
    Maps any scalar age in [0, 100] to its canonical 3-5 year interval.
    Returns (interval_name, interval_index, low, high).
    """
    if math.isnan(age) or math.isinf(age):
        raise ValueError(f"Cannot map non-finite age {age!r} to a fine age interval.")
    age_clamped = max(0.0, min(100.0, float(age)))
    age_int = int(math.floor(age_clamped + 0.5))
    age_int = max(0, min(100, age_int))
    for idx, (name, low, high) in enumerate(FINE_AGE_INTERVALS):
        if low <= age_int <= high:
            return name, idx, low, high
    name, low, high = FINE_AGE_INTERVALS[-1]
    return name, NUM_FINE_AGE_INTERVALS - 1, low, high


def get_overlapping_fine_intervals(
    age_min: float,
    age_max: float,
) -> List[Tuple[int, str, float]]:
    """
    Computes all fine 3-5 year intervals overlapping [age_min, age_max]
    along with normalized overlap mass weights summing to 1.0.
    """
    if (
        math.isnan(float(age_min))
        or math.isinf(float(age_min))
        or math.isnan(float(age_max))
        or math.isinf(float(age_max))
    ):
        raise ValueError(f"Cannot compute overlapping intervals for non-finite bounds [{age_min!r}, {age_max!r}].")
    a_low = max(0.0, min(100.0, float(min(age_min, age_max))))
    a_high = max(0.0, min(100.0, float(max(age_min, age_max))))

    # Treat continuous support of [a_low, a_high] as [a_low - 0.5, a_high + 0.5]
    q_low = max(-0.5, a_low - 0.5)
    q_high = min(100.5, a_high + 0.5)

    overlaps: List[Tuple[int, str, float]] = []
    total_overlap = 0.0
    for idx, (name, low, high) in enumerate(FINE_AGE_INTERVALS):
        b_low = float(low) - 0.5
        b_high = float(high) + 0.5
        inter = max(0.0, min(q_high, b_high) - max(q_low, b_low))
        if inter > 0.0:
            overlaps.append((idx, name, inter))
            total_overlap += inter

    if not overlaps or total_overlap <= 0.0:
        name, idx, _, _ = get_fine_interval_for_age(0.5 * (a_low + a_high))
        return [(idx, name, 1.0)]

    return [(idx, name, float(w / total_overlap)) for idx, name, w in overlaps]


def map_bounds_to_fine_interval(
    age_min: float,
    age_max: float,
) -> Tuple[str, int, int, int]:
    """
    Maps an arbitrary interval [age_min, age_max] to its best-matching
    canonical 3-5 year interval. Uses midpoint alignment within the highest
    overlap candidates to guarantee deterministic, centered assignment.
    """
    if (
        math.isnan(float(age_min))
        or math.isinf(float(age_min))
        or math.isnan(float(age_max))
        or math.isinf(float(age_max))
    ):
        raise ValueError(f"Cannot map non-finite age bounds [{age_min!r}, {age_max!r}] to a fine interval.")
    a_low = max(0.0, min(100.0, float(min(age_min, age_max))))
    a_high = max(0.0, min(100.0, float(max(age_min, age_max))))
    midpoint = 0.5 * (a_low + a_high)
    return get_fine_interval_for_age(midpoint)


def parse_age_interval_string(
    raw: Any,
    snap_point_to_fine_interval: bool = False,
) -> Optional[Tuple[float, float, str]]:
    """
    Parses heterogeneous dataset age / bracket annotations into
    (age_min, age_max, label_type).

    Supported formats:
    - Tuple strings (Adience): "(0, 2)", "(4, 6)", "(8, 12)", "(25, 32)", "(60, 100)"
    - Dash / range strings (FairFace, Axon, Lagenda): "20-29", "[26, 30]", "26_30", "26 to 30"
    - Open-ended strings: "70+", "65+", ">60", "more than 70"
    - Exact scalars / numeric strings: 27, "27.5"
    """
    if raw is None:
        return None

    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        val = float(raw)
        if math.isnan(val) or math.isinf(val) or val < 0.0 or val > 120.0:
            return None
        val_clamped = min(100.0, val)
        if snap_point_to_fine_interval:
            _, _, low, high = get_fine_interval_for_age(val_clamped)
            return float(low), float(high), "fine_interval"
        return val_clamped, val_clamped, "exact"

    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        try:
            low_v = float(raw[0])
            high_v = float(raw[1])
            if (
                math.isnan(low_v)
                or math.isinf(low_v)
                or math.isnan(high_v)
                or math.isinf(high_v)
                or low_v < 0.0
                or high_v < 0.0
                or min(low_v, high_v) > 120.0
            ):
                return None
            a_low = min(100.0, min(low_v, high_v))
            a_high = min(100.0, max(low_v, high_v))
            key = (int(math.floor(a_low + 0.5)), int(math.floor(a_high + 0.5)))
            if key in ADIENCE_TYPO_MAP:
                a_low, a_high = float(ADIENCE_TYPO_MAP[key][0]), float(ADIENCE_TYPO_MAP[key][1])
            return a_low, a_high, "interval"
        except (TypeError, ValueError):
            return None

    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null", "unknown", "n/a", ""}:
        return None

    # 1. Open-ended senior brackets: "70+", "65+", "more than 70", ">= 65", ">60"
    m_open = re.match(
        r"^(?:more\s+than\s+|>\s*=?\s*)?(\d+(?:\.\d+)?)\s*\+?$",
        text,
        flags=re.IGNORECASE,
    )
    if m_open and ("+" in text or ">" in text or "more" in text.lower()):
        low_v = float(m_open.group(1))
        if 0.0 <= low_v <= 100.0:
            return low_v, 100.0, "interval"

    # 2. Tuple / bracket / dash / "to" range: "(25, 32)", "[18, 21]", "20-29", "26_30", "18 to 21"
    m_range = re.match(
        r"^[\(\[\s]*(\d+(?:\.\d+)?)\s*(?:[-_,]|to)\s*(\d+(?:\.\d+)?)[\)\]\s]*$",
        text,
        flags=re.IGNORECASE,
    )
    if m_range:
        v1 = float(m_range.group(1))
        v2 = float(m_range.group(2))
        if v1 < 0.0 or v2 < 0.0 or min(v1, v2) > 100.0:
            return None
        a_low = max(0.0, min(100.0, min(v1, v2)))
        a_high = max(0.0, min(100.0, max(v1, v2)))
        key = (int(math.floor(a_low + 0.5)), int(math.floor(a_high + 0.5)))
        if key in ADIENCE_TYPO_MAP:
            a_low, a_high = float(ADIENCE_TYPO_MAP[key][0]), float(ADIENCE_TYPO_MAP[key][1])
        label_type = "exact" if abs(a_high - a_low) < 1e-4 else "interval"
        return a_low, a_high, label_type

    # 3. Plain scalar number string: "28", "28.5"
    try:
        val = float(text)
        if math.isnan(val) or math.isinf(val) or val < 0.0 or val > 120.0:
            return None
        val_clamped = min(100.0, val)
        if snap_point_to_fine_interval:
            _, _, low, high = get_fine_interval_for_age(val_clamped)
            return float(low), float(high), "fine_interval"
        return val_clamped, val_clamped, "exact"
    except ValueError:
        return None


def compute_interval_box_ldl(
    age_min: float,
    age_max: float,
    sigma: Optional[float] = None,
    num_bins: int = 101,
) -> np.ndarray:
    """
    Box-Convolved Gaussian Label Distribution Learning (Interval-BC-LDL).

    Given an interval label [age_min, age_max]:
    - If age_min == age_max (point label), delegates to Boundary-Calibrated
      Gaussian LDL (`compute_gaussian_ldl`).
    - If age_max > age_min (interval label, e.g. 3-5 year bracket or 10-year bin),
      models the latent chronological age X ~ Uniform(age_min - 0.5, age_max + 0.5)
      convolved with biological aging variance epsilon ~ N(0, sigma^2):
          p(k) \\propto \\Phi((k - L)/sigma) - \\Phi((k - U)/sigma)
      with exact boundary reflection across the half-integer domain walls
      w_L = -0.5 and w_U = (num_bins - 1) + 0.5 to prevent truncation bias on
      infant ('0-2'), centenarian ('96-100'), and full-span ('0-100') brackets.

    Guarantees strict simplex normalization: p_k > 0 and sum(p) == 1.0.
    """
    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2, got {num_bins}")
    if (
        math.isnan(float(age_min))
        or math.isinf(float(age_min))
        or math.isnan(float(age_max))
        or math.isinf(float(age_max))
    ):
        raise ValueError(f"age_min and age_max must be finite, got [{age_min!r}, {age_max!r}]")

    K = float(num_bins - 1)
    a_low = max(0.0, min(K, float(min(age_min, age_max))))
    a_high = max(0.0, min(K, float(max(age_min, age_max))))
    midpoint = 0.5 * (a_low + a_high)

    if sigma is None or math.isnan(float(sigma)) or math.isinf(float(sigma)) or float(sigma) <= 0.0:
        base_sigma = 2.0
    else:
        base_sigma = float(sigma)

    if abs(a_high - a_low) < 1e-3:
        return compute_gaussian_ldl(mean_age=midpoint, sigma=base_sigma, num_bins=num_bins)

    eff_sigma = max(0.45, get_adaptive_sigma(midpoint, base_sigma=base_sigma))
    L = a_low - 0.5
    U = a_high + 0.5
    sqrt2_sigma = eff_sigma * math.sqrt(2.0)

    bins = np.arange(num_bins, dtype=np.float64)
    erf_vec = np.vectorize(math.erf)

    def _box_gauss_density(x: np.ndarray) -> np.ndarray:
        cdf_l = 0.5 * (1.0 + erf_vec((x - L) / sqrt2_sigma))
        cdf_u = 0.5 * (1.0 + erf_vec((x - U) / sqrt2_sigma))
        return np.maximum(0.0, cdf_l - cdf_u)

    # Reflect across the continuous domain walls w_L = -0.5 and w_U = K + 0.5:
    # reflection of x across -0.5 is 2*(-0.5) - x = -1.0 - x
    # reflection of x across K + 0.5 is 2*(K + 0.5) - x = 2*K + 1.0 - x
    q_pos = _box_gauss_density(bins)
    q_neg_lower = _box_gauss_density(-1.0 - bins)
    q_neg_upper = _box_gauss_density(2.0 * K + 1.0 - bins)

    q = q_pos + q_neg_lower + q_neg_upper

    q = np.maximum(q, 1e-12)
    q /= np.sum(q)
    return q.astype(np.float32)


def integrate_101_to_fine_intervals(dist_101: np.ndarray) -> np.ndarray:
    """
    Integrates a 101-bin continuous age distribution into the 23 fine-grained
    3-5 year age intervals. Supports both 1D (101,) and 2D (B, 101) arrays.
    """
    arr = np.asarray(dist_101, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != 101:
            raise ValueError(f"Expected 101 age bins, got shape {arr.shape}")
        out = np.zeros(NUM_FINE_AGE_INTERVALS, dtype=np.float64)
        for idx, (_, low, high) in enumerate(FINE_AGE_INTERVALS):
            out[idx] = np.sum(arr[low : high + 1])
        out = np.maximum(out, 1e-12)
        out /= np.sum(out)
        return out.astype(np.float32)
    elif arr.ndim == 2:
        if arr.shape[1] != 101:
            raise ValueError(f"Expected 101 age bins along axis 1, got shape {arr.shape}")
        out = np.zeros((arr.shape[0], NUM_FINE_AGE_INTERVALS), dtype=np.float64)
        for idx, (_, low, high) in enumerate(FINE_AGE_INTERVALS):
            out[:, idx] = np.sum(arr[:, low : high + 1], axis=1)
        out = np.maximum(out, 1e-12)
        out /= np.sum(out, axis=1, keepdims=True)
        return out.astype(np.float32)
    else:
        raise ValueError(f"Unsupported distribution rank {arr.ndim}; expected 1D or 2D.")


def compute_interval_epistemic_weight(
    age_min: float,
    age_max: float,
    label_type: str = "interval",
) -> float:
    """
    Computes an information-theoretic sample reliability weight in [0.0, 1.0]
    based on age interval width and annotation provenance:
    - 'missing' or invalid -> 0.0
    - 'exact' or span <= 1 yr -> 1.0
    - 3-5 yr fine interval (span 2..5 yrs) -> 0.90 - 0.96
    - 6-10 yr medium interval (FairFace / Adience) -> 0.70 - 0.85
    - 11-30 yr coarse interval -> 0.35 - 0.65
    - >30 yr binary / ultra-coarse interval -> 0.15
    """
    ltype = (label_type or "").lower()
    if ltype in {"missing", "none", "invalid", "binary_excluded"}:
        return 0.0

    span = abs(float(age_max) - float(age_min))
    if ltype == "exact" or span <= 1.0:
        return 1.0
    if span <= 5.0:
        # 3-5 year interval: high epistemic certainty
        return float(round(0.98 - 0.02 * span, 4))
    if span <= 10.0:
        # 6-10 year interval (e.g., FairFace 20-29, Adience 25-32)
        return float(round(0.88 - 0.03 * (span - 5.0), 4))
    if span <= 30.0:
        return float(round(0.73 - 0.019 * (span - 10.0), 4))
    return 0.15


def aggregate_101_to_fine_intervals_torch(probs_101):
    """
    Differentiable PyTorch aggregation of 101-bin age distribution (B, 101) or (101,)
    into 23 fine-grained 3-5 year age intervals (B, 23) or (23,).
    """
    import torch

    is_1d = probs_101.ndim == 1
    p = probs_101.unsqueeze(0) if is_1d else probs_101
    p = p.float()
    p = p / (torch.sum(p, dim=-1, keepdim=True) + 1e-12)
    slices = [torch.sum(p[:, low : high + 1], dim=-1, keepdim=True) for (_, low, high) in FINE_AGE_INTERVALS]
    fine_p = torch.cat(slices, dim=-1)
    fine_p = fine_p / (torch.sum(fine_p, dim=-1, keepdim=True) + 1e-12)
    return fine_p.squeeze(0) if is_1d else fine_p

