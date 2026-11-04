"""
Biological Morphometry & Sexual Dimorphism Attenuation
======================================================
1. Life-course craniofacial sexual dimorphism weight:
   - Pre-pubertal children (<12 years): isometric skull development, w_dim in [0.25, 1.0].
   - Adults [12, 65 years]: fully manifested secondary sexual characteristics, w_dim = 1.0.
   - Geriatric population (>65 years): soft tissue convergence and hormonal attenuation, w_dim = 0.85.
2. Ethnic non-linear aging kinetics:
   - East Asian aging cliff at ~48 years.
   - Caucasian epidermal thinning.
   - African melanin preservation.
"""

from __future__ import annotations
import math
from typing import Tuple, Dict


from .intervals import (
    FINE_AGE_INTERVALS,
    NUM_FINE_AGE_INTERVALS,
    get_fine_interval_for_age,
    map_bounds_to_fine_interval,
)

AGE_COHORTS = [
    ('0-2', 0, 2),
    ('3-6', 3, 6),
    ('7-12', 7, 12),
    ('13-17', 13, 17),
    ('18-24', 18, 24),
    ('25-34', 25, 34),
    ('35-44', 35, 44),
    ('45-54', 45, 54),
    ('55-64', 55, 64),
    ('65+', 65, 100)
]


def get_cohort_for_age(age: float) -> Tuple[str, int]:
    """Returns (cohort_name, cohort_index) for continuous age in [0, 100]."""
    if math.isnan(float(age)) or math.isinf(float(age)):
        return ('25-34', 5)
    age_clamped = max(0.0, min(100.0, float(age)))
    age_int = int(math.floor(age_clamped + 0.5))
    for idx, (name, low, high) in enumerate(AGE_COHORTS):
        if low <= age_int <= high:
            return name, idx
    return ('65+', 9)



def calculate_sexual_dimorphism_weight(age: float) -> float:
    """
    Computes w_dimorphism in [0.25, 1.0].
    Mitigates spurious correlation learning on pre-pubertal and geriatric faces.
    """
    if age < 0.0:
        return 0.25
    elif age < 12.0:
        # Linear ramp from infant 0.25 to adolescent 1.0
        return float(round(0.25 + (0.75 * (age / 12.0)), 4))
    elif age <= 65.0:
        return 1.0
    else:
        # Geriatric soft-tissue attenuation
        return 0.85


def calculate_ethnic_aging_offset(age: float, ethnicity: str) -> float:
    """
    Computes apparent age visual offset based on ethnic dermis kinetics.
    - East Asian: youthful plateau until ~48 years, followed by sharp cliff.
    - African: melanin protection retains ~2-3 years visual youthfulness.
    - Caucasian: linear photo-aging progression.
    """
    eth = ethnicity.lower()
    if "asian" in eth:
        if age < 48.0:
            return -2.5  # Appears ~2.5 years younger
        else:
            # Post-cliff acceleration
            years_past = age - 48.0
            return float(min(4.0, -2.5 + (0.35 * years_past)))
    elif "black" in eth or "african" in eth:
        return -3.0  # Melanin photoprotection
    elif "caucasian" in eth or "white" in eth:
        if age > 40.0:
            return 1.5   # Solar elastosis progression
        return 0.0
    return 0.0
