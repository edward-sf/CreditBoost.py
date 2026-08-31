"""Probability to risk band.

Thresholds are business policy, not model properties: they live in config and
change without retraining, which is why they are not stored in the artifact.
"""

from __future__ import annotations

from typing import Literal

from . import config

RiskBand = Literal["low", "medium", "high"]


def risk_band(probability: float) -> RiskBand:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if probability < config.RISK_BAND_LOW_MAX:
        return "low"
    if probability < config.RISK_BAND_MEDIUM_MAX:
        return "medium"
    return "high"
