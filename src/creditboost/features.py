"""The single feature transform, imported by both training and serving.

There is exactly one implementation of this logic on purpose. Training records
the resulting FEATURE_ORDER into the artifact metadata, and serving refuses to
start if the two disagree, which is what makes train/serve skew impossible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from . import config

FEATURE_ORDER = config.FEATURE_ORDER

Records = pd.DataFrame | Sequence[Mapping[str, Any]]


def _as_frame(records: Records) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records
    return pd.DataFrame(list(records))


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Always return a Series, so a missing input column degrades to NaN."""
    if name in frame.columns:
        return frame[name]
    return pd.Series([None] * len(frame), index=frame.index, dtype="object")


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(_column(frame, name), errors="coerce").astype("float64")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, yielding NaN where the denominator is zero or missing."""
    return numerator / denominator.mask(denominator == 0)


def transform(records: Records) -> pd.DataFrame:
    """Turn raw applicant records into the model's feature matrix.

    Accepts either a DataFrame (training) or a sequence of mappings (serving);
    both paths run identical logic. Missing values are never imputed — XGBoost
    routes NaN natively, and for a thin-file borrower a missing external score
    is itself signal.
    """
    frame = _as_frame(records)
    if frame.empty:
        frame = pd.DataFrame(index=pd.RangeIndex(0))

    out = pd.DataFrame(index=frame.index)

    for name in config.NUMERIC_FEATURES:
        out[name] = _numeric(frame, name)

    # Scrub the not-employed sentinel BEFORE deriving ratios from it. Left raw,
    # 365243 becomes a ~1000-year tenure that reads as a plausible value.
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].mask(
        out["DAYS_EMPLOYED"] == config.DAYS_EMPLOYED_SENTINEL
    )

    for name in config.BINARY_FEATURES:
        mapped = _column(frame, name).map({"Y": 1.0, "N": 0.0})
        out[name] = pd.to_numeric(mapped, errors="coerce").astype("float64")

    for name, levels in config.CATEGORICAL_LEVELS.items():
        raw = _column(frame, name)
        # Mask any value outside the declared levels to NaN *before* building the
        # Categorical. pandas 3 still accepts out-of-category values passed via
        # `categories=` and silently NaNs them, but it does so as a deprecated
        # path (Pandas4Warning) that will raise in a future pandas major version.
        # Masking explicitly here keeps "unknown level -> NaN, never raise" true
        # once that behavior actually flips, not just for as long as it happens
        # to still warn.
        out[name] = pd.Categorical(raw.where(raw.isin(levels)), categories=list(levels))

    days_birth = _numeric(frame, "DAYS_BIRTH")
    income = out["AMT_INCOME_TOTAL"]
    out["credit_to_income"] = _safe_divide(out["AMT_CREDIT"], income)
    out["annuity_to_income"] = _safe_divide(out["AMT_ANNUITY"], income)
    out["employed_to_age"] = _safe_divide(out["DAYS_EMPLOYED"], days_birth)

    return out[list(FEATURE_ORDER)]
