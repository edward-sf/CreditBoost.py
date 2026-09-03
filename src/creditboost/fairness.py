"""Disparate impact measurement.

Milestone 3 established that no protected attribute is a model feature. That is
a statement about inputs, and it is not fairness: neutral features act as
proxies, which is why disparate impact is a distinct doctrine from disparate
treatment. This module measures outcomes instead.

The adverse outcome is `band != "low"` -- the applicant is not auto-approved --
rather than `band == "high"`. At the shipped model's 97% approval rate, ratios
of approval rates compress toward 1.0 and cannot discriminate: an applicant
under 62 is denied at 9.19 times the rate of one aged 62 or over, and the
four-fifths test on approvals returns 0.974, a comfortable pass. Regulation B
defines adverse action to include credit granted on substantially different
terms, so a medium band is adverse action.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from . import config
from .banding import risk_band
from .schema import AttributeFairness, FairnessReport, GroupRate

ADVERSE_DEFINITION = "band != low"


def _age_buckets(frame: pd.DataFrame) -> pd.Series:
    """Bucket at the ECOA line. Unknown ages become NA rather than falling into
    the younger bucket: NaN >= 62 is False, which would silently report a rate
    for applicants whose age is not known."""
    years = -pd.to_numeric(frame["DAYS_BIRTH"], errors="coerce") / 365.25
    over = years >= config.ECOA_PROTECTED_AGE

    labels = pd.Series(pd.NA, index=frame.index, dtype="string")
    labels[over] = f"{config.ECOA_PROTECTED_AGE} and over"
    labels[~over & years.notna()] = f"under {config.ECOA_PROTECTED_AGE}"
    return labels


def _groups(frame: pd.DataFrame, attribute: str) -> pd.Series:
    if attribute == "DAYS_BIRTH":
        return _age_buckets(frame)
    return frame[attribute].astype("string")


def adverse_impact_ratios(
    frame: pd.DataFrame,
    adverse: Sequence[bool],
    min_group_size: int = config.MIN_FAIRNESS_GROUP_SIZE,
) -> list[AttributeFairness]:
    """Adverse impact ratio per protected attribute, given an explicit adverse
    mask.

    The mask is a parameter so the alternative search can score a candidate at a
    threshold matched to another candidate's approval rate, without
    reimplementing the age bucketing, the minimum group size, or -- above all --
    the min/max direction. There is one implementation of that arithmetic.
    """
    adverse_series = pd.Series(list(adverse), index=frame.index)

    attributes: list[AttributeFairness] = []
    for attribute in config.FAIRNESS_ATTRIBUTES:
        table = pd.DataFrame({"group": _groups(frame, attribute), "adverse": adverse_series})
        table = table.dropna(subset=["group"])
        summary = table.groupby("group")["adverse"].agg(["mean", "size"])
        eligible = summary[summary["size"] >= min_group_size]

        group_rates = [
            GroupRate(group=str(name), adverse_rate=float(row["mean"]), n=int(row["size"]))
            for name, row in eligible.iterrows()
        ]

        if len(eligible) < 2:
            attributes.append(
                AttributeFairness(
                    attribute=attribute,
                    unmeasured_reason=(
                        f"fewer than two groups reached the minimum size of {min_group_size}"
                    ),
                    groups=group_rates,
                )
            )
            continue

        favourable = 1.0 - eligible["mean"]
        if favourable.max() <= 0.0:
            attributes.append(
                AttributeFairness(
                    attribute=attribute,
                    unmeasured_reason=("no applicant in any group received the favourable outcome"),
                    groups=group_rates,
                )
            )
            continue

        # min over max, never the reverse. Inverted, a failing model reads as
        # passing and the gate silently permits everything.
        attributes.append(
            AttributeFairness(
                attribute=attribute,
                adverse_impact_ratio=float(favourable.min() / favourable.max()),
                groups=group_rates,
            )
        )

    return attributes


def evaluate(
    frame: pd.DataFrame,
    probabilities: Sequence[float],
    min_group_size: int = config.MIN_FAIRNESS_GROUP_SIZE,
) -> FairnessReport:
    """Adverse impact ratio per protected attribute, over the rows in `frame`.

    min_group_size is a parameter rather than read from config directly so tests
    can lower it; train.py takes the configured default.
    """
    adverse = [risk_band(float(p)) != "low" for p in probabilities]
    return FairnessReport(
        adverse_definition=ADVERSE_DEFINITION,
        band_low_max=config.RISK_BAND_LOW_MAX,
        min_group_size=min_group_size,
        attributes=adverse_impact_ratios(frame, adverse, min_group_size),
    )


def failing_attributes(
    report: FairnessReport,
    minimum: float = config.MIN_ADVERSE_IMPACT_RATIO,
) -> list[AttributeFairness]:
    """Attributes whose measured ratio falls below the floor.

    An unmeasured attribute is never returned: it has established nothing, so it
    is neither a failure to assert nor a pass to claim.
    """
    return [
        attribute
        for attribute in report.attributes
        if attribute.adverse_impact_ratio is not None and attribute.adverse_impact_ratio < minimum
    ]
