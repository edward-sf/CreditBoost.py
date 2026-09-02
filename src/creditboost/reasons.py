"""Adverse action reason codes.

Under ECOA / Regulation B 1002.9 a creditor taking adverse action must disclose
the specific principal reasons for it. This module turns a row of per-feature
model contributions into at most four of them.

Deliberately pure: contributions arrive as a plain mapping rather than a numpy
array, so the grouping, ranking and wording logic is testable with dictionaries
and no model at all. The array handling lives at the edge, in serve/app.py.
"""

from __future__ import annotations

from collections.abc import Mapping

from . import config
from .schema import Reason


def _concept_total(features: tuple[str, ...], contributions: Mapping[str, float]) -> float:
    """Sum a concept's contributions. A feature absent from the mapping counts as
    zero rather than raising: the partition test already guarantees the map and
    FEATURE_ORDER agree, so this cannot silently hide a missing feature."""
    return sum(contributions.get(name, 0.0) for name in features)


def _is_fully_absent(features: tuple[str, ...], missing: Mapping[str, bool]) -> bool:
    """True only when every feature in the concept was missing after the
    transform. Following the single largest contributor instead would be more
    precise and more surprising: an applicant with two of three bureau scores on
    file could be told no score is on file, because the absent one dominated."""
    return all(missing.get(name, False) for name in features)


def principal_reasons(
    contributions: Mapping[str, float],
    missing: Mapping[str, bool],
) -> list[Reason]:
    """The principal factors increasing this applicant's risk, most significant
    first, at most config.MAX_REASONS of them.

    Only positive contributions are eligible: a feature that pushed the applicant
    toward approval is not a reason for denial.
    """
    totals = [
        (concept, _concept_total(features, contributions))
        for concept, features in config.REASON_CONCEPTS.items()
    ]
    adverse = [(concept, total) for concept, total in totals if total > 0.0]

    # Stable sort, so equal totals fall back to REASON_CONCEPTS' declaration
    # order: the same request always yields the same reasons in the same order.
    adverse.sort(key=lambda pair: pair[1], reverse=True)

    reasons: list[Reason] = []
    for concept, _total in adverse[: config.MAX_REASONS]:
        text = config.REASON_TEXT[concept]
        # Narrowed inside the branch rather than via a separate boolean: mypy
        # does not carry `is not None` across an assignment.
        if text.absent is not None and _is_fully_absent(config.REASON_CONCEPTS[concept], missing):
            description = text.absent
        else:
            description = text.unfavourable
        reasons.append(Reason(code=text.code, description=description))
    return reasons
