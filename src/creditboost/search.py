"""The less discriminatory alternative search.

Milestone 4 measures disparate impact and refuses to ship a model that fails the
four-fifths rule. It leaves exactly one response to a failing gate: stop. This
module supplies the other one -- a recorded search for a model that does better.

Disparate impact is a burden-shifting doctrine: business necessity rebuts a
prima facie case only if no less discriminatory alternative achieving comparable
performance exists. Searching for one is the obligation, and a recorded search,
including one that finds nothing, is the evidence the obligation was met.

Two rules run through everything here:

  * Candidates are compared at a MATCHED APPROVAL RATE, never at a fixed
    threshold. At a fixed threshold the ratio reports how lenient a candidate is
    rather than how fair, and the search would reliably pick whichever model
    approves the most people.

  * The band threshold is never a search axis. It dominates every model effect
    -- moving RISK_BAND_LOW_MAX from 0.10 to 0.12 buys more ratio than every
    model variant measured -- and it is risk appetite, not fairness. Selecting it
    for its fairness number is precisely the failure Milestone 4 was written to
    avoid. CandidateSpec has no field that can express one.

This module is training-side: it imports data.split, so serve/ must never reach
it. The import-linter contract in pyproject.toml enforces that.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from . import config
from .features import transform


class UnknownFeatureError(ValueError):
    """A candidate names a column that is not a model feature."""


@dataclass(frozen=True)
class CandidateSpec:
    """One model specification to try.

    Three axes, none of which is a protected attribute and none of which is the
    band threshold:

      drops      -- columns removed from the transform's output. They apply
                    AFTER the transform, so removing DAYS_EMPLOYED does not
                    disturb employed_to_age, which is already computed.
      collapses  -- {column: {level: replacement_or_None}}, applied to the raw
                    frame BEFORE the transform, because the transform builds a
                    Categorical from the declared levels. This is how the
                    NAME_INCOME_TYPE proxy-level questions enter the search as
                    measurements rather than judgments.
      params     -- xgboost parameter overrides.
    """

    name: str
    drops: tuple[str, ...] = ()
    collapses: Mapping[str, Mapping[str, str | None]] = field(default_factory=dict)
    params: Mapping[str, object] = field(default_factory=dict)


BASELINE = CandidateSpec(name="baseline")

# The catalog must be wide enough to contain real trade-offs. A space of small
# perturbations was measured and contains nothing: sixteen ablation and
# regularization variants spanned a minimum AIR of 0.8058 to 0.8146, entirely
# inside the sampling noise. The large feature-subset candidates are the ones
# that produced a frontier, so they are here on purpose, not as padding.
#
# The two NAME_INCOME_TYPE candidates carry the proxy-level question deferred
# since Milestone 3: "Maternity leave" is a sex proxy and "Pensioner" an age
# proxy. Measured, they move the ratio by 0.0001. They stay in the catalog so
# that finding is re-established on every search rather than remembered.
CANDIDATES: tuple[CandidateSpec, ...] = (
    BASELINE,
    CandidateSpec(
        name="maternity-to-working",
        collapses={"NAME_INCOME_TYPE": {"Maternity leave": "Working"}},
    ),
    CandidateSpec(
        name="income-type-proxies-dropped",
        collapses={"NAME_INCOME_TYPE": {"Maternity leave": None, "Pensioner": None}},
    ),
    CandidateSpec(name="no-income-type", drops=("NAME_INCOME_TYPE",)),
    CandidateSpec(name="no-occupation", drops=("OCCUPATION_TYPE",)),
    CandidateSpec(name="no-housing-type", drops=("NAME_HOUSING_TYPE",)),
    CandidateSpec(name="no-household", drops=("CNT_CHILDREN", "CNT_FAM_MEMBERS")),
    CandidateSpec(name="no-employed-to-age", drops=("employed_to_age",)),
    CandidateSpec(name="no-employment", drops=("DAYS_EMPLOYED", "employed_to_age")),
    # The large subsets. external-scores-only was measured at min AIR 0.866
    # against the baseline's 0.810, at a cost of 0.028 AUC -- the only candidate
    # so far to move the ratio by more than the noise.
    CandidateSpec(
        name="external-scores-only",
        drops=tuple(
            column for column in config.FEATURE_ORDER if not column.startswith("EXT_SOURCE")
        ),
    ),
    CandidateSpec(
        name="no-external-scores",
        drops=tuple(column for column in config.FEATURE_ORDER if column.startswith("EXT_SOURCE")),
    ),
    CandidateSpec(name="depth-3", params={"max_depth": 3}),
    CandidateSpec(name="depth-4", params={"max_depth": 4}),
    CandidateSpec(name="min-child-weight-50", params={"min_child_weight": 50}),
    CandidateSpec(name="lambda-10", params={"lambda": 10}),
    CandidateSpec(
        name="income-type-proxies-dropped-depth-3",
        collapses={"NAME_INCOME_TYPE": {"Maternity leave": None, "Pensioner": None}},
        params={"max_depth": 3},
    ),
)


def apply(spec: CandidateSpec, frame: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw frame into this candidate's feature matrix.

    Collapses run before the transform and drops after it, which is what keeps
    a derived feature intact when its source column is dropped.
    """
    unknown = [column for column in spec.drops if column not in config.FEATURE_ORDER]
    if unknown:
        raise UnknownFeatureError(
            f"candidate {spec.name!r} drops {unknown}, which are not model "
            f"features. Known features: {list(config.FEATURE_ORDER)}"
        )

    raw = frame
    for column, mapping in spec.collapses.items():
        if column in raw.columns:
            raw = raw.assign(**{column: raw[column].replace(dict(mapping))})  # type: ignore[arg-type]

    out = transform(raw)
    return out[[column for column in out.columns if column not in spec.drops]]
