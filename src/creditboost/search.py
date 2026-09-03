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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from . import config
from .data import split
from .fairness import adverse_impact_ratios
from .features import transform
from .schema import CandidateResult


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


class BaselineMissingError(ValueError):
    """The baseline candidate is absent from the frontier, or failed to train."""


def select(
    candidates: Sequence[CandidateResult],
    baseline: str,
    auc_budget: float = config.MAX_AUC_SACRIFICE,
    min_improvement: float = config.MIN_AIR_IMPROVEMENT,
) -> str:
    """The name of the candidate to adopt.

    Among candidates whose ROC-AUC is within `auc_budget` of the best scored
    candidate, take the highest minimum adverse impact ratio; then keep the
    baseline unless the winner beats it by more than `min_improvement`.

    Every candidate here was scored at the same matched approval rate, so the
    comparison is like-for-like by construction. Ties break by AUC and then by
    declaration order, which makes the result deterministic under a fixed seed.
    """
    scored = [c for c in candidates if c.failed_reason is None]
    order = {candidate.name: position for position, candidate in enumerate(candidates)}

    base = next((c for c in scored if c.name == baseline), None)
    if base is None:
        raise BaselineMissingError(
            f"baseline {baseline!r} is not among the scored candidates. The "
            "baseline must always train: without it there is nothing to "
            "compare an alternative against."
        )

    best_auc = max(c.roc_auc for c in scored)  # type: ignore[type-var]
    eligible = [c for c in scored if c.roc_auc >= best_auc - auc_budget]  # type: ignore[operator]

    winner = min(
        eligible,
        key=lambda c: (-c.min_adverse_impact_ratio, -c.roc_auc, order[c.name]),  # type: ignore[operator]
    )

    improvement = winner.min_adverse_impact_ratio - base.min_adverse_impact_ratio  # type: ignore[operator]
    if improvement <= min_improvement:
        return base.name
    return winner.name


RANKING_BASIS = "matched approval rate on the selection split"


@dataclass(frozen=True)
class Ranking:
    """A scored frontier, before any selection has been made."""

    target_approval_rate: float
    candidates: list[CandidateResult]


def matched_adverse_mask(probabilities: np.ndarray, approval_rate: float) -> np.ndarray:
    """The adverse mask that approves `approval_rate` of these applicants.

    Comparing candidates at a fixed threshold measures how lenient each one is,
    not how fair. A candidate whose probabilities are simply lower approves more
    people and its ratio drifts toward 1.0 for that reason alone -- measured, a
    single-feature model read 0.984 at a fixed threshold and 0.930 at a matched
    rate. Taking a quantile of each candidate's own predictions removes the
    scale entirely, so the comparison is of who is ranked adversely, not of how
    many.
    """
    threshold = float(np.quantile(probabilities, approval_rate))
    return probabilities > threshold


def _fit(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    score_features: pd.DataFrame,
    score_labels: pd.Series,
    params: Mapping[str, object],
) -> np.ndarray:
    """Train one candidate and return its predictions on the scoring frame.

    Early stopping watches the SCORING frame, not the training one. Watching the
    training frame would never stop early -- training loss keeps improving -- so
    every candidate would run the full NUM_BOOST_ROUND and be ranked on an
    overfit tail. This mirrors exactly what train.fit does with its validation
    split, one level down.

    train.py's constants are imported lazily because train.py imports this
    module.
    """
    from .train import EARLY_STOPPING_ROUNDS, NUM_BOOST_ROUND, PARAMS

    dtrain = xgb.DMatrix(train_features, label=train_labels, enable_categorical=True)
    dscore = xgb.DMatrix(score_features, label=score_labels, enable_categorical=True)
    booster = xgb.train(
        {**PARAMS, **params},
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dscore, "select")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    booster = booster[: booster.best_iteration + 1]
    return np.asarray(booster.predict(dscore))


def rank(
    train_frame: pd.DataFrame,
    seed: int = config.RANDOM_SEED,
    min_group_size: int = config.MIN_FAIRNESS_GROUP_SIZE,
) -> Ranking:
    """Score every candidate on a split nested inside the training frame.

    This function receives the TRAINING frame and nothing else. It is therefore
    structurally incapable of touching the validation split, which is what keeps
    selection bias out of the ratio the artifact reports.
    """
    inner_train, selection = split(train_frame, seed=seed, validation_size=config.SELECTION_SIZE)
    labels = inner_train[config.TARGET_COLUMN]
    truth = selection[config.TARGET_COLUMN]

    baseline_probabilities = _fit(
        apply(BASELINE, inner_train),
        labels,
        apply(BASELINE, selection),
        truth,
        BASELINE.params,
    )
    target_approval_rate = float((baseline_probabilities <= config.RISK_BAND_LOW_MAX).mean())

    results: list[CandidateResult] = []
    for spec in CANDIDATES:
        try:
            features = apply(spec, inner_train)
            if features.empty or not len(features.columns):
                raise ValueError("no features remain after drops")
            probabilities = (
                baseline_probabilities
                if spec is BASELINE
                else _fit(features, labels, apply(spec, selection), truth, spec.params)
            )
            mask = matched_adverse_mask(probabilities, target_approval_rate)
            attributes = adverse_impact_ratios(selection, mask.tolist(), min_group_size)
            ratios = {
                a.attribute: a.adverse_impact_ratio
                for a in attributes
                if a.adverse_impact_ratio is not None
            }
            if not ratios:
                raise ValueError("no protected attribute could be measured on the selection split")
            results.append(
                CandidateResult(
                    name=spec.name,
                    n_features=len(features.columns),
                    roc_auc=float(roc_auc_score(truth, probabilities)),
                    min_adverse_impact_ratio=min(ratios.values()),
                    adverse_impact_ratios=ratios,
                )
            )
        except Exception as error:  # noqa: BLE001 - recorded, never swallowed
            # A candidate that could not be scored is recorded, not skipped. A
            # missing candidate would misrepresent the breadth of the search.
            results.append(CandidateResult(name=spec.name, n_features=0, failed_reason=str(error)))

    return Ranking(target_approval_rate=target_approval_rate, candidates=results)
