"""The alternative search: the catalog, the spec transform, the matched mask,
the selection rule, and the ranking."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from creditboost import config
from creditboost.features import transform
from creditboost.schema import CandidateResult
from creditboost.search import (
    BASELINE,
    CANDIDATES,
    BaselineMissingError,
    CandidateSpec,
    Ranking,
    UnknownFeatureError,
    apply,
    matched_adverse_mask,
    rank,
    select,
)


def a_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [100_000.0, 200_000.0, 150_000.0],
            "AMT_CREDIT": [400_000.0, 500_000.0, 300_000.0],
            "AMT_ANNUITY": [20_000.0, 25_000.0, 15_000.0],
            "DAYS_BIRTH": [-15_000, -20_000, -12_000],
            "DAYS_EMPLOYED": [-2_000, -3_000, -500],
            "NAME_INCOME_TYPE": ["Working", "Pensioner", "Maternity leave"],
        }
    )


def test_the_baseline_is_first_and_changes_nothing():
    frame = a_raw_frame()
    assert CANDIDATES[0] is BASELINE
    pd.testing.assert_frame_equal(apply(BASELINE, frame), transform(frame))


def test_apply_drops_only_the_named_columns():
    frame = a_raw_frame()
    spec = CandidateSpec(name="x", drops=("CNT_CHILDREN",))
    result = apply(spec, frame)
    assert "CNT_CHILDREN" not in result.columns
    assert len(result.columns) == len(config.FEATURE_ORDER) - 1


def test_dropping_a_source_column_leaves_its_derived_feature_intact():
    """Drops apply to the transform's OUTPUT, so a derived feature is already
    computed by the time its source is removed. employed_to_age must survive
    DAYS_EMPLOYED being dropped, and keep its value."""
    frame = a_raw_frame()
    spec = CandidateSpec(name="x", drops=("DAYS_EMPLOYED",))
    result = apply(spec, frame)
    assert "DAYS_EMPLOYED" not in result.columns
    assert "employed_to_age" in result.columns
    pd.testing.assert_series_equal(result["employed_to_age"], transform(frame)["employed_to_age"])


def test_apply_collapses_a_level_before_the_transform():
    frame = a_raw_frame()
    spec = CandidateSpec(name="x", collapses={"NAME_INCOME_TYPE": {"Maternity leave": "Working"}})
    result = apply(spec, frame)
    assert list(result["NAME_INCOME_TYPE"].astype("string")) == [
        "Working",
        "Pensioner",
        "Working",
    ]


def test_a_collapse_target_of_none_becomes_missing():
    frame = a_raw_frame()
    spec = CandidateSpec(name="x", collapses={"NAME_INCOME_TYPE": {"Pensioner": None}})
    result = apply(spec, frame)
    assert result["NAME_INCOME_TYPE"].isna().tolist() == [False, True, False]


def test_apply_rejects_a_drop_naming_an_unknown_column():
    spec = CandidateSpec(name="x", drops=("NOT_A_FEATURE",))
    with pytest.raises(UnknownFeatureError, match="NOT_A_FEATURE"):
        apply(spec, a_raw_frame())


def test_candidate_names_are_unique():
    names = [spec.name for spec in CANDIDATES]
    assert len(names) == len(set(names))


def test_no_candidate_names_a_protected_attribute():
    """ECOA: a protected attribute is never a model feature, so no candidate can
    drop one (it was never there) or collapse its levels."""
    for spec in CANDIDATES:
        for prohibited in config.PROTECTED_ATTRIBUTES:
            assert prohibited not in spec.drops
            assert prohibited not in spec.collapses


def test_every_candidate_drop_names_a_real_feature():
    for spec in CANDIDATES:
        for column in spec.drops:
            assert column in config.FEATURE_ORDER, f"{spec.name} drops unknown {column}"


def test_a_candidate_cannot_vary_the_band_threshold():
    """The band threshold dominates every model effect, so selecting it for its
    fairness number would manufacture assurance rather than establish it. A
    CandidateSpec has no field that can express one, and this pins that."""
    assert {field.name for field in dataclasses.fields(CandidateSpec)} == {
        "name",
        "drops",
        "collapses",
        "params",
    }


def scored(name, auc, air):
    return CandidateResult(
        name=name,
        n_features=20,
        roc_auc=auc,
        min_adverse_impact_ratio=air,
        adverse_impact_ratios={"DAYS_BIRTH": air},
    )


def failed(name, reason="did not train"):
    return CandidateResult(name=name, n_features=0, failed_reason=reason)


def test_the_fairest_candidate_within_the_budget_wins():
    candidates = [
        scored("baseline", auc=0.750, air=0.810),
        scored("fairer", auc=0.745, air=0.860),
    ]
    assert select(candidates, baseline="baseline", auc_budget=0.01) == "fairer"


def test_a_candidate_outside_the_auc_budget_cannot_win():
    candidates = [
        scored("baseline", auc=0.750, air=0.810),
        scored("much-fairer-but-worse", auc=0.700, air=0.950),
    ]
    assert select(candidates, baseline="baseline", auc_budget=0.01) == "baseline"


def test_the_budget_boundary_is_inclusive():
    """Exactly at best - budget is eligible; a hair below is not."""
    inside = [
        scored("baseline", auc=0.750, air=0.810),
        scored("edge", auc=0.740, air=0.900),
    ]
    assert select(inside, baseline="baseline", auc_budget=0.01) == "edge"

    outside = [
        scored("baseline", auc=0.750, air=0.810),
        scored("edge", auc=0.7399, air=0.900),
    ]
    assert select(outside, baseline="baseline", auc_budget=0.01) == "baseline"


def test_the_noise_guard_keeps_the_baseline_on_an_improvement_within_noise():
    """AIR's measured sd is about 0.005 on the full validation split and larger
    on the smaller selection split. Without this guard the search churns the
    shipped model on noise."""
    candidates = [
        scored("baseline", auc=0.750, air=0.810),
        scored("marginally-fairer", auc=0.750, air=0.815),
    ]
    assert select(candidates, baseline="baseline", min_improvement=0.01) == "baseline"


def test_the_noise_guard_boundary_is_exclusive():
    """An improvement of exactly min_improvement keeps the baseline. Written in
    exactly-representable binary fractions so the assertion tests the rule and
    not the float arithmetic: 0.75 - 0.5 is exactly 0.25."""
    candidates = [
        scored("baseline", auc=0.750, air=0.5),
        scored("exactly-at-the-guard", auc=0.750, air=0.75),
    ]
    assert select(candidates, baseline="baseline", min_improvement=0.25) == "baseline"

    candidates = [
        scored("baseline", auc=0.750, air=0.5),
        scored("past-the-guard", auc=0.750, air=0.9),
    ]
    assert select(candidates, baseline="baseline", min_improvement=0.25) == "past-the-guard"


def test_the_noise_guard_yields_to_an_improvement_above_it():
    candidates = [
        scored("baseline", auc=0.750, air=0.810),
        scored("really-fairer", auc=0.750, air=0.8201),
    ]
    assert select(candidates, baseline="baseline", min_improvement=0.01) == "really-fairer"


def test_a_failed_candidate_is_never_selected():
    candidates = [scored("baseline", auc=0.750, air=0.810), failed("broken")]
    assert select(candidates, baseline="baseline") == "baseline"


def test_ties_break_by_auc_then_by_declaration_order():
    candidates = [
        scored("baseline", auc=0.700, air=0.810),
        scored("first", auc=0.750, air=0.900),
        scored("second", auc=0.750, air=0.900),
        scored("third", auc=0.749, air=0.900),
    ]
    assert select(candidates, baseline="baseline", auc_budget=0.01) == "first"


def test_selection_is_deterministic():
    candidates = [
        scored("baseline", auc=0.750, air=0.810),
        scored("a", auc=0.749, air=0.900),
        scored("b", auc=0.749, air=0.900),
    ]
    assert len({select(candidates, baseline="baseline") for _ in range(20)}) == 1


def test_a_missing_or_failed_baseline_is_an_error():
    with pytest.raises(BaselineMissingError):
        select([scored("other", auc=0.75, air=0.81)], baseline="baseline")
    with pytest.raises(BaselineMissingError):
        select([failed("baseline")], baseline="baseline")


def test_the_matched_mask_hits_the_requested_approval_rate():
    probabilities = np.linspace(0.0, 1.0, 1000)
    mask = matched_adverse_mask(probabilities, approval_rate=0.75)
    assert abs((~mask).mean() - 0.75) < 0.01


def test_a_more_lenient_candidate_gains_nothing_from_being_lenient():
    """THE load-bearing test of this milestone.

    Candidate B's probabilities are candidate A's, halved. B ranks applicants
    identically -- it is the same model wearing a lower scale -- so it must
    score identically. At a FIXED threshold it would not: B approves everyone
    and its ratio would read near 1.0, and a search ranking on that number
    would reliably select whichever candidate approves the most people.
    """
    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 0.4, size=1000)
    b = a * 0.5

    target = float((a <= config.RISK_BAND_LOW_MAX).mean())

    mask_a = matched_adverse_mask(a, target)
    mask_b = matched_adverse_mask(b, target)

    assert np.array_equal(mask_a, mask_b)

    # And the naive alternative really would have differed, which is what makes
    # this test meaningful rather than vacuous.
    assert not np.array_equal(a > config.RISK_BAND_LOW_MAX, b > config.RISK_BAND_LOW_MAX)


def test_the_matched_mask_is_invariant_to_any_monotone_rescaling():
    rng = np.random.default_rng(1)
    probabilities = rng.uniform(0.01, 0.99, size=500)
    rescaled = probabilities**2  # strictly increasing on (0, 1)
    assert np.array_equal(
        matched_adverse_mask(probabilities, 0.6), matched_adverse_mask(rescaled, 0.6)
    )


@pytest.mark.slow
def test_rank_scores_every_candidate_and_puts_the_baseline_first(fixture_path):
    frame = pd.read_csv(fixture_path)
    ranking = rank(frame, min_group_size=10)

    assert isinstance(ranking, Ranking)
    assert [c.name for c in ranking.candidates][0] == BASELINE.name
    assert len(ranking.candidates) == len(CANDIDATES)
    assert 0.0 <= ranking.target_approval_rate <= 1.0
    for candidate in ranking.candidates:
        assert (candidate.failed_reason is None) != (candidate.roc_auc is None)


@pytest.mark.slow
def test_rank_is_deterministic(fixture_path):
    frame = pd.read_csv(fixture_path)
    first = rank(frame, min_group_size=10)
    second = rank(frame, min_group_size=10)
    assert first == second
