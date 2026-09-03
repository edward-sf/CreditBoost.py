"""The alternative search: the catalog, the spec transform, the matched mask,
the selection rule, and the ranking."""

import dataclasses

import pandas as pd
import pytest

from creditboost import config
from creditboost.features import transform
from creditboost.search import BASELINE, CANDIDATES, CandidateSpec, UnknownFeatureError, apply


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
