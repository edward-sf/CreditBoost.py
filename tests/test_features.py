import numpy as np
import pandas as pd
import pytest

from creditboost import config
from creditboost.features import FEATURE_ORDER, transform


def base_record() -> dict:
    return {
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.6,
        "EXT_SOURCE_3": 0.7,
        "AMT_INCOME_TOTAL": 100_000.0,
        "AMT_CREDIT": 400_000.0,
        "AMT_ANNUITY": 20_000.0,
        "AMT_GOODS_PRICE": 380_000.0,
        "DAYS_EMPLOYED": -2000,
        "DAYS_BIRTH": -12000,
        "CNT_CHILDREN": 1,
        "CNT_FAM_MEMBERS": 3.0,
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "N",
        "NAME_CONTRACT_TYPE": "Cash loans",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "OCCUPATION_TYPE": "Managers",
    }


def test_output_columns_are_exactly_feature_order():
    result = transform([base_record()])
    assert list(result.columns) == list(FEATURE_ORDER)


def test_raw_age_is_not_in_the_output():
    assert "DAYS_BIRTH" not in transform([base_record()]).columns


def test_marital_status_is_not_in_the_output():
    """base_record() still supplies NAME_FAMILY_STATUS, exactly as a real caller
    may. The transform must ignore it: accepted is not the same as modelled."""
    assert "NAME_FAMILY_STATUS" not in transform([base_record()]).columns


def test_not_employed_sentinel_becomes_nan():
    record = base_record() | {"DAYS_EMPLOYED": config.DAYS_EMPLOYED_SENTINEL}
    assert np.isnan(transform([record])["DAYS_EMPLOYED"].iloc[0])


def test_sentinel_is_scrubbed_before_the_ratio_is_derived():
    """A 365243 tenure must not leak into employed_to_age as a plausible number."""
    record = base_record() | {"DAYS_EMPLOYED": config.DAYS_EMPLOYED_SENTINEL}
    assert np.isnan(transform([record])["employed_to_age"].iloc[0])


def test_missing_external_scores_stay_nan_and_are_not_imputed():
    record = base_record() | {"EXT_SOURCE_1": None, "EXT_SOURCE_2": None, "EXT_SOURCE_3": None}
    row = transform([record]).iloc[0]
    for column in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        assert np.isnan(row[column]), f"{column} was imputed"


def test_derived_ratios_are_computed_correctly():
    row = transform([base_record()]).iloc[0]
    assert row["credit_to_income"] == pytest.approx(4.0)
    assert row["annuity_to_income"] == pytest.approx(0.2)
    assert row["employed_to_age"] == pytest.approx(2000 / 12000)


def test_zero_denominator_yields_nan_rather_than_raising():
    record = base_record() | {"AMT_INCOME_TOTAL": 0.0}
    row = transform([record]).iloc[0]
    assert np.isnan(row["credit_to_income"])
    assert np.isnan(row["annuity_to_income"])


def test_binary_flags_map_to_one_and_zero():
    row = transform([base_record()]).iloc[0]
    assert row["FLAG_OWN_CAR"] == 1.0
    assert row["FLAG_OWN_REALTY"] == 0.0


def test_unrecognised_binary_flag_becomes_nan():
    record = base_record() | {"FLAG_OWN_CAR": "maybe"}
    assert np.isnan(transform([record])["FLAG_OWN_CAR"].iloc[0])


def test_unknown_categorical_level_becomes_nan_without_raising():
    record = base_record() | {"OCCUPATION_TYPE": "Astronaut"}
    result = transform([record])
    assert pd.isna(result["OCCUPATION_TYPE"].iloc[0])


def test_categorical_columns_carry_the_declared_levels():
    result = transform([base_record()])
    for column, levels in config.CATEGORICAL_LEVELS.items():
        assert list(result[column].cat.categories) == list(levels)


def test_missing_input_column_becomes_nan_rather_than_raising():
    record = base_record()
    del record["AMT_GOODS_PRICE"]
    assert np.isnan(transform([record])["AMT_GOODS_PRICE"].iloc[0])


def test_transform_is_deterministic():
    first = transform([base_record()])
    second = transform([base_record()])
    pd.testing.assert_frame_equal(first, second)


def test_empty_input_returns_empty_frame_with_correct_columns():
    result = transform([])
    assert len(result) == 0
    assert list(result.columns) == list(FEATURE_ORDER)


def test_dict_and_dataframe_paths_produce_identical_matrices():
    """The parity guarantee: serving and training cannot diverge."""
    record = base_record()
    from_api = transform([record])
    from_training = transform(pd.DataFrame([record]))
    pd.testing.assert_frame_equal(from_api, from_training)


def test_parity_holds_across_the_whole_fixture():
    frame = pd.read_csv("tests/fixtures/sample.csv")
    records = frame.to_dict(orient="records")
    pd.testing.assert_frame_equal(
        transform(frame).reset_index(drop=True),
        transform(records).reset_index(drop=True),
    )
