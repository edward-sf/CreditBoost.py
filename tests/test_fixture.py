import pandas as pd
import pytest

from creditboost import config

FIXTURE = "tests/fixtures/sample.csv"


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


def test_fixture_has_every_request_field_and_the_target(frame):
    for column in config.REQUEST_FIELDS:
        assert column in frame.columns
    assert config.TARGET_COLUMN in frame.columns


def test_fixture_has_200_rows(frame):
    assert len(frame) == 200


def test_fixture_contains_both_target_classes(frame):
    assert set(frame[config.TARGET_COLUMN].unique()) == {0, 1}


def test_fixture_contains_the_not_employed_sentinel(frame):
    """The 365243 sentinel must be present so the scrub is exercised."""
    assert (frame["DAYS_EMPLOYED"] == config.DAYS_EMPLOYED_SENTINEL).sum() >= 5


def test_fixture_contains_rows_with_all_external_scores_missing(frame):
    """The thin-file borrower: no external credit score at all."""
    all_missing = frame[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isna().all(axis=1)
    assert all_missing.sum() >= 5


def test_fixture_categorical_values_are_declared_levels(frame):
    for column, levels in config.CATEGORICAL_LEVELS.items():
        present = set(frame[column].dropna().unique())
        assert present <= set(levels), f"{column} has undeclared levels: {present - set(levels)}"


def test_generator_is_deterministic():
    """Regenerating must not produce a spurious diff."""
    from tests.fixtures.generate_fixture import build_fixture

    first = build_fixture()
    second = build_fixture()
    pd.testing.assert_frame_equal(first, second)


def test_fixture_has_every_fairness_attribute(frame):
    """CODE_GENDER is not a request field -- the service never accepts it -- but
    fairness measurement needs it, so the fixture carries it anyway."""
    for column in config.FAIRNESS_ATTRIBUTES:
        assert column in frame.columns


def test_fixture_gender_groups_are_both_large_enough_to_measure(frame):
    """A 200-row fixture split evenly gives 100 per group, which is exactly the
    default minimum. Drawing it randomly could land at 95/105 and make the
    integration test flaky, so the split is deterministic."""
    counts = frame["CODE_GENDER"].value_counts()
    assert set(counts.index) == {"F", "M"}
    assert counts.min() == 100
