import pandas as pd
import pytest

from creditboost import config
from creditboost.fairness import adverse_impact_ratios, evaluate, failing_attributes
from creditboost.schema import AttributeFairness, FairnessReport


def frame_and_probabilities(
    groups: list[str],
    probabilities: list[float],
    attribute: str = "CODE_GENDER",
) -> tuple[pd.DataFrame, list[float]]:
    """Build a frame carrying every fairness attribute, varying only the one
    under test. The others are held constant so they fall to a single group and
    are recorded unmeasured, which keeps each test about one thing."""
    frame = pd.DataFrame(
        {
            "CODE_GENDER": "F",
            "DAYS_BIRTH": -12_000,
            "NAME_FAMILY_STATUS": "Married",
        },
        index=range(len(groups)),
    )
    frame[attribute] = groups
    return frame, probabilities


def attribute_fairness(attribute_name: str, report: FairnessReport) -> AttributeFairness:
    found = [a for a in report.attributes if a.attribute == attribute_name]
    assert found, f"{attribute_name} missing from the report"
    return found[0]


def test_the_ratio_is_min_over_max_not_max_over_min():
    """The load-bearing test. Inverted, a model at 0.81 reads as 1.23 and every
    model passes the gate -- a silent, total failure of the whole milestone.

    Group A: 2 of 10 adverse -> favourable 0.8.  Group B: 6 of 10 adverse ->
    favourable 0.4.  min/max = 0.4/0.8 = 0.5, which must be < 1.
    """
    groups = ["A"] * 10 + ["B"] * 10
    probs = ([0.5] * 2 + [0.01] * 8) + ([0.5] * 6 + [0.01] * 4)
    frame, probabilities = frame_and_probabilities(groups, probs)

    report = evaluate(frame, probabilities, min_group_size=5)

    assert attribute_fairness("CODE_GENDER", report).adverse_impact_ratio == pytest.approx(0.5)


def test_adverse_means_not_banded_low_not_merely_high():
    """A medium-banded applicant is adverse. Counting only `high` is the
    definition this design rejects: it cannot discriminate at a high approval
    rate. 0.15 bands medium under the default thresholds, never high."""
    groups = ["A"] * 10 + ["B"] * 10
    probs = ([0.15] * 10) + ([0.01] * 10)
    frame, probabilities = frame_and_probabilities(groups, probs)

    report = evaluate(frame, probabilities, min_group_size=5)

    measured = attribute_fairness("CODE_GENDER", report)
    by_group = {g.group: g.adverse_rate for g in measured.groups}
    assert by_group["A"] == pytest.approx(1.0), "medium must count as adverse"
    assert by_group["B"] == pytest.approx(0.0)
    assert measured.adverse_impact_ratio == pytest.approx(0.0)


def test_groups_below_the_minimum_size_are_excluded():
    groups = ["A"] * 10 + ["B"] * 3
    probs = [0.01] * 13
    frame, probabilities = frame_and_probabilities(groups, probs)

    report = evaluate(frame, probabilities, min_group_size=5)

    measured = attribute_fairness("CODE_GENDER", report)
    assert [g.group for g in measured.groups] == ["A"]
    assert measured.adverse_impact_ratio is None
    assert "minimum size" in measured.unmeasured_reason


def test_an_unmeasured_attribute_never_fails_the_gate():
    """Unmeasured establishes nothing either way. It must not be reported as a
    failure, and it must not be reported as a pass."""
    groups = ["A"] * 10 + ["B"] * 3
    probs = [0.01] * 13
    frame, probabilities = frame_and_probabilities(groups, probs)

    report = evaluate(frame, probabilities, min_group_size=5)

    assert failing_attributes(report) == []


def test_a_ratio_below_the_floor_is_reported_failing():
    groups = ["A"] * 10 + ["B"] * 10
    probs = ([0.5] * 5 + [0.01] * 5) + ([0.01] * 10)
    frame, probabilities = frame_and_probabilities(groups, probs)

    report = evaluate(frame, probabilities, min_group_size=5)

    failures = failing_attributes(report)
    assert [a.attribute for a in failures] == ["CODE_GENDER"]


def test_exactly_the_floor_passes():
    """0.80 is the threshold, and the comparison is strictly less-than, so a
    model landing exactly on it is not failed."""
    groups = ["A"] * 10 + ["B"] * 10
    probs = ([0.5] * 2 + [0.01] * 8) + ([0.01] * 10)
    frame, probabilities = frame_and_probabilities(groups, probs)

    report = evaluate(frame, probabilities, min_group_size=5)

    assert attribute_fairness("CODE_GENDER", report).adverse_impact_ratio == pytest.approx(0.8)
    assert failing_attributes(report) == []


def test_age_is_bucketed_at_the_ecoa_line():
    """15 U.S.C. 1691(a)(1) protects applicants 62 and over, so that is the
    boundary -- not a quantile of whatever population happens to be present."""
    older = -int(70 * 365.25)
    younger = -int(30 * 365.25)
    groups = [older] * 10 + [younger] * 10
    frame, probabilities = frame_and_probabilities(groups, [0.01] * 20, attribute="DAYS_BIRTH")

    report = evaluate(frame, probabilities, min_group_size=5)

    labels = {g.group for g in attribute_fairness("DAYS_BIRTH", report).groups}
    assert labels == {"62 and over", "under 62"}


def test_a_missing_age_is_excluded_rather_than_bucketed_as_young():
    """NaN >= 62 is False, so a careless implementation files unknown ages under
    'under 62' and reports a rate for people it knows nothing about."""
    older = -int(70 * 365.25)
    groups = [older] * 10 + [None] * 10
    frame, probabilities = frame_and_probabilities(groups, [0.01] * 20, attribute="DAYS_BIRTH")

    report = evaluate(frame, probabilities, min_group_size=5)

    measured = attribute_fairness("DAYS_BIRTH", report)
    assert [g.group for g in measured.groups] == ["62 and over"]
    assert measured.groups[0].n == 10


def test_a_report_records_the_policy_it_was_measured_under():
    groups = ["A"] * 10 + ["B"] * 10
    frame, probabilities = frame_and_probabilities(groups, [0.01] * 20)

    report = evaluate(frame, probabilities, min_group_size=5)

    assert report.band_low_max == config.RISK_BAND_LOW_MAX
    assert report.adverse_definition == "band != low"
    assert report.min_group_size == 5


def test_every_fairness_attribute_appears_in_the_report():
    groups = ["A"] * 10 + ["B"] * 10
    frame, probabilities = frame_and_probabilities(groups, [0.01] * 20)

    report = evaluate(frame, probabilities, min_group_size=5)

    assert [a.attribute for a in report.attributes] == list(config.FAIRNESS_ATTRIBUTES)


def test_a_wholly_adverse_population_is_unmeasured_not_zero():
    """Every group entirely adverse makes max favourable 0 and the ratio 0/0.
    A degenerate model the gate cannot speak to, not a failure it can assert."""
    groups = ["A"] * 10 + ["B"] * 10
    frame, probabilities = frame_and_probabilities(groups, [0.9] * 20)

    report = evaluate(frame, probabilities, min_group_size=5)

    measured = attribute_fairness("CODE_GENDER", report)
    assert measured.adverse_impact_ratio is None
    assert "favourable" in measured.unmeasured_reason


def test_adverse_impact_ratios_is_driven_by_the_mask_it_is_given():
    """Same frame, two different adverse masks, two different ratios. This is
    what lets the search score a candidate at a threshold of its own choosing
    without reimplementing any of the grouping or the min/max direction."""
    frame, _ = frame_and_probabilities(["F"] * 200 + ["M"] * 200, [0.0] * 400)

    everyone_adverse = [True] * len(frame)
    nobody_adverse = [False] * len(frame)

    all_adverse = adverse_impact_ratios(frame, everyone_adverse, min_group_size=10)
    none_adverse = adverse_impact_ratios(frame, nobody_adverse, min_group_size=10)

    sex_all = next(a for a in all_adverse if a.attribute == "CODE_GENDER")
    sex_none = next(a for a in none_adverse if a.attribute == "CODE_GENDER")

    # Everyone adverse is the degenerate case: no favourable outcome anywhere.
    assert sex_all.adverse_impact_ratio is None
    assert sex_all.unmeasured_reason is not None
    # Nobody adverse is perfect parity.
    assert sex_none.adverse_impact_ratio == 1.0


def test_evaluate_delegates_to_adverse_impact_ratios():
    """evaluate is exactly adverse_impact_ratios over a band-derived mask, so a
    caller passing the equivalent mask gets an identical answer."""
    from creditboost.banding import risk_band

    probabilities = [0.05 + 0.4 * (i % 3 == 0) for i in range(400)]
    frame, _ = frame_and_probabilities(
        ["F"] * 200 + ["M"] * 200,
        probabilities,
    )

    via_evaluate = evaluate(frame, probabilities, min_group_size=10).attributes
    mask = [risk_band(p) != "low" for p in probabilities]
    via_mask = adverse_impact_ratios(frame, mask, min_group_size=10)

    assert via_evaluate == via_mask
