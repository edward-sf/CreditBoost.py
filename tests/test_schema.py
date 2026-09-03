import pytest
from pydantic import ValidationError

from creditboost import config
from creditboost.schema import (
    AttributeFairness,
    CandidateResult,
    FairnessReport,
    GroupRate,
    ModelMetadata,
    PredictRequest,
    PredictResponse,
    SearchReport,
)
from tests.conftest import a_passing_fairness_report


def minimal_payload() -> dict:
    return {"AMT_INCOME_TOTAL": 100_000.0, "AMT_CREDIT": 400_000.0, "DAYS_BIRTH": -12000}


def test_minimal_payload_is_accepted():
    """A thin-file borrower supplies very little; only three fields are required."""
    request = PredictRequest(**minimal_payload())
    assert request.AMT_INCOME_TOTAL == 100_000.0
    assert request.EXT_SOURCE_1 is None


def test_omitted_child_count_stays_missing_rather_than_becoming_zero():
    """An unanswered field must not be imputed: 0 children is a claim, not a non-answer."""
    assert PredictRequest(**minimal_payload()).CNT_CHILDREN is None


def test_request_exposes_exactly_the_configured_fields():
    assert set(PredictRequest.model_fields) == set(config.REQUEST_FIELDS)


def test_gender_is_not_an_accepted_field():
    assert "CODE_GENDER" not in PredictRequest.model_fields


@pytest.mark.parametrize("field", ["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH"])
def test_omitting_a_required_field_is_rejected(field):
    payload = minimal_payload()
    del payload[field]
    with pytest.raises(ValidationError):
        PredictRequest(**payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("AMT_INCOME_TOTAL", -1.0),
        ("AMT_INCOME_TOTAL", 0.0),
        ("AMT_CREDIT", -5.0),
        ("AMT_ANNUITY", 0.0),
        ("EXT_SOURCE_1", 1.5),
        ("EXT_SOURCE_2", -0.1),
        ("CNT_CHILDREN", -1),
        ("DAYS_BIRTH", 100),
    ],
)
def test_out_of_range_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        PredictRequest(**(minimal_payload() | {field: value}))


def test_days_employed_accepts_the_positive_sentinel():
    """365243 is a valid input meaning 'not employed'; the transform scrubs it."""
    request = PredictRequest(**(minimal_payload() | {"DAYS_EMPLOYED": 365243}))
    assert request.DAYS_EMPLOYED == 365243


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(**(minimal_payload() | {"FAVOURITE_COLOUR": "blue"}))


def test_response_carries_the_model_version():
    response = PredictResponse(probability=0.2, risk_band="medium", model_version="0.1.0")
    assert response.model_dump()["model_version"] == "0.1.0"


def test_metadata_round_trips_through_json():
    metadata = ModelMetadata(
        version="0.1.0",
        trained_at="2026-08-30T12:00:00Z",
        dataset_sha256="abc123",
        n_train_rows=160,
        feature_order=list(config.FEATURE_ORDER),
        metrics={"roc_auc": 0.75, "pr_auc": 0.24, "brier": 0.068},
        xgboost_version="2.1.0",
        provenance="fixture",
        fairness=a_passing_fairness_report(),
    )
    assert ModelMetadata.model_validate_json(metadata.model_dump_json()) == metadata


def test_metadata_rejects_an_unknown_provenance():
    with pytest.raises(ValidationError):
        ModelMetadata(
            version="0.1.0",
            trained_at="2026-08-30T12:00:00Z",
            dataset_sha256="abc123",
            n_train_rows=160,
            feature_order=list(config.FEATURE_ORDER),
            metrics={},
            xgboost_version="2.1.0",
            provenance="guesswork",
            fairness=a_passing_fairness_report(),
        )


def test_a_measured_attribute_is_accepted():
    attribute = AttributeFairness(
        attribute="CODE_GENDER",
        adverse_impact_ratio=0.87,
        groups=[
            GroupRate(group="F", adverse_rate=0.222, n=40561),
            GroupRate(group="M", adverse_rate=0.325, n=20940),
        ],
    )
    assert attribute.unmeasured_reason is None


def test_an_unmeasured_attribute_is_accepted():
    attribute = AttributeFairness(
        attribute="NAME_FAMILY_STATUS",
        unmeasured_reason="fewer than two groups reached the minimum size of 100",
    )
    assert attribute.adverse_impact_ratio is None


def test_an_attribute_cannot_be_both_measured_and_unmeasured():
    """'Not measured' reading as 'passed' is the failure that would make the
    whole report worthless, so the two states are mutually exclusive."""

    with pytest.raises(ValidationError):
        AttributeFairness(
            attribute="CODE_GENDER",
            adverse_impact_ratio=0.87,
            unmeasured_reason="also unmeasured, somehow",
        )


def test_an_attribute_must_be_one_or_the_other():
    with pytest.raises(ValidationError):
        AttributeFairness(attribute="CODE_GENDER")


def test_a_ratio_above_one_is_rejected():
    """min/max over favourable rates cannot exceed 1. A value above it means the
    ratio was computed upside down."""

    with pytest.raises(ValidationError):
        AttributeFairness(attribute="CODE_GENDER", adverse_impact_ratio=1.23)


def test_a_ratio_below_zero_is_rejected():
    """A ratio of two rates in [0, 1] can never be negative. A negative value
    means the ratio was computed from something other than favourable rates."""

    with pytest.raises(ValidationError):
        AttributeFairness(attribute="CODE_GENDER", adverse_impact_ratio=-0.1)


def test_a_fairness_report_round_trips_through_json():
    report = FairnessReport(
        adverse_definition="band != low",
        band_low_max=0.10,
        min_group_size=100,
        attributes=[
            AttributeFairness(
                attribute="CODE_GENDER",
                adverse_impact_ratio=0.868,
                groups=[GroupRate(group="F", adverse_rate=0.222, n=40561)],
            )
        ],
    )

    assert FairnessReport.model_validate_json(report.model_dump_json()) == report


def a_candidate(name="baseline", auc=0.75, air=0.81):
    return CandidateResult(
        name=name,
        n_features=20,
        roc_auc=auc,
        min_adverse_impact_ratio=air,
        adverse_impact_ratios={"CODE_GENDER": 0.87, "DAYS_BIRTH": air},
    )


def a_report(candidates=None, baseline="baseline", selected="baseline"):
    candidates = candidates if candidates is not None else [a_candidate()]
    return SearchReport(
        baseline=baseline,
        selected=selected,
        auc_budget=0.01,
        min_air_improvement=0.01,
        target_approval_rate=0.743,
        ranking_basis="matched approval rate on the selection split",
        candidates=candidates,
    )


def test_search_report_round_trips_through_json():
    report = a_report()
    assert SearchReport.model_validate_json(report.model_dump_json()) == report


def test_a_failed_candidate_carries_a_reason_and_no_scores():
    result = CandidateResult(name="empty", n_features=0, failed_reason="no features remain")
    assert result.roc_auc is None
    assert result.min_adverse_impact_ratio is None
    assert result.adverse_impact_ratios == {}


def test_a_candidate_cannot_be_both_scored_and_failed():
    with pytest.raises(ValidationError):
        CandidateResult(
            name="x",
            n_features=3,
            roc_auc=0.7,
            min_adverse_impact_ratio=0.9,
            failed_reason="boom",
        )


def test_a_candidate_must_be_either_scored_or_failed():
    # A default of 0.0 would silently drag the recorded frontier downward and
    # misrepresent what the search found.
    with pytest.raises(ValidationError):
        CandidateResult(name="x", n_features=3)


def test_a_candidate_cannot_have_only_roc_auc_without_air():
    """Both score fields must be set together. A half-score is invalid."""
    with pytest.raises(ValidationError):
        CandidateResult(name="x", n_features=3, roc_auc=0.7)


def test_a_candidate_cannot_have_only_air_without_roc_auc():
    """Both score fields must be set together. A half-score is invalid."""
    with pytest.raises(ValidationError):
        CandidateResult(name="x", n_features=3, min_adverse_impact_ratio=0.9)


def test_a_candidate_cannot_have_roc_auc_with_failed_reason():
    """A candidate with only one score field set cannot also have a failure reason."""
    with pytest.raises(ValidationError):
        CandidateResult(name="x", n_features=3, roc_auc=0.7, failed_reason="boom")


def test_a_candidate_cannot_have_air_with_failed_reason():
    """A candidate claiming a fairness score cannot also claim it failed."""
    with pytest.raises(ValidationError):
        CandidateResult(name="x", n_features=3, min_adverse_impact_ratio=0.9, failed_reason="boom")


def test_selected_must_name_a_candidate_in_the_frontier():
    with pytest.raises(ValidationError):
        a_report(selected="a-candidate-that-was-never-run")


def test_baseline_must_name_a_candidate_in_the_frontier():
    with pytest.raises(ValidationError):
        a_report(baseline="not-in-the-list", selected="baseline")


def test_model_metadata_loads_without_a_selection_report():
    metadata = ModelMetadata(
        version="0.4.0",
        trained_at="2026-09-02T00:00:00+00:00",
        dataset_sha256="0" * 64,
        n_train_rows=100,
        feature_order=["EXT_SOURCE_1"],
        metrics={"roc_auc": 0.75},
        xgboost_version="3.4.1",
        provenance="fixture",
        fairness=a_passing_fairness_report(),
    )
    assert metadata.selection is None
