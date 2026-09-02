import pytest
from fastapi.testclient import TestClient

from creditboost import config
from creditboost.artifact import FeatureOrderMismatchError
from creditboost.data import load_training_frame, split
from creditboost.schema import ModelMetadata
from creditboost.serve.app import create_app
from creditboost.train import fit


@pytest.fixture(scope="module")
def artifact_paths(fixture_path, tmp_path_factory):
    from creditboost.artifact import save

    directory = tmp_path_factory.mktemp("artifact")
    model_path, meta_path = directory / "model.json", directory / "meta.json"

    frame = load_training_frame(fixture_path)
    train_frame, valid_frame = split(frame)
    booster, metrics = fit(train_frame, valid_frame)

    save(
        booster,
        ModelMetadata(
            version=config.MODEL_VERSION,
            trained_at="2026-08-30T12:00:00Z",
            dataset_sha256="fixture",
            n_train_rows=len(train_frame),
            feature_order=list(config.FEATURE_ORDER),
            metrics=metrics,
            xgboost_version="test",
            provenance="fixture",
        ),
        model_path,
        meta_path,
    )
    return model_path, meta_path


@pytest.fixture(scope="module")
def client(artifact_paths):
    with TestClient(create_app(*artifact_paths)) as test_client:
        yield test_client


def minimal_payload() -> dict:
    return {"AMT_INCOME_TOTAL": 100_000.0, "AMT_CREDIT": 400_000.0, "DAYS_BIRTH": -12000}


def test_health_reports_the_model_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_version"] == config.MODEL_VERSION


def test_metadata_exposes_provenance_and_feature_order(client):
    body = client.get("/metadata").json()
    assert body["provenance"] == "fixture"
    assert body["feature_order"] == list(config.FEATURE_ORDER)


def test_predict_returns_a_probability_band_and_version(client):
    body = client.post("/predict", json=minimal_payload()).json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}
    assert body["model_version"] == config.MODEL_VERSION


def test_predict_band_agrees_with_the_configured_thresholds(client):
    from creditboost.banding import risk_band

    body = client.post("/predict", json=minimal_payload()).json()
    assert body["risk_band"] == risk_band(body["probability"])


def test_thin_file_borrower_with_no_external_scores_is_scored(client):
    """The product's whole purpose: score someone with no credit score."""
    payload = minimal_payload() | {
        "EXT_SOURCE_1": None,
        "EXT_SOURCE_2": None,
        "EXT_SOURCE_3": None,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_unemployed_applicant_sentinel_is_accepted(client):
    payload = minimal_payload() | {"DAYS_EMPLOYED": config.DAYS_EMPLOYED_SENTINEL}
    assert client.post("/predict", json=payload).status_code == 200


def test_unknown_occupation_degrades_instead_of_failing(client):
    payload = minimal_payload() | {"OCCUPATION_TYPE": "Astronaut"}
    assert client.post("/predict", json=payload).status_code == 200


def test_missing_required_field_is_rejected(client):
    payload = minimal_payload()
    del payload["AMT_CREDIT"]
    assert client.post("/predict", json=payload).status_code == 422


def test_negative_income_is_rejected(client):
    payload = minimal_payload() | {"AMT_INCOME_TOTAL": -1.0}
    assert client.post("/predict", json=payload).status_code == 422


def test_startup_fails_when_the_artifact_is_missing(tmp_path):
    """A container that cannot score correctly must never accept traffic."""
    application = create_app(tmp_path / "absent.json", tmp_path / "absent_meta.json")
    with pytest.raises(FileNotFoundError), TestClient(application):
        pass


def test_startup_fails_on_a_feature_order_mismatch(artifact_paths, tmp_path):
    import json
    import shutil

    model_path, meta_path = artifact_paths
    stale_model, stale_meta = tmp_path / "model.json", tmp_path / "meta.json"
    shutil.copy(model_path, stale_model)

    metadata = json.loads(meta_path.read_text())
    metadata["feature_order"] = metadata["feature_order"][:-1]
    stale_meta.write_text(json.dumps(metadata))

    with pytest.raises(FeatureOrderMismatchError), TestClient(create_app(stale_model, stale_meta)):
        pass


def test_prediction_logs_carry_no_applicant_financial_data(client):
    """Financial fields are exactly the PII that must not accumulate in log
    aggregation. Only band, version, latency, and request id may be logged.

    Inspects the actual formatted JSON the production JsonFormatter emits --
    not record.getMessage(), which is a constant string ("prediction
    served"). Every field the app logs travels via `extra=`, so a check of
    getMessage() alone is blind to all of it and would pass even if the app
    logged the applicant's income directly. A temporary handler carrying the
    real JsonFormatter is attached to the "creditboost.serve" logger for the
    duration of the request, and the assertions run against its rendered
    output -- the same bytes a running container would write to stdout.
    """
    import io
    import json
    import logging

    from creditboost.serve.logging_config import JsonFormatter

    distinctive_income = 823456.0
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    serve_logger = logging.getLogger("creditboost.serve")
    serve_logger.addHandler(handler)
    try:
        response = client.post(
            "/predict", json=minimal_payload() | {"AMT_INCOME_TOTAL": distinctive_income}
        )
        assert response.status_code == 200
    finally:
        serve_logger.removeHandler(handler)

    emitted_lines = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    served = [line for line in emitted_lines if line.get("message") == "prediction served"]
    assert served, "expected a 'prediction served' log line"
    record = served[-1]

    emitted_text = stream.getvalue()
    assert "823456" not in emitted_text
    for field in ("AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "EXT_SOURCE_1"):
        assert field not in emitted_text

    # The permitted fields must actually be present -- this is what makes the
    # test able to fail: a guard that never checks for anything positive
    # would also pass if logging emitted nothing at all.
    for expected_key in ("request_id", "latency_ms", "model_version", "risk_band"):
        assert expected_key in record, f"expected {expected_key!r} in {record!r}"


def test_predict_returns_at_most_four_reasons(client):
    response = client.post("/predict", json=minimal_payload())

    assert response.status_code == 200
    reasons = response.json()["reasons"]
    assert 0 <= len(reasons) <= 4


def test_every_reason_has_a_code_and_a_description(client):
    reasons = client.post("/predict", json=minimal_payload()).json()["reasons"]

    catalog = {text.code for text in config.REASON_TEXT.values()}
    for reason in reasons:
        assert reason["code"] in catalog
        assert reason["description"]


def test_the_same_request_yields_the_same_reasons(client):
    payload = minimal_payload()

    first = client.post("/predict", json=payload).json()["reasons"]
    second = client.post("/predict", json=payload).json()["reasons"]

    assert first == second


def test_an_applicant_with_no_external_scores_is_told_exactly_that(client):
    """The thin-file case the service exists for. If external_credit ranks at
    all, it must say no score is on file -- never that the score is low."""
    # minimal_payload() carries the three required fields only, so all three
    # external scores are already absent -- which is the thin-file case exactly.
    reasons = client.post("/predict", json=minimal_payload()).json()["reasons"]

    external = [r for r in reasons if r["code"] == "EXTERNAL_CREDIT"]
    for reason in external:
        assert reason["description"] == "No external credit score on file"


def test_the_not_employed_sentinel_reads_as_no_employment_history(client):
    """365243 is scrubbed to NaN by the transform, so an unemployed applicant
    must never be told their employment is merely short."""
    payload = minimal_payload() | {"DAYS_EMPLOYED": config.DAYS_EMPLOYED_SENTINEL}

    reasons = client.post("/predict", json=payload).json()["reasons"]

    employment = [r for r in reasons if r["code"] == "EMPLOYMENT_TENURE"]
    for reason in employment:
        assert reason["description"] == "No employment history on record"


def test_marital_status_is_still_accepted_and_never_reported(client):
    """Quarantine, end to end: the field is accepted without error and cannot
    appear in any disclosure, because it is not a feature at all."""
    payload = minimal_payload() | {"NAME_FAMILY_STATUS": "Widow"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    text = " ".join(r["description"] for r in response.json()["reasons"]).lower()
    for term in ("marital", "widow", "married", "spouse"):
        assert term not in text


def test_contributions_plus_bias_reconstruct_the_predicted_probability(client):
    """pred_contribs returns n_features + 1 values, bias last. An off-by-one in
    that row produces reasons that are entirely plausible and entirely wrong --
    a failure with no natural symptom, so it gets an explicit test.

    Summing the whole row (contributions plus bias) gives the margin; the
    logistic of that margin must equal the probability the service reports.
    """
    import math

    import xgboost as xgb

    from creditboost.features import transform
    from creditboost.serve import deps

    payload = minimal_payload()
    probability = client.post("/predict", json=payload).json()["probability"]

    frame = transform([payload])
    matrix = xgb.DMatrix(frame, enable_categorical=True)
    row = deps.get_model().booster.predict(matrix, pred_contribs=True)[0]

    assert len(row) == len(config.FEATURE_ORDER) + 1
    reconstructed = 1.0 / (1.0 + math.exp(-float(row.sum())))
    assert reconstructed == pytest.approx(probability, abs=1e-6)
