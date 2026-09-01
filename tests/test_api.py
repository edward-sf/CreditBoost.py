import subprocess
import sys

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


def test_prediction_logs_carry_no_applicant_financial_data(client, caplog):
    """Financial fields are exactly the PII that must not accumulate in log
    aggregation. Only band, version, latency, and request id may be logged."""
    import logging

    with caplog.at_level(logging.INFO, logger="creditboost.serve"):
        client.post("/predict", json=minimal_payload() | {"AMT_INCOME_TOTAL": 123456.0})

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "123456" not in logged
    for field in ("AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "EXT_SOURCE_1"):
        assert field not in logged


def test_serving_does_not_import_the_training_stack():
    """Enforces the one-way dependency rule: serve/ never reaches into training,
    which is what keeps scikit-learn out of the runtime image.

    Checks directly for creditboost.data and creditboost.train in sys.modules
    after importing the serving app, rather than checking for a downstream
    package like scikit-learn as a proxy: sklearn's presence depends on
    whichever extras happen to be installed in the venv running the test, and
    on xgboost's own unrelated optional integration with sklearn -- neither of
    which serve/ controls. Naming the training modules themselves sidesteps
    both of those, and -- unlike a proxy check -- also catches a *guarded*
    forbidden import: e.g. a stray
    `try:\n    from ..data import split\nexcept ImportError:\n    pass`
    inside serve/ would silently no-op under a proxy check, but it still
    leaves `creditboost.data` sitting in sys.modules, which this test catches.
    """
    code = (
        "import sys\n"
        "import creditboost.serve.app\n"
        "forbidden = [m for m in ('creditboost.data', 'creditboost.train') if m in sys.modules]\n"
        "assert not forbidden, f'serve/ pulled in forbidden training module(s): {forbidden}'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
