import json

import pytest
import xgboost as xgb

from creditboost import config
from creditboost.artifact import load
from creditboost.data import load_training_frame, split
from creditboost.train import fit, main


@pytest.fixture(scope="module")
def trained(fixture_path):
    frame = load_training_frame(fixture_path)
    train_frame, valid_frame = split(frame)
    return fit(train_frame, valid_frame)


def test_fit_returns_a_booster_and_the_three_metrics(trained):
    _, metrics = trained
    assert set(metrics) == {"roc_auc", "pr_auc", "brier"}


def test_accuracy_is_not_reported(trained):
    """At an 8% base rate accuracy carries no information; reporting it misleads."""
    _, metrics = trained
    assert "accuracy" not in metrics


def test_metrics_are_in_valid_ranges(trained):
    _, metrics = trained
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["brier"] <= 1.0


@pytest.mark.slow
def test_cli_writes_a_loadable_artifact(fixture_path, tmp_path):
    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    code = main(
        [
            "--data",
            str(fixture_path),
            "--model-out",
            str(model_path),
            "--metadata-out",
            str(meta_path),
            "--min-auc",
            "0.0",
            "--provenance",
            "fixture",
        ]
    )
    assert code == 0

    loaded = load(model_path, meta_path)
    assert loaded.metadata.feature_order == list(config.FEATURE_ORDER)
    assert loaded.metadata.provenance == "fixture"
    assert loaded.metadata.version == config.MODEL_VERSION
    assert loaded.metadata.n_train_rows == 160
    assert loaded.metadata.dataset_sha256


@pytest.mark.slow
def test_cli_refuses_to_write_a_model_below_the_auc_floor(fixture_path, tmp_path):
    """A bad retrain must produce no artifact at all, so nothing downstream
    needs to detect one."""
    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    code = main(
        [
            "--data",
            str(fixture_path),
            "--model-out",
            str(model_path),
            "--metadata-out",
            str(meta_path),
            "--min-auc",
            "1.01",
        ]
    )
    assert code == 1
    assert not model_path.exists()
    assert not meta_path.exists()


def test_committed_artifact_is_present_and_is_production_provenance():
    """Tasks 9 and 10 depend on this artifact existing in the repo, and this
    is what's shipped in the container CI publishes to GHCR: it must be the
    real thing, not a fixture-trained stand-in. `provenance in {"fixture",
    "production"}` would be vacuous -- the Pydantic Literal already
    guarantees that -- so this asserts the actually meaningful claim."""
    metadata = json.loads(config.METADATA_PATH.read_text())
    assert metadata["provenance"] == "production"
    assert metadata["feature_order"] == list(config.FEATURE_ORDER)


def test_committed_artifact_excludes_gender_and_raw_age():
    """ECOA / Regulation B: sex and age are prohibited bases for credit
    decisions. test_config.py asserts this against the source constants;
    this asserts it against the actual committed, shipped artifact -- both
    the metadata sidecar's feature_order and the booster's own feature_names
    baked into model.json -- so a fairness regression in the shipped model
    itself, not just in the code that produced it, would be caught."""
    metadata = json.loads(config.METADATA_PATH.read_text())
    assert "CODE_GENDER" not in metadata["feature_order"]
    assert "DAYS_BIRTH" not in metadata["feature_order"]

    booster = xgb.Booster()
    booster.load_model(str(config.MODEL_PATH))
    feature_names = booster.feature_names or []
    assert "CODE_GENDER" not in feature_names
    assert "DAYS_BIRTH" not in feature_names
