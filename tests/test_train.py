import pytest

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


def test_training_stamps_a_fairness_report(fixture_path, tmp_path):
    """Every shipped model carries measured disparate impact -- that is what
    making the metadata field required buys."""
    import json

    from creditboost import config
    from creditboost.train import main

    model_path = tmp_path / "model.json"
    meta_path = tmp_path / "meta.json"
    code = main(
        [
            "--data",
            str(fixture_path),
            "--model-out",
            str(model_path),
            "--metadata-out",
            str(meta_path),
            "--provenance",
            "fixture",
        ]
    )

    assert code == 0
    report = json.loads(meta_path.read_text())["fairness"]
    assert report["band_low_max"] == config.RISK_BAND_LOW_MAX
    assert report["adverse_definition"] == "band != low"
    assert [a["attribute"] for a in report["attributes"]] == list(config.FAIRNESS_ATTRIBUTES)


def test_a_failing_ratio_writes_no_artifact(fixture_path, tmp_path, monkeypatch):
    """The gate, proven by forcing a failure rather than hoping for one. A real
    model that fails is exactly what this must refuse to write."""
    from creditboost import train
    from creditboost.schema import AttributeFairness, FairnessReport, GroupRate

    failing = FairnessReport(
        adverse_definition="band != low",
        band_low_max=0.10,
        min_group_size=100,
        attributes=[
            AttributeFairness(
                attribute="CODE_GENDER",
                adverse_impact_ratio=0.61,
                groups=[
                    GroupRate(group="F", adverse_rate=0.20, n=500),
                    GroupRate(group="M", adverse_rate=0.51, n=500),
                ],
            )
        ],
    )
    monkeypatch.setattr(train, "evaluate", lambda *a, **k: failing)

    model_path = tmp_path / "model.json"
    meta_path = tmp_path / "meta.json"
    code = train.main(
        [
            "--data",
            str(fixture_path),
            "--model-out",
            str(model_path),
            "--metadata-out",
            str(meta_path),
            "--provenance",
            "fixture",
        ]
    )

    assert code == 1
    assert not model_path.exists()
    assert not meta_path.exists()
