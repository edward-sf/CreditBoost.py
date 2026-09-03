import pytest

from creditboost import config, search
from creditboost.artifact import load
from creditboost.data import load_training_frame, split
from creditboost.schema import CandidateResult, ModelMetadata
from creditboost.train import fit, main
from tests.conftest import a_passing_fairness_report


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


def test_evaluate_is_called_with_the_aligned_validation_frame_and_predictions(
    fixture_path, tmp_path, monkeypatch
):
    """Regression guard for the one property no other test checks: that the
    frame and probabilities handed to `evaluate` are the validation split's own,
    aligned row-for-row.

    The fixture's validation split is only ~40 rows, so every attribute falls
    below `config.MIN_FAIRNESS_GROUP_SIZE` and comes back unmeasured with empty
    `groups` (see test_fixture.py) -- the *ratios* in the stamped report cannot
    catch a wiring mistake here. If `train.py` were changed to pass
    `booster.predict(_matrix(train_frame))` instead of `valid_frame`, or to
    sort or filter the frame before grouping, every existing test would still
    pass: the report would carry plausible-looking numbers computed from the
    wrong rows, and a production retrain would stamp them into a shipped
    artifact with no other symptom. This test intercepts `evaluate`'s actual
    arguments rather than trusting its return value.

    Note: `evaluate`'s `min_group_size` default is bound at its own
    definition in fairness.py, so monkeypatching `config.MIN_FAIRNESS_GROUP_SIZE`
    would not affect it and is not a substitute for this test.
    """
    from creditboost import train

    captured: dict[str, object] = {}

    def spy(frame, probabilities, **kwargs):
        captured["frame"] = frame
        captured["probabilities"] = list(probabilities)
        return a_passing_fairness_report()

    monkeypatch.setattr(train, "evaluate", spy)

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
    assert code == 0

    # Recompute the same seeded split independently to get the expected
    # validation frame, rather than trusting train.py to have used it.
    _, expected_valid_frame = split(load_training_frame(fixture_path))
    assert list(captured["frame"].index) == list(expected_valid_frame.index)

    loaded = load(model_path, meta_path)
    expected_probabilities = loaded.booster.predict(train._matrix(expected_valid_frame)).tolist()
    assert captured["probabilities"] == pytest.approx(expected_probabilities)


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


def test_fit_accepts_a_candidate_spec(fixture_path):
    """A spec with drops produces a model over fewer features."""
    frame = load_training_frame(fixture_path)
    train_frame, valid_frame = split(frame)
    spec = search.CandidateSpec(name="fewer", drops=("CNT_CHILDREN",))
    booster, _ = fit(train_frame, valid_frame, spec=spec)
    assert "CNT_CHILDREN" not in (booster.feature_names or [])


@pytest.mark.slow
def test_search_stamps_a_report_naming_a_real_candidate(tmp_path, fixture_path):
    model_out = tmp_path / "model.json"
    metadata_out = tmp_path / "model_meta.json"
    code = main(
        [
            "--data",
            str(fixture_path),
            "--model-out",
            str(model_out),
            "--metadata-out",
            str(metadata_out),
            "--provenance",
            "fixture",
            "--search",
        ]
    )
    assert code == 0
    metadata = ModelMetadata.model_validate_json(metadata_out.read_text())
    assert metadata.selection is not None
    names = {c.name for c in metadata.selection.candidates}
    assert metadata.selection.selected in names
    assert metadata.selection.baseline == search.BASELINE.name
    assert metadata.selection.ranking_basis == search.RANKING_BASIS


@pytest.mark.slow
def test_the_stamped_fairness_report_is_the_one_measured_at_the_band_threshold(
    tmp_path, fixture_path
):
    """The search scores candidates at a matched threshold of its own. That
    number must never become the artifact's fairness report, which is measured
    at config.RISK_BAND_LOW_MAX on the validation split."""
    model_out = tmp_path / "model.json"
    metadata_out = tmp_path / "model_meta.json"
    main(
        [
            "--data",
            str(fixture_path),
            "--model-out",
            str(model_out),
            "--metadata-out",
            str(metadata_out),
            "--provenance",
            "fixture",
            "--search",
        ]
    )
    metadata = ModelMetadata.model_validate_json(metadata_out.read_text())
    assert metadata.fairness.band_low_max == config.RISK_BAND_LOW_MAX
    assert metadata.fairness.adverse_definition == "band != low"


def test_without_search_no_selection_report_is_written(tmp_path, fixture_path):
    model_out = tmp_path / "model.json"
    metadata_out = tmp_path / "model_meta.json"
    code = main(
        [
            "--data",
            str(fixture_path),
            "--model-out",
            str(model_out),
            "--metadata-out",
            str(metadata_out),
            "--provenance",
            "fixture",
        ]
    )
    assert code == 0
    metadata = ModelMetadata.model_validate_json(metadata_out.read_text())
    assert metadata.selection is None


def test_a_non_baseline_winner_writes_nothing(tmp_path, fixture_path, monkeypatch, caplog):
    """Adopting a candidate is a reviewed code change. Training must refuse and
    say what to change, not silently ship a model whose feature set disagrees
    with the code."""
    model_out = tmp_path / "model.json"
    metadata_out = tmp_path / "model_meta.json"

    monkeypatch.setattr(
        "creditboost.train.search.rank",
        lambda frame, **kwargs: search.Ranking(
            target_approval_rate=0.75,
            candidates=[
                CandidateResult(
                    name=search.BASELINE.name,
                    n_features=20,
                    roc_auc=0.75,
                    min_adverse_impact_ratio=0.81,
                    adverse_impact_ratios={"DAYS_BIRTH": 0.81},
                ),
                CandidateResult(
                    name="no-occupation",
                    n_features=19,
                    roc_auc=0.75,
                    min_adverse_impact_ratio=0.95,
                    adverse_impact_ratios={"DAYS_BIRTH": 0.95},
                ),
            ],
        ),
    )

    code = main(
        [
            "--data",
            str(fixture_path),
            "--model-out",
            str(model_out),
            "--metadata-out",
            str(metadata_out),
            "--provenance",
            "fixture",
            "--search",
        ]
    )
    assert code == 1
    assert not model_out.exists()
    assert not metadata_out.exists()
    assert "no-occupation" in caplog.text


def test_ranking_never_sees_the_validation_split(fixture_path, tmp_path, monkeypatch):
    """rank() takes the training frame and nothing else. If a later refactor
    ever hands it the whole frame, this fails."""
    frame = load_training_frame(fixture_path)
    train_frame, valid_frame = split(frame)
    seen = {}

    def spy(passed_frame, **kwargs):
        seen["n"] = len(passed_frame)
        return search.Ranking(
            target_approval_rate=0.75,
            candidates=[
                CandidateResult(
                    name=search.BASELINE.name,
                    n_features=20,
                    roc_auc=0.75,
                    min_adverse_impact_ratio=0.81,
                    adverse_impact_ratios={"DAYS_BIRTH": 0.81},
                )
            ],
        )

    monkeypatch.setattr("creditboost.train.search.rank", spy)
    main(
        [
            "--data",
            str(fixture_path),
            "--model-out",
            str(tmp_path / "model.json"),
            "--metadata-out",
            str(tmp_path / "model_meta.json"),
            "--provenance",
            "fixture",
            "--search",
        ]
    )

    assert seen["n"] == len(train_frame)
    assert seen["n"] != len(frame)
