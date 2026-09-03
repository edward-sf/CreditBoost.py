import numpy as np
import pytest
import xgboost as xgb

from creditboost import config
from creditboost.artifact import (
    CorruptModelError,
    FeatureOrderMismatchError,
    XGBoostVersionMismatchError,
    load,
    save,
)
from creditboost.schema import ModelMetadata
from tests.conftest import a_passing_fairness_report


@pytest.fixture
def booster() -> xgb.Booster:
    rng = np.random.default_rng(0)
    matrix = xgb.DMatrix(
        rng.normal(size=(40, len(config.FEATURE_ORDER))), label=rng.integers(0, 2, 40)
    )
    return xgb.train({"objective": "binary:logistic"}, matrix, num_boost_round=2)


def metadata(feature_order: list[str] | None = None) -> ModelMetadata:
    return ModelMetadata(
        version=config.MODEL_VERSION,
        trained_at="2026-08-30T12:00:00Z",
        dataset_sha256="abc123",
        n_train_rows=40,
        feature_order=feature_order if feature_order is not None else list(config.FEATURE_ORDER),
        metrics={"roc_auc": 0.75},
        xgboost_version=xgb.__version__,
        provenance="fixture",
        fairness=a_passing_fairness_report(),
    )


def test_save_then_load_round_trips(tmp_path, booster):
    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    save(booster, metadata(), model_path, meta_path)
    loaded = load(model_path, meta_path)
    assert loaded.metadata == metadata()
    assert loaded.booster.num_boosted_rounds() == booster.num_boosted_rounds()


def test_load_rejects_a_mismatched_feature_order(tmp_path, booster):
    """The gate: a model trained on different features must never serve."""
    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    stale = [*config.FEATURE_ORDER[:-1], "some_removed_feature"]
    save(booster, metadata(stale), model_path, meta_path)
    with pytest.raises(FeatureOrderMismatchError):
        load(model_path, meta_path)


def test_load_rejects_a_merely_reordered_feature_list(tmp_path, booster):
    """Order matters, not just membership: XGBoost scores by position."""
    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    swapped = list(config.FEATURE_ORDER)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    save(booster, metadata(swapped), model_path, meta_path)
    with pytest.raises(FeatureOrderMismatchError):
        load(model_path, meta_path)


def test_load_rejects_a_booster_whose_own_feature_names_disagree(tmp_path):
    """The metadata sidecar can agree with FEATURE_ORDER while the booster's
    own feature_names -- baked directly into model.json -- disagree, e.g. a
    transposed-features artifact. Both must be checked, or such an artifact
    loads fine, passes HEALTHCHECK, and 500s on every request.

    Uses a DMatrix built with an explicit (transposed) feature_names list
    rather than the module's `booster` fixture, whose numpy-array DMatrix has
    feature_names is None -- exactly the "no names recorded" case that must
    be skipped, not raised on.
    """
    rng = np.random.default_rng(0)
    transposed_names = list(config.FEATURE_ORDER)
    transposed_names[0], transposed_names[1] = transposed_names[1], transposed_names[0]
    matrix = xgb.DMatrix(
        rng.normal(size=(40, len(config.FEATURE_ORDER))),
        label=rng.integers(0, 2, 40),
        feature_names=transposed_names,
    )
    named_booster = xgb.train({"objective": "binary:logistic"}, matrix, num_boost_round=2)

    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    save(named_booster, metadata(), model_path, meta_path)
    with pytest.raises(FeatureOrderMismatchError):
        load(model_path, meta_path)


def test_load_accepts_a_booster_with_no_recorded_feature_names(tmp_path, booster):
    """A booster trained from a raw numpy-array DMatrix has feature_names is
    None. That must be treated as 'no names recorded' and skipped, not
    raised on -- this is exactly the shape of the `booster` fixture used
    throughout this module."""
    assert booster.feature_names is None
    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    save(booster, metadata(), model_path, meta_path)
    load(model_path, meta_path)  # must not raise


def test_load_raises_when_the_model_file_is_absent(tmp_path, booster):
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(metadata().model_dump_json())
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing.json", meta_path)


def test_save_creates_the_parent_directory(tmp_path, booster):
    nested = tmp_path / "models" / "v1"
    save(booster, metadata(), nested / "model.json", nested / "meta.json")
    assert (nested / "model.json").exists()


def test_mismatch_error_names_the_offending_feature(tmp_path, booster):
    """A same-length mismatch must still be diagnosable: name the transposed feature."""
    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    transposed = list(config.FEATURE_ORDER)
    transposed[3], transposed[4] = transposed[4], transposed[3]
    save(booster, metadata(transposed), model_path, meta_path)
    with pytest.raises(FeatureOrderMismatchError) as excinfo:
        load(model_path, meta_path)
    assert config.FEATURE_ORDER[3] in str(excinfo.value)


def test_round_trip_preserves_predictions(tmp_path, booster):
    """A weaker check (num_boosted_rounds only) would pass even if weights were mangled."""
    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    save(booster, metadata(), model_path, meta_path)
    loaded = load(model_path, meta_path)

    rng = np.random.default_rng(1)
    matrix = xgb.DMatrix(rng.normal(size=(10, len(config.FEATURE_ORDER))))
    np.testing.assert_array_equal(loaded.booster.predict(matrix), booster.predict(matrix))


def test_load_raises_when_the_metadata_file_is_absent(tmp_path, booster):
    model_path = tmp_path / "model.json"
    booster.save_model(str(model_path))
    with pytest.raises(FileNotFoundError):
        load(model_path, tmp_path / "missing_meta.json")


def test_load_rejects_a_major_xgboost_version_mismatch(tmp_path, booster):
    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    stale_metadata = metadata()
    stale_metadata = stale_metadata.model_copy(update={"xgboost_version": "1.7.6"})
    save(booster, stale_metadata, model_path, meta_path)
    with pytest.raises(XGBoostVersionMismatchError):
        load(model_path, meta_path)


def test_load_succeeds_with_a_minor_xgboost_version_difference(tmp_path, booster):
    """Same major, different full version string: warn and continue, never raise."""
    running_major = xgb.__version__.split(".")[0]
    fake_version = f"{running_major}.0.0"
    if fake_version == xgb.__version__:
        fake_version = f"{running_major}.0.1"

    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    close_metadata = metadata().model_copy(update={"xgboost_version": fake_version})
    save(booster, close_metadata, model_path, meta_path)
    loaded = load(model_path, meta_path)
    assert loaded.metadata.xgboost_version == fake_version


def test_load_succeeds_with_an_unparseable_xgboost_version(tmp_path, booster):
    """Protects a later task that constructs test metadata with xgboost_version='test'."""
    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    unparseable_metadata = metadata().model_copy(update={"xgboost_version": "test"})
    save(booster, unparseable_metadata, model_path, meta_path)
    loaded = load(model_path, meta_path)
    assert loaded.metadata.xgboost_version == "test"


def test_load_rejects_a_corrupt_model_file(tmp_path, booster):
    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    save(booster, metadata(), model_path, meta_path)
    model_path.write_bytes(b"not a valid xgboost model file")
    with pytest.raises(CorruptModelError):
        load(model_path, meta_path)
