from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from creditboost import config, lockfile
from creditboost.artifact import FeatureOrderMismatchError
from creditboost.artifact_cli import (
    ChecksumMismatchError,
    ProvenanceError,
    VersionMismatchError,
    main,
    verify_artifact,
)
from creditboost.schema import ModelMetadata


def make_artifact(
    directory: Path,
    *,
    provenance: str = "production",
    version: str | None = None,
    metadata_feature_order: list[str] | None = None,
    booster_feature_names: list[str] | None = None,
) -> tuple[Path, Path]:
    """Write a model.json + model_meta.json pair into `directory`.

    Defaults produce a valid production artifact; each keyword corrupts exactly
    one property so a test can assert a single failure reason.
    """
    directory.mkdir(parents=True, exist_ok=True)
    expected = list(config.FEATURE_ORDER)
    names = booster_feature_names if booster_feature_names is not None else expected

    rows = np.zeros((4, len(names)), dtype=float)
    dmatrix = xgb.DMatrix(rows, label=[0, 1, 0, 1], feature_names=names)
    booster = xgb.train({"objective": "binary:logistic", "seed": 0}, dmatrix, num_boost_round=1)

    model_path = directory / lockfile.MODEL_FILENAME
    metadata_path = directory / lockfile.METADATA_FILENAME
    booster.save_model(str(model_path))

    metadata = ModelMetadata(
        version=config.MODEL_VERSION if version is None else version,
        trained_at="2026-09-01T00:00:00+00:00",
        dataset_sha256="0" * 64,
        n_train_rows=4,
        feature_order=metadata_feature_order if metadata_feature_order is not None else expected,
        metrics={"roc_auc": 0.75, "pr_auc": 0.24, "brier": 0.07},
        xgboost_version=xgb.__version__,
        provenance=provenance,  # type: ignore[arg-type]
    )
    metadata_path.write_text(metadata.model_dump_json(indent=2) + "\n")
    return model_path, metadata_path


def lock_for(directory: Path, tmp_path: Path) -> lockfile.ModelLock:
    """Lock the artifact as it currently stands on disk."""
    return lockfile.write(
        tmp_path / "model.lock.json",
        release_tag="model-v0.1.0",
        model_path=directory / lockfile.MODEL_FILENAME,
        metadata_path=directory / lockfile.METADATA_FILENAME,
    )


def test_a_well_formed_production_artifact_verifies(tmp_path: Path) -> None:
    directory = tmp_path / "models"
    make_artifact(directory)
    verify_artifact(directory, lock_for(directory, tmp_path))


def test_altered_model_bytes_are_rejected(tmp_path: Path) -> None:
    """This is the reason the lockfile carries hashes at all."""
    directory = tmp_path / "models"
    make_artifact(directory)
    lock = lock_for(directory, tmp_path)

    model_path = directory / lockfile.MODEL_FILENAME
    model_path.write_bytes(model_path.read_bytes() + b"\n")

    with pytest.raises(ChecksumMismatchError, match="model.json"):
        verify_artifact(directory, lock)


def test_altered_metadata_bytes_are_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "models"
    make_artifact(directory)
    lock = lock_for(directory, tmp_path)

    metadata_path = directory / lockfile.METADATA_FILENAME
    metadata_path.write_text(metadata_path.read_text() + "\n")

    with pytest.raises(ChecksumMismatchError, match="model_meta.json"):
        verify_artifact(directory, lock)


def test_a_fixture_artifact_is_rejected_by_default(tmp_path: Path) -> None:
    """The assertion that has no other home once models/ leaves git: a
    fixture-trained model must never reach GHCR."""
    directory = tmp_path / "models"
    make_artifact(directory, provenance="fixture")

    with pytest.raises(ProvenanceError, match="fixture"):
        verify_artifact(directory, lock_for(directory, tmp_path))


def test_a_fixture_artifact_is_allowed_with_the_explicit_flag(tmp_path: Path) -> None:
    directory = tmp_path / "models"
    make_artifact(directory, provenance="fixture")
    verify_artifact(directory, lock_for(directory, tmp_path), allow_fixture=True)


def test_a_version_disagreeing_with_config_is_rejected(tmp_path: Path) -> None:
    """Forces the lockfile bump and the MODEL_VERSION bump into one commit, so
    /health cannot report a version the artifact does not have."""
    directory = tmp_path / "models"
    make_artifact(directory, version="9.9.9")

    with pytest.raises(VersionMismatchError, match="9.9.9"):
        verify_artifact(directory, lock_for(directory, tmp_path))


def test_a_reordered_metadata_feature_order_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "models"
    reordered = list(config.FEATURE_ORDER)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    make_artifact(directory, metadata_feature_order=reordered)

    with pytest.raises(FeatureOrderMismatchError):
        verify_artifact(directory, lock_for(directory, tmp_path))


def test_a_clean_sidecar_cannot_hide_a_dirty_booster(tmp_path: Path) -> None:
    """The strongest case, and the reason this guard replaces rather than
    merely relocates the old committed-artifact test.

    The metadata sidecar reads as perfectly clean -- correct feature_order, no
    CODE_GENDER -- while the booster's OWN feature_names, baked into
    model.json, carry a protected attribute. Under ECOA / Regulation B that
    artifact must never be shippable.
    """
    directory = tmp_path / "models"
    dirty_names = list(config.FEATURE_ORDER)
    dirty_names[0] = "CODE_GENDER"
    make_artifact(directory, booster_feature_names=dirty_names)

    with pytest.raises(FeatureOrderMismatchError, match="CODE_GENDER"):
        verify_artifact(directory, lock_for(directory, tmp_path))


def test_a_missing_asset_is_reported_clearly(tmp_path: Path) -> None:
    directory = tmp_path / "models"
    make_artifact(directory)
    lock = lock_for(directory, tmp_path)
    (directory / lockfile.MODEL_FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match="model.json"):
        verify_artifact(directory, lock)


def test_verify_via_the_cli_returns_zero_on_success(tmp_path: Path) -> None:
    directory = tmp_path / "models"
    make_artifact(directory)
    lock_path = tmp_path / "model.lock.json"
    lock_for(directory, tmp_path)

    assert main(["verify", "--dir", str(directory), "--lockfile", str(lock_path)]) == 0


def test_verify_via_the_cli_returns_one_and_explains_on_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Docker build depends on a non-zero exit; a stack trace alone is not
    an acceptable failure mode for someone reading build logs."""
    directory = tmp_path / "models"
    make_artifact(directory, provenance="fixture")
    lock_path = tmp_path / "model.lock.json"
    lock_for(directory, tmp_path)

    assert main(["verify", "--dir", str(directory), "--lockfile", str(lock_path)]) == 1
    assert "fixture" in capsys.readouterr().err


# The dependency rule for this module -- that it never reaches data.py or
# train.py, which import scikit-learn and would crash the Docker builder --
# is enforced by the import-linter contract, extended in Step 4 below. It is
# deliberately NOT a test here: the rule is static, so a static contract that
# sees the whole import graph beats per-module sys.modules inspection.
