import http.server
import threading
from collections.abc import Iterator
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
from tests.conftest import a_passing_fairness_report


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
        fairness=a_passing_fairness_report(),
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


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serves files from `directory`, optionally failing the first N requests
    per path so the retry path can be exercised."""

    directory: Path
    failures_remaining: dict[str, int] = {}

    def do_GET(self) -> None:  # noqa: N802  (http.server's required name)
        name = self.path.rsplit("/", 1)[-1]
        if self.failures_remaining.get(name, 0) > 0:
            self.failures_remaining[name] -= 1
            self.send_error(503, "temporarily unavailable")
            return
        target = self.directory / name
        if not target.exists():
            self.send_error(404, "not found")
            return
        payload = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:
        return  # keep pytest output clean


@pytest.fixture
def asset_server(tmp_path: Path) -> Iterator[tuple[str, Path, dict[str, int]]]:
    """Yields (base_url, served_directory, failures_remaining)."""
    served = tmp_path / "served"
    served.mkdir()
    failures: dict[str, int] = {}

    handler = type("Handler", (_Handler,), {"directory": served, "failures_remaining": failures})
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/releases/download", served, failures
    finally:
        server.shutdown()
        server.server_close()


def test_fetch_downloads_both_assets(tmp_path: Path, asset_server) -> None:
    base_url, served, _ = asset_server
    make_artifact(served)
    lock = lockfile.write(
        tmp_path / "model.lock.json",
        release_tag="model-v0.1.0",
        model_path=served / lockfile.MODEL_FILENAME,
        metadata_path=served / lockfile.METADATA_FILENAME,
        asset_base_url=base_url,
    )

    from creditboost.artifact_cli import fetch_artifact

    destination = tmp_path / "downloaded"
    fetch_artifact(destination, lock)

    assert (destination / lockfile.MODEL_FILENAME).exists()
    assert (destination / lockfile.METADATA_FILENAME).exists()


def test_fetched_assets_then_verify(tmp_path: Path, asset_server) -> None:
    """fetch and verify are two halves of one job; this is the pair the
    Dockerfile actually runs."""
    base_url, served, _ = asset_server
    make_artifact(served)
    lock = lockfile.write(
        tmp_path / "model.lock.json",
        release_tag="model-v0.1.0",
        model_path=served / lockfile.MODEL_FILENAME,
        metadata_path=served / lockfile.METADATA_FILENAME,
        asset_base_url=base_url,
    )

    from creditboost.artifact_cli import fetch_artifact

    destination = tmp_path / "downloaded"
    fetch_artifact(destination, lock)
    verify_artifact(destination, lock)


def test_a_missing_release_asset_fails_without_retrying(
    tmp_path: Path, asset_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 404 is a real defect -- a deleted release or a wrong tag. Retrying it
    only delays a truthful failure."""
    import creditboost.artifact_cli as cli

    base_url, served, _ = asset_server
    make_artifact(served)
    lock = lockfile.write(
        tmp_path / "model.lock.json",
        release_tag="model-v0.1.0",
        model_path=served / lockfile.MODEL_FILENAME,
        metadata_path=served / lockfile.METADATA_FILENAME,
        asset_base_url=base_url,
    )
    (served / lockfile.MODEL_FILENAME).unlink()

    attempts = 0
    original = cli._download_once

    def counting(url: str, dest: Path) -> None:
        nonlocal attempts
        attempts += 1
        original(url, dest)

    monkeypatch.setattr(cli, "_download_once", counting)

    with pytest.raises(cli.AssetNotFoundError, match="model-v0.1.0"):
        cli.fetch_artifact(tmp_path / "downloaded", lock)
    assert attempts == 1, "a 404 must not be retried"


def test_a_transient_failure_is_retried_and_succeeds(
    tmp_path: Path, asset_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    import creditboost.artifact_cli as cli

    monkeypatch.setattr(cli, "RETRY_BACKOFF_SECONDS", 0.0)

    base_url, served, failures = asset_server
    make_artifact(served)
    lock = lockfile.write(
        tmp_path / "model.lock.json",
        release_tag="model-v0.1.0",
        model_path=served / lockfile.MODEL_FILENAME,
        metadata_path=served / lockfile.METADATA_FILENAME,
        asset_base_url=base_url,
    )
    failures[lockfile.MODEL_FILENAME] = 2  # fail twice, succeed on the third

    destination = tmp_path / "downloaded"
    cli.fetch_artifact(destination, lock)

    assert (destination / lockfile.MODEL_FILENAME).exists()


def test_a_persistent_transient_failure_eventually_gives_up(
    tmp_path: Path, asset_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    import creditboost.artifact_cli as cli

    monkeypatch.setattr(cli, "RETRY_BACKOFF_SECONDS", 0.0)

    base_url, served, failures = asset_server
    make_artifact(served)
    lock = lockfile.write(
        tmp_path / "model.lock.json",
        release_tag="model-v0.1.0",
        model_path=served / lockfile.MODEL_FILENAME,
        metadata_path=served / lockfile.METADATA_FILENAME,
        asset_base_url=base_url,
    )
    failures[lockfile.MODEL_FILENAME] = 99

    with pytest.raises(cli.AssetDownloadError):
        cli.fetch_artifact(tmp_path / "downloaded", lock)


def test_fetch_via_the_cli_returns_zero(tmp_path: Path, asset_server) -> None:
    base_url, served, _ = asset_server
    make_artifact(served)
    lock_path = tmp_path / "model.lock.json"
    lockfile.write(
        lock_path,
        release_tag="model-v0.1.0",
        model_path=served / lockfile.MODEL_FILENAME,
        metadata_path=served / lockfile.METADATA_FILENAME,
        asset_base_url=base_url,
    )

    destination = tmp_path / "downloaded"
    assert main(["fetch", "--dir", str(destination), "--lockfile", str(lock_path)]) == 0
    assert (destination / lockfile.MODEL_FILENAME).exists()


def test_lock_writes_a_lockfile_matching_the_local_artifact(tmp_path: Path) -> None:
    from creditboost.hashing import file_sha256

    directory = tmp_path / "models"
    make_artifact(directory)
    lock_path = tmp_path / "model.lock.json"

    exit_code = main(
        ["lock", "--tag", "model-v0.1.0", "--dir", str(directory), "--lockfile", str(lock_path)]
    )

    assert exit_code == 0
    written = lockfile.read(lock_path)
    assert written.release_tag == "model-v0.1.0"
    assert written.model_sha256 == file_sha256(directory / lockfile.MODEL_FILENAME)


def test_a_freshly_locked_artifact_verifies_immediately(tmp_path: Path) -> None:
    """lock then verify is the round trip the release script depends on; if it
    does not hold, every release is born broken."""
    directory = tmp_path / "models"
    make_artifact(directory)
    lock_path = tmp_path / "model.lock.json"

    assert (
        main(
            [
                "lock",
                "--tag",
                "model-v0.1.0",
                "--dir",
                str(directory),
                "--lockfile",
                str(lock_path),
            ]
        )
        == 0
    )
    assert main(["verify", "--dir", str(directory), "--lockfile", str(lock_path)]) == 0


def test_lock_honours_an_explicit_asset_base_url(tmp_path: Path) -> None:
    directory = tmp_path / "models"
    make_artifact(directory)
    lock_path = tmp_path / "model.lock.json"

    main(
        [
            "lock",
            "--tag",
            "model-v0.1.0",
            "--dir",
            str(directory),
            "--lockfile",
            str(lock_path),
            "--asset-base-url",
            "https://example.test/dl",
        ]
    )

    assert lockfile.read(lock_path).asset_base_url == "https://example.test/dl"
