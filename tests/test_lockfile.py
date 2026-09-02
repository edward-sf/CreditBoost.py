import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from creditboost import lockfile
from creditboost.lockfile import LockfileError, ModelLock


@pytest.fixture
def assets(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "model.json"
    metadata = tmp_path / "model_meta.json"
    model.write_bytes(b"pretend booster")
    metadata.write_bytes(b'{"pretend": "metadata"}')
    return model, metadata


def test_write_then_read_round_trips(tmp_path: Path, assets: tuple[Path, Path]) -> None:
    model, metadata = assets
    path = tmp_path / "model.lock.json"

    written = lockfile.write(
        path, release_tag="model-v0.1.0", model_path=model, metadata_path=metadata
    )
    read_back = lockfile.read(path)

    assert read_back == written
    assert read_back.release_tag == "model-v0.1.0"


def test_write_records_the_actual_content_hashes(tmp_path: Path, assets: tuple[Path, Path]) -> None:
    """The whole point of the lockfile: the hashes must be of the real bytes,
    not copied from somewhere that could drift."""
    from creditboost.hashing import file_sha256

    model, metadata = assets
    path = tmp_path / "model.lock.json"

    written = lockfile.write(
        path, release_tag="model-v0.1.0", model_path=model, metadata_path=metadata
    )

    assert written.model_sha256 == file_sha256(model)
    assert written.metadata_sha256 == file_sha256(metadata)


def test_written_file_is_human_reviewable_json(tmp_path: Path, assets: tuple[Path, Path]) -> None:
    """A model bump should be a readable diff, which is the reason the bytes
    left git in the first place."""
    model, metadata = assets
    path = tmp_path / "model.lock.json"

    lockfile.write(path, release_tag="model-v0.1.0", model_path=model, metadata_path=metadata)

    text = path.read_text()
    assert text.endswith("\n")
    assert "\n  " in text, "expected indented, not minified, JSON"
    assert set(json.loads(text)) == {
        "release_tag",
        "asset_base_url",
        "model_sha256",
        "metadata_sha256",
        "released_at",
    }


def test_asset_url_joins_base_tag_and_filename(tmp_path: Path, assets: tuple[Path, Path]) -> None:
    model, metadata = assets
    path = tmp_path / "model.lock.json"
    written = lockfile.write(
        path,
        release_tag="model-v0.1.0",
        model_path=model,
        metadata_path=metadata,
        asset_base_url="https://example.test/releases/download",
    )

    assert written.asset_url("model.json") == (
        "https://example.test/releases/download/model-v0.1.0/model.json"
    )


def test_asset_url_tolerates_a_trailing_slash_on_the_base(
    tmp_path: Path, assets: tuple[Path, Path]
) -> None:
    """A hand-edited lockfile is the expected way to repoint a fork, so a
    stray trailing slash must not produce a double slash in the URL."""
    model, metadata = assets
    path = tmp_path / "model.lock.json"
    written = lockfile.write(
        path,
        release_tag="model-v0.1.0",
        model_path=model,
        metadata_path=metadata,
        asset_base_url="https://example.test/releases/download/",
    )

    assert written.asset_url("model.json") == (
        "https://example.test/releases/download/model-v0.1.0/model.json"
    )


def test_read_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "model.lock.json"
    path.write_text("{not json")

    with pytest.raises(LockfileError, match="could not be parsed"):
        lockfile.read(path)


def test_read_rejects_a_missing_required_field(tmp_path: Path) -> None:
    path = tmp_path / "model.lock.json"
    path.write_text(json.dumps({"release_tag": "model-v0.1.0"}))

    with pytest.raises(LockfileError, match="model_sha256"):
        lockfile.read(path)


def test_read_reports_a_missing_file_clearly(tmp_path: Path) -> None:
    with pytest.raises(LockfileError, match="not found"):
        lockfile.read(tmp_path / "absent.lock.json")


def test_lockfile_error_is_an_artifact_error() -> None:
    """All artifact-related failures share one root so callers can catch broadly."""
    from creditboost.artifact import ArtifactError

    assert issubclass(LockfileError, ArtifactError)


def test_model_lock_rejects_unknown_fields() -> None:
    """A typo'd key must fail loudly rather than being silently ignored."""
    with pytest.raises(ValidationError):
        ModelLock(
            release_tag="model-v0.1.0",
            asset_base_url="https://example.test",
            model_sha256="a" * 64,
            metadata_sha256="b" * 64,
            released_at="2026-09-01T00:00:00+00:00",
            typo_field="oops",
        )
