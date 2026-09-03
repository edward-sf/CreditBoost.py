"""The creditboost-artifact CLI: fetch, verify, and lock model release assets.

This runs inside the Docker builder stage, which installs the base package
only. It must therefore never import data.py or train.py, both of which pull
in scikit-learn. The import-linter contract in pyproject.toml enforces that.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config, lockfile
from .artifact import ArtifactError, FeatureOrderMismatchError, load
from .hashing import file_sha256
from .lockfile import METADATA_FILENAME, MODEL_FILENAME, ModelLock
from .schema import ModelMetadata

PROHIBITED_FEATURES = ("CODE_GENDER", "DAYS_BIRTH")

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.0
DOWNLOAD_TIMEOUT_SECONDS = 30


class ChecksumMismatchError(ArtifactError):
    """A downloaded asset's content hash disagrees with the lockfile."""


class ProvenanceError(ArtifactError):
    """The artifact is not the provenance this build requires."""


class VersionMismatchError(ArtifactError):
    """The artifact's version disagrees with config.MODEL_VERSION."""


class AssetNotFoundError(ArtifactError):
    """The release asset does not exist. Not retried: this is a real defect."""


class AssetDownloadError(ArtifactError):
    """The release asset could not be downloaded after retrying."""


class SelectionError(ArtifactError):
    """A production artifact carries no record of an alternative search."""


def _download_once(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        dest.write_bytes(response.read())


def _download(url: str, dest: Path, release_tag: str) -> None:
    """Download with backoff, but never retry a 404.

    A transient blip should not fail an unrelated PR build; a missing asset is
    a genuine defect and retrying it only delays a truthful failure.
    """
    last_error: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            _download_once(url, dest)
            return
        except urllib.error.HTTPError as err:
            if err.code == 404:
                raise AssetNotFoundError(
                    f"release asset not found: {url} (release {release_tag}). "
                    "Either the release was never created or it was deleted. "
                    "Run scripts/release-model.sh to create it, and note that "
                    "deleting a model release breaks every build pinned to it."
                ) from err
            last_error = err
        except (urllib.error.URLError, OSError) as err:
            last_error = err
        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise AssetDownloadError(
        f"could not download {url} after {RETRY_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def fetch_artifact(directory: Path, lock: ModelLock) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename in (MODEL_FILENAME, METADATA_FILENAME):
        _download(lock.asset_url(filename), directory / filename, lock.release_tag)


def _check_digest(path: Path, expected: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"artifact asset not found: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ChecksumMismatchError(
            f"{path.name} does not match the lockfile.\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            "Either the release asset changed after it was locked, or the "
            "download was corrupted. Do not proceed with this artifact."
        )


def verify_artifact(directory: Path, lock: ModelLock, allow_fixture: bool = False) -> None:
    """Raise on the first thing wrong with the artifact in `directory`.

    Cheap, explicit checks run first so failures are legible; artifact.load()
    runs last as the end-to-end proof, which also covers xgboost major-version
    compatibility and parseability for free.
    """
    model_path = directory / MODEL_FILENAME
    metadata_path = directory / METADATA_FILENAME

    _check_digest(model_path, lock.model_sha256)
    _check_digest(metadata_path, lock.metadata_sha256)

    metadata = ModelMetadata.model_validate_json(metadata_path.read_text())

    if metadata.version != config.MODEL_VERSION:
        raise VersionMismatchError(
            f"artifact version {metadata.version!r} does not match this build's "
            f"config.MODEL_VERSION {config.MODEL_VERSION!r}. The lockfile and "
            "MODEL_VERSION must be bumped in the same commit, so that /health "
            "reports a version the artifact actually has."
        )

    expected = list(config.FEATURE_ORDER)
    if metadata.feature_order != expected:
        raise FeatureOrderMismatchError(
            "artifact metadata feature_order does not match this build's "
            f"FEATURE_ORDER; artifact has {len(metadata.feature_order)} features, "
            f"code expects {len(expected)}. Retrain the model against this code."
        )

    for prohibited in PROHIBITED_FEATURES:
        if prohibited in metadata.feature_order:
            raise FeatureOrderMismatchError(
                f"artifact metadata lists {prohibited!r} as a model feature. Sex "
                "and age are prohibited bases for a US credit decision under "
                "ECOA / Regulation B; this artifact must not be shipped."
            )

    if not allow_fixture and metadata.provenance != "production":
        raise ProvenanceError(
            f"artifact provenance is {metadata.provenance!r}, but this build "
            "requires 'production'. A fixture-trained model must never be "
            "published. Pass --allow-fixture only for local experiments."
        )

    if metadata.provenance == "production":
        if metadata.selection is None:
            raise SelectionError(
                "artifact provenance is 'production' but it carries no selection "
                "report: this model was never searched for a less discriminatory "
                "alternative. Disparate impact is a burden-shifting doctrine, and "
                "business necessity rebuts a prima facie case only if no such "
                "alternative exists -- an unsearched model cannot support that "
                "claim. Retrain with `creditboost-train --search`."
            )
        if all(candidate.failed_reason is not None for candidate in metadata.selection.candidates):
            raise SelectionError(
                "artifact provenance is 'production' but no candidate in its "
                "selection report could be scored: the search established "
                "nothing. A frontier of failures must never read as a search "
                "that found no better model. Retrain on data large enough to "
                "measure disparate impact."
            )

    # The end-to-end proof. load() re-checks the sidecar, checks the booster's
    # OWN feature_names baked into model.json, and checks xgboost versions.
    loaded = load(model_path, metadata_path)

    booster_names = list(loaded.booster.feature_names or [])
    for prohibited in PROHIBITED_FEATURES:
        if prohibited in booster_names:
            raise FeatureOrderMismatchError(
                f"artifact booster feature_names contains {prohibited!r}. Sex and "
                "age are prohibited bases for a US credit decision under ECOA / "
                "Regulation B; this artifact must not be shipped."
            )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dir", type=Path, default=None, dest="directory")
    parser.add_argument("--lockfile", type=Path, default=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="creditboost-artifact")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="download the pinned release assets")
    _add_common(fetch_parser)

    verify_parser = subparsers.add_parser("verify", help="check an artifact against the lockfile")
    _add_common(verify_parser)
    verify_parser.add_argument(
        "--allow-fixture",
        action="store_true",
        help="permit a fixture-provenance artifact; never use for a published image",
    )

    lock_parser = subparsers.add_parser("lock", help="write a lockfile for a local artifact")
    _add_common(lock_parser)
    lock_parser.add_argument("--tag", required=True, help="release tag, e.g. model-v0.1.0")
    lock_parser.add_argument("--asset-base-url", default=lockfile.DEFAULT_ASSET_BASE_URL)

    args = parser.parse_args(argv)
    directory = args.directory if args.directory is not None else config.MODEL_DIR
    lock_path = args.lockfile if args.lockfile is not None else config.LOCKFILE_PATH

    try:
        if args.command == "fetch":
            lock = lockfile.read(lock_path)
            fetch_artifact(directory, lock)
            print(f"fetched {lock.release_tag} into {directory}")
        elif args.command == "verify":
            lock = lockfile.read(lock_path)
            verify_artifact(directory, lock, allow_fixture=args.allow_fixture)
            print(f"artifact in {directory} verified against {lock.release_tag}")
        elif args.command == "lock":
            written = lockfile.write(
                lock_path,
                release_tag=args.tag,
                model_path=directory / MODEL_FILENAME,
                metadata_path=directory / METADATA_FILENAME,
                asset_base_url=args.asset_base_url,
            )
            print(f"wrote {lock_path} for {written.release_tag}")
    except (ArtifactError, FileNotFoundError, ValueError) as err:
        print(f"{type(err).__name__}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
