"""The committed pointer to a model release.

The model's bytes live in a GitHub Release; this file is what git tracks in
their place. It names the release and pins a sha256 for each asset, so a
build either gets exactly the reviewed bytes or fails.

Release coordination lives here rather than in config.py on purpose: config.py
is about features, thresholds and paths, and a fork should be able to repoint
asset_base_url at its own releases without touching code.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from . import config
from .artifact import ArtifactError
from .hashing import file_sha256

DEFAULT_ASSET_BASE_URL = "https://github.com/edward-sf/CreditBoost.py/releases/download"

MODEL_FILENAME = "model.json"
METADATA_FILENAME = "model_meta.json"


class LockfileError(ArtifactError):
    """The lockfile is missing, unparseable, or incomplete."""


class ModelLock(BaseModel):
    """Pins one model release. Deliberately has no model_version field: that
    would be a third place the version could disagree with itself. The CLI
    cross-checks the downloaded metadata's version against config.MODEL_VERSION
    instead."""

    model_config = ConfigDict(extra="forbid")

    release_tag: str
    asset_base_url: str
    model_sha256: str
    metadata_sha256: str
    released_at: str

    def asset_url(self, filename: str) -> str:
        return f"{self.asset_base_url.rstrip('/')}/{self.release_tag}/{filename}"


def read(path: Path | None = None) -> ModelLock:
    """Read the lockfile. The default is resolved at call time, not import
    time, because config.LOCKFILE_PATH derives from an environment variable
    that tests and the container both override."""
    resolved = config.LOCKFILE_PATH if path is None else path
    if not resolved.exists():
        raise LockfileError(
            f"model lockfile not found: {resolved}. A checkout must carry one; "
            "run scripts/release-model.sh to create a release and write it."
        )
    # Parsed separately from validation so the two failures read differently:
    # pydantic v2 folds a JSON syntax error into ValidationError, which would
    # otherwise report a malformed file as a schema problem.
    try:
        payload = json.loads(resolved.read_text())
    except ValueError as err:
        raise LockfileError(f"model lockfile at {resolved} could not be parsed: {err}") from err
    try:
        return ModelLock.model_validate(payload)
    except ValidationError as err:
        raise LockfileError(f"model lockfile at {resolved} is invalid: {err}") from err


def write(
    path: Path,
    release_tag: str,
    model_path: Path,
    metadata_path: Path,
    asset_base_url: str = DEFAULT_ASSET_BASE_URL,
) -> ModelLock:
    lock = ModelLock(
        release_tag=release_tag,
        asset_base_url=asset_base_url,
        model_sha256=file_sha256(model_path),
        metadata_sha256=file_sha256(metadata_path),
        released_at=datetime.now(UTC).isoformat(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lock.model_dump_json(indent=2) + "\n")
    return lock
