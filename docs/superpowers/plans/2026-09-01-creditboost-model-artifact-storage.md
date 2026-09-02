# Model Artifact Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the trained model's bytes out of git into a GitHub Release, leaving a checksum-pinned lockfile in their place, with a build-time guard that makes shipping a bad artifact structurally impossible.

**Architecture:** A committed `models/model.lock.json` pins a release tag and a sha256 per asset. A new `creditboost-artifact` CLI fetches assets from the release and verifies them; both run inside the Docker builder stage, so `docker build .` still takes zero arguments and the runtime image and boot-time skew gate are untouched. A `scripts/release-model.sh` helper creates the release after a manual local training run.

**Tech Stack:** Python 3.12, pydantic v2, xgboost, argparse, `urllib.request` (no `curl`, no `requests`), `http.server` for hermetic tests, pytest, Docker multi-stage, GitHub Actions, `gh` CLI.

**Spec:** `docs/superpowers/specs/2026-09-01-creditboost-model-artifact-storage-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Python 3.12 only.** `requires-python = ">=3.12"`.
- **On macOS, `brew install libomp` is required** before `pip install`, or `import xgboost` fails.
- **The new CLI must never import `data.py` or `train.py`.** Both import scikit-learn at
  module scope, and scikit-learn is **not installed in the Docker builder stage**
  (`pip install .`, never `.[train]` — sklearn is in the `train` extra). Importing either
  would crash the build, not merely offend the layering. The CLI may import only
  `artifact`, `config`, `schema`, `banding`, `lockfile`, and `hashing`.
- **Test that rule by naming those two modules — never by asserting
  `'sklearn' not in sys.modules`.** The proxy check fails on *correct* code: xgboost
  imports sklearn itself whenever sklearn is installed, which it is in a `[train,dev]`
  dev venv. `tests/test_api.py:186-200` documents this trap and shows the right idiom.
- **`pytest` must stay fully hermetic and offline.** No test may touch the network. Loopback `http.server` is fine.
- **`ruff` line-length is 100**, lint rules `["E", "F", "I", "UP", "B"]`. Run `ruff check . && ruff format --check .` before every commit; CI enforces formatting.
- **`mypy src/` must pass.**
- **`CODE_GENDER` and raw `DAYS_BIRTH` must never appear** in any feature list, request schema, or transform. ECOA / Regulation B.
- **No applicant financial field may ever be logged.**
- **Never lower `config.MIN_VALIDATION_AUC`** (0.70) to force a model through.
- Repository is **public**: `edward-sf/CreditBoost.py`. Release assets download with no credentials.
- Current committed artifact checksums, needed verbatim in Task 6:
  - `model.json` → `97869896fdb65eacd86b901438adbc2d11d52cc96baad715f9081b703450e4ac`
  - `model_meta.json` → `0df0975d33efe7d66386c5a0ed349600e79ebec85eb4b2f9218c6e806c6d4b7d`

## File Structure

| File | Responsibility |
|---|---|
| `src/creditboost/hashing.py` | **Create.** `file_sha256` only. Zero dependencies beyond `hashlib`, so both `data.py` (sklearn side) and the CLI (runtime side) can import it. |
| `src/creditboost/lockfile.py` | **Create.** The `ModelLock` pydantic model plus `read`/`write`. No network, no CLI. |
| `src/creditboost/artifact_cli.py` | **Create.** The `creditboost-artifact` console script: `fetch`, `verify`, `lock`. |
| `src/creditboost/data.py` | **Modify.** Import `file_sha256` from `hashing` instead of defining it. |
| `src/creditboost/config.py` | **Modify.** Add `LOCKFILE_PATH`. |
| `models/model.lock.json` | **Create.** The committed pointer. |
| `scripts/release-model.sh` | **Create.** `gh release create` + upload + `creditboost-artifact lock`. |
| `Dockerfile` | **Modify.** Builder fetches and verifies; runtime copies from builder. |
| `.github/workflows/ci.yml` | **Modify.** Read the model version from `config`, not the deleted metadata file. |
| `.gitignore`, `.dockerignore` | **Modify.** Ignore `models/*.json`, except the lockfile. |
| `pyproject.toml` | **Modify.** Register the console script. |
| `tests/test_hashing.py`, `tests/test_lockfile.py`, `tests/test_artifact_cli.py` | **Create.** |
| `tests/test_train.py` | **Modify.** Remove the two committed-artifact tests. |
| `CLAUDE.md`, `README.md` | **Modify.** Invariant ledger and workflow docs. |

---

### Task 1: Extract `file_sha256` into a dependency-free module

`file_sha256` currently lives in `data.py`, which imports scikit-learn. The new CLI runs in the Docker builder stage, where scikit-learn is **not installed**, so it cannot import `data.py` at all. This task moves the function somewhere both sides can reach. It is a pure refactor: no behaviour changes.

**Files:**
- Create: `src/creditboost/hashing.py`
- Modify: `src/creditboost/data.py:5` (drop the now-unused `hashlib` import), `src/creditboost/data.py:18-24` (remove the function definition)
- Test: `tests/test_hashing.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `creditboost.hashing.file_sha256(path: Path) -> str`. Tasks 2, 3 and 5 depend on this exact name and signature. `creditboost.data.file_sha256` remains importable as a re-export so `train.py` needs no change.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hashing.py`:

```python
from pathlib import Path

from creditboost.hashing import file_sha256

# sha256 of the literal bytes b"abc", a published test vector.
ABC_DIGEST = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_file_sha256_matches_the_known_vector_for_abc(tmp_path: Path) -> None:
    target = tmp_path / "abc.txt"
    target.write_bytes(b"abc")
    assert file_sha256(target) == ABC_DIGEST


def test_file_sha256_reads_in_chunks_so_a_file_larger_than_the_buffer_still_hashes(
    tmp_path: Path,
) -> None:
    """The implementation reads 1MB at a time; a file spanning several chunks
    must hash identically to hashing the whole payload at once."""
    import hashlib

    payload = b"x" * (1024 * 1024 * 3 + 17)
    target = tmp_path / "big.bin"
    target.write_bytes(payload)
    assert file_sha256(target) == hashlib.sha256(payload).hexdigest()


def test_hashing_module_does_not_reach_the_training_modules() -> None:
    """hashing.py exists precisely so the Docker builder stage -- which installs
    the base package only -- can hash files.

    This names creditboost.data and creditboost.train rather than checking for
    sklearn, following the idiom tests/test_api.py:186-200 established and
    explained: sklearn's presence depends on installed extras AND on xgboost's
    own unrelated optional sklearn integration, so a proxy check on sklearn is
    both false-positive and false-negative prone. Naming the forbidden modules
    sidesteps both.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import creditboost.hashing\n"
        "forbidden = [m for m in ('creditboost.data', 'creditboost.train') if m in sys.modules]\n"
        "assert not forbidden, f'hashing pulled in training module(s): {forbidden}'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_hashing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'creditboost.hashing'`

- [ ] **Step 3: Create the module**

Create `src/creditboost/hashing.py`:

```python
"""Content hashing. Deliberately dependency-free.

This lives apart from data.py because data.py imports scikit-learn, which is
not installed in the Docker builder stage. The artifact CLI runs there and
needs to hash files, so the function cannot live behind an sklearn import.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK_BYTES = 1024 * 1024


def file_sha256(path: Path) -> str:
    """Content hash, used both to trace a model to its training data and to
    verify a downloaded release asset against the lockfile."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

- [ ] **Step 4: Update `data.py` to re-export rather than redefine**

In `src/creditboost/data.py`, delete the `import hashlib` line and the whole `file_sha256` definition (lines 18-24), then add the import alongside the existing relative imports:

```python
from . import config
from .hashing import file_sha256

__all__ = ["MissingColumnsError", "file_sha256", "load_training_frame", "split"]
```

The `__all__` entry is what keeps `from .data import file_sha256` working in `train.py:22` without touching `train.py`. Do not remove that re-export.

- [ ] **Step 5: Run the full suite to verify nothing regressed**

Run: `pytest -v`
Expected: PASS, including the pre-existing `tests/test_data.py` and `tests/test_train.py`. This is a refactor; a failure here means the re-export is wrong.

- [ ] **Step 6: Lint and type-check**

Run: `ruff check . && ruff format --check . && mypy src/`
Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add src/creditboost/hashing.py src/creditboost/data.py tests/test_hashing.py
git commit -m "refactor: extract file_sha256 into a dependency-free hashing module

The artifact CLI runs in the Docker builder stage, which installs the base
package only and therefore has no scikit-learn. data.py imports sklearn, so
the hash helper had to move somewhere both sides can import."
```

---

### Task 2: The lockfile model

**Files:**
- Create: `src/creditboost/lockfile.py`
- Modify: `src/creditboost/config.py` (add `LOCKFILE_PATH` beside the existing path constants)
- Test: `tests/test_lockfile.py`

**Interfaces:**
- Consumes: `creditboost.hashing.file_sha256` from Task 1.
- Produces:
  - `creditboost.lockfile.ModelLock` — pydantic model with fields `release_tag: str`, `asset_base_url: str`, `model_sha256: str`, `metadata_sha256: str`, `released_at: str`
  - `creditboost.lockfile.LockfileError` (subclasses `creditboost.artifact.ArtifactError`)
  - `read(path: Path) -> ModelLock`
  - `write(path: Path, release_tag: str, model_path: Path, metadata_path: Path, asset_base_url: str = DEFAULT_ASSET_BASE_URL) -> ModelLock`
  - `ModelLock.asset_url(filename: str) -> str`
  - `DEFAULT_ASSET_BASE_URL: str`
  - `creditboost.config.LOCKFILE_PATH: Path`
- Tasks 3, 4 and 5 depend on every name above.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lockfile.py`:

```python
import json
from pathlib import Path

import pytest

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

    written = lockfile.write(path, release_tag="model-v0.1.0", model_path=model, metadata_path=metadata)
    read_back = lockfile.read(path)

    assert read_back == written
    assert read_back.release_tag == "model-v0.1.0"


def test_write_records_the_actual_content_hashes(tmp_path: Path, assets: tuple[Path, Path]) -> None:
    """The whole point of the lockfile: the hashes must be of the real bytes,
    not copied from somewhere that could drift."""
    from creditboost.hashing import file_sha256

    model, metadata = assets
    path = tmp_path / "model.lock.json"

    written = lockfile.write(path, release_tag="model-v0.1.0", model_path=model, metadata_path=metadata)

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


def test_asset_url_tolerates_a_trailing_slash_on_the_base(tmp_path: Path, assets: tuple[Path, Path]) -> None:
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
    with pytest.raises(Exception):
        ModelLock(
            release_tag="model-v0.1.0",
            asset_base_url="https://example.test",
            model_sha256="a" * 64,
            metadata_sha256="b" * 64,
            released_at="2026-09-01T00:00:00+00:00",
            typo_field="oops",
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_lockfile.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'creditboost.lockfile'`

- [ ] **Step 3: Add the path constant to `config.py`**

In `src/creditboost/config.py`, immediately after the existing `METADATA_PATH` line, add:

```python
LOCKFILE_PATH = MODEL_DIR / "model.lock.json"
```

This is a path, which is what `config.py` is for. Release coordination — the tag and the base URL — deliberately stays in the lockfile, not here.

- [ ] **Step 4: Write the module**

Create `src/creditboost/lockfile.py`:

```python
"""The committed pointer to a model release.

The model's bytes live in a GitHub Release; this file is what git tracks in
their place. It names the release and pins a sha256 for each asset, so a
build either gets exactly the reviewed bytes or fails.

Release coordination lives here rather than in config.py on purpose: config.py
is about features, thresholds and paths, and a fork should be able to repoint
asset_base_url at its own releases without touching code.
"""

from __future__ import annotations

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
    try:
        return ModelLock.model_validate_json(resolved.read_text())
    except ValidationError as err:
        raise LockfileError(f"model lockfile at {resolved} is invalid: {err}") from err
    except ValueError as err:
        raise LockfileError(f"model lockfile at {resolved} could not be parsed: {err}") from err


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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_lockfile.py -v`
Expected: PASS, all 10 tests.

- [ ] **Step 6: Lint and type-check**

Run: `ruff check . && ruff format --check . && mypy src/`
Expected: all clean. `mypy` in particular must not report an implicit `Optional`.

- [ ] **Step 7: Commit**

```bash
git add src/creditboost/lockfile.py src/creditboost/config.py tests/test_lockfile.py
git commit -m "feat: add the model lockfile model and reader

Pins a release tag plus a sha256 per asset. No model_version field: the CLI
cross-checks the downloaded metadata against config.MODEL_VERSION rather than
adding a third place the version could disagree with itself."
```

---

### Task 3: `creditboost-artifact verify`

The guard. This is the task that makes shipping a bad artifact structurally impossible, and it is where the two deleted `test_train.py` tests are reborn in stronger form.

**Files:**
- Create: `src/creditboost/artifact_cli.py`
- Modify: `pyproject.toml` (register the console script)
- Test: `tests/test_artifact_cli.py`

**Interfaces:**
- Consumes: `lockfile.read`, `lockfile.ModelLock`, `lockfile.MODEL_FILENAME`, `lockfile.METADATA_FILENAME`, `hashing.file_sha256`, `artifact.load`, `artifact.ArtifactError`, `schema.ModelMetadata`, `config.FEATURE_ORDER`, `config.MODEL_VERSION`.
- Produces:
  - `creditboost.artifact_cli.main(argv: list[str] | None = None) -> int`
  - `verify_artifact(directory: Path, lock: ModelLock, allow_fixture: bool = False) -> None` — raises on any failure, returns `None` on success
  - `ChecksumMismatchError`, `ProvenanceError`, `VersionMismatchError` (all subclass `artifact.ArtifactError`)
  - Console script `creditboost-artifact`
- Task 4 adds `fetch` to this same `main`; Task 5 adds `lock`. Task 7 invokes `creditboost-artifact verify --dir <path>`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_artifact_cli.py`. The `make_artifact` helper builds a real booster so the tests exercise the same code path a real artifact does — a fake file would not have `feature_names` baked in, which is exactly the thing one test needs to corrupt.

```python
import json
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


def test_the_cli_never_reaches_the_training_modules() -> None:
    """artifact_cli runs in the Docker builder stage, which installs the base
    package only -- scikit-learn is in the [train] extra and is absent there.
    data.py imports sklearn at module scope, so an accidental import of it
    would not merely be untidy: it would crash the build.

    Do NOT rewrite this as `assert 'sklearn' not in sys.modules`. That check
    FAILS even when the code is correct, because xgboost imports sklearn
    itself whenever sklearn happens to be installed -- which it is in a
    [train,dev] dev venv. tests/test_api.py:186-200 documents this trap; this
    test follows the same idiom for the same reason.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import creditboost.artifact_cli\n"
        "forbidden = [m for m in ('creditboost.data', 'creditboost.train') if m in sys.modules]\n"
        "assert not forbidden, f'artifact_cli pulled in training module(s): {forbidden}'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_artifact_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'creditboost.artifact_cli'`

- [ ] **Step 3: Write the module**

Create `src/creditboost/artifact_cli.py`:

```python
"""The creditboost-artifact CLI: fetch, verify, and lock model release assets.

This runs inside the Docker builder stage, which installs the base package
only. It must therefore never import data.py or train.py, both of which pull
in scikit-learn. tests/test_artifact_cli.py enforces that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, lockfile
from .artifact import ArtifactError, FeatureOrderMismatchError, load
from .hashing import file_sha256
from .lockfile import METADATA_FILENAME, MODEL_FILENAME, ModelLock
from .schema import ModelMetadata

PROHIBITED_FEATURES = ("CODE_GENDER", "DAYS_BIRTH")


class ChecksumMismatchError(ArtifactError):
    """A downloaded asset's content hash disagrees with the lockfile."""


class ProvenanceError(ArtifactError):
    """The artifact is not the provenance this build requires."""


class VersionMismatchError(ArtifactError):
    """The artifact's version disagrees with config.MODEL_VERSION."""


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

    verify_parser = subparsers.add_parser("verify", help="check an artifact against the lockfile")
    _add_common(verify_parser)
    verify_parser.add_argument(
        "--allow-fixture",
        action="store_true",
        help="permit a fixture-provenance artifact; never use for a published image",
    )

    args = parser.parse_args(argv)
    directory = args.directory if args.directory is not None else config.MODEL_DIR
    lock_path = args.lockfile if args.lockfile is not None else config.LOCKFILE_PATH

    try:
        if args.command == "verify":
            lock = lockfile.read(lock_path)
            verify_artifact(directory, lock, allow_fixture=args.allow_fixture)
            print(f"artifact in {directory} verified against {lock.release_tag}")
    except (ArtifactError, FileNotFoundError, ValueError) as err:
        print(f"{type(err).__name__}: {err}", file=sys.stderr)
        return 1
    return 0
```

Note the ordering deliberately puts the booster `feature_names` ECOA check *after* `load()`: `load()` already rejects any booster whose names differ from `FEATURE_ORDER`, so a `CODE_GENDER` booster raises there first with a `FeatureOrderMismatchError`. The explicit loop after it is defence in depth for the case where `load()`'s check is ever relaxed. Both raise the same type, so the test asserting `match="CODE_GENDER"` needs `load()`'s message to name the offending feature — it does, via `_first_disagreement`, which prints `artifact has 'CODE_GENDER'`.

- [ ] **Step 4: Register the console script**

In `pyproject.toml`, change the `[project.scripts]` block to:

```toml
[project.scripts]
creditboost-train = "creditboost.train:main"
creditboost-artifact = "creditboost.artifact_cli:main"
```

Then reinstall so the entry point exists: `pip install -e ".[train,dev]"`

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_artifact_cli.py -v`
Expected: PASS, all 13 tests. If `test_a_clean_sidecar_cannot_hide_a_dirty_booster` fails on the `match="CODE_GENDER"` assertion, check that `artifact.load()`'s message includes the artifact's own feature name — do not weaken the test to make it pass.

- [ ] **Step 6: Verify the console script works end to end**

Run: `creditboost-artifact verify --help`
Expected: usage text listing `--dir`, `--lockfile`, and `--allow-fixture`.

- [ ] **Step 7: Lint and type-check**

Run: `ruff check . && ruff format --check . && mypy src/`
Expected: all clean.

- [ ] **Step 8: Commit**

```bash
git add src/creditboost/artifact_cli.py tests/test_artifact_cli.py pyproject.toml
git commit -m "feat: add creditboost-artifact verify, the build-time artifact guard

Checks the lockfile digests, that metadata.version matches MODEL_VERSION,
feature order, ECOA prohibited features in both the sidecar and the booster's
own feature_names, and provenance, then proves the whole thing by calling
artifact.load(). Replaces the committed-artifact tests with a guard that
rejects a dirty artifact rather than asserting one particular artifact is clean."
```

---

### Task 4: `creditboost-artifact fetch`

**Files:**
- Modify: `src/creditboost/artifact_cli.py` (add the `fetch` subcommand and download helpers)
- Test: `tests/test_artifact_cli.py` (append)

**Interfaces:**
- Consumes: `ModelLock.asset_url` from Task 2; the CLI scaffolding from Task 3.
- Produces:
  - `fetch_artifact(directory: Path, lock: ModelLock) -> None`
  - `AssetNotFoundError`, `AssetDownloadError` (both subclass `artifact.ArtifactError`)
  - `RETRY_ATTEMPTS: int`, `RETRY_BACKOFF_SECONDS: float` (module-level, monkeypatched by tests to keep them fast)
- Task 7 invokes `creditboost-artifact fetch --dir <path>`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_artifact_cli.py`. These use a real loopback HTTP server rather than monkeypatching `urlopen`, so the retry logic is genuinely exercised while the suite stays offline.

```python
import http.server
import threading
from collections.abc import Iterator


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_artifact_cli.py -k "fetch or transient or missing_release" -v`
Expected: FAIL with `AttributeError: module 'creditboost.artifact_cli' has no attribute 'fetch_artifact'`

- [ ] **Step 3: Add the download implementation**

In `src/creditboost/artifact_cli.py`, add these imports at the top:

```python
import time
import urllib.error
import urllib.request
```

Add the constants and exceptions beside the existing ones:

```python
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.0
DOWNLOAD_TIMEOUT_SECONDS = 30


class AssetNotFoundError(ArtifactError):
    """The release asset does not exist. Not retried: this is a real defect."""


class AssetDownloadError(ArtifactError):
    """The release asset could not be downloaded after retrying."""
```

Add the functions:

```python
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
```

Note `_download` calls `_download_once` through the module namespace so the test can count attempts by monkeypatching it. Do not inline `_download_once` back into `_download`.

- [ ] **Step 4: Wire the subcommand into `main`**

In `main`, add the parser beside the `verify` one, immediately after the `subparsers` line:

```python
    fetch_parser = subparsers.add_parser("fetch", help="download the pinned release assets")
    _add_common(fetch_parser)
```

and add the dispatch branch before the `verify` branch inside the `try`:

```python
        if args.command == "fetch":
            lock = lockfile.read(lock_path)
            fetch_artifact(directory, lock)
            print(f"fetched {lock.release_tag} into {directory}")
```

Change the existing `if args.command == "verify":` to `elif args.command == "verify":`.

Also widen the caught exception tuple so download failures exit 1 rather than traceback — `AssetNotFoundError` and `AssetDownloadError` already subclass `ArtifactError`, so the existing `except` clause covers them. No change needed there; confirm by reading it.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_artifact_cli.py -v`
Expected: PASS, all 19 tests. The retry tests should complete in well under a second because `RETRY_BACKOFF_SECONDS` is monkeypatched to 0.

- [ ] **Step 6: Confirm the suite is still offline**

Run: `pytest tests/test_artifact_cli.py -v` with networking disabled if you have a way to do so, or read the tests and confirm every URL is `127.0.0.1`.
Expected: no test references a non-loopback host.

- [ ] **Step 7: Lint and type-check**

Run: `ruff check . && ruff format --check . && mypy src/`
Expected: all clean.

- [ ] **Step 8: Commit**

```bash
git add src/creditboost/artifact_cli.py tests/test_artifact_cli.py
git commit -m "feat: add creditboost-artifact fetch with 404-aware retries

Downloads via urllib rather than curl, which keeps an apt-get layer out of the
Docker builder stage. Transient failures back off and retry; a 404 fails
immediately, because a missing release is a defect and retrying it only delays
a truthful failure. Tested against a loopback http.server, so the suite stays
hermetic."
```

---

### Task 5: `creditboost-artifact lock` and the release helper

**Files:**
- Modify: `src/creditboost/artifact_cli.py` (add the `lock` subcommand)
- Create: `scripts/release-model.sh`
- Test: `tests/test_artifact_cli.py` (append)

**Interfaces:**
- Consumes: `lockfile.write` from Task 2; the CLI scaffolding from Task 3.
- Produces: the `lock` subcommand, taking `--tag`, `--dir`, `--lockfile`, `--asset-base-url`; and `scripts/release-model.sh <version>`.
- Task 6 runs both.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_artifact_cli.py`:

```python
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

    assert main(["lock", "--tag", "model-v0.1.0", "--dir", str(directory), "--lockfile", str(lock_path)]) == 0
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_artifact_cli.py -k lock -v`
Expected: FAIL with an argparse error, `invalid choice: 'lock'`

- [ ] **Step 3: Add the subcommand**

In `main`, add the parser after the `verify` parser:

```python
    lock_parser = subparsers.add_parser("lock", help="write a lockfile for a local artifact")
    _add_common(lock_parser)
    lock_parser.add_argument("--tag", required=True, help="release tag, e.g. model-v0.1.0")
    lock_parser.add_argument("--asset-base-url", default=lockfile.DEFAULT_ASSET_BASE_URL)
```

and the dispatch branch:

```python
        elif args.command == "lock":
            written = lockfile.write(
                lock_path,
                release_tag=args.tag,
                model_path=directory / MODEL_FILENAME,
                metadata_path=directory / METADATA_FILENAME,
                asset_base_url=args.asset_base_url,
            )
            print(f"wrote {lock_path} for {written.release_tag}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_artifact_cli.py -v`
Expected: PASS, all 22 tests.

- [ ] **Step 5: Write the release script**

Create `scripts/release-model.sh`:

```bash
#!/usr/bin/env bash
# scripts/release-model.sh — publish a locally trained model as a GitHub Release
# and rewrite the lockfile that pins it.
#
# Usage:
#   creditboost-train --data data/application_train.csv --provenance production
#   ./scripts/release-model.sh 0.2.0
#
# Training stays a manual, local, credentialed step: the Kaggle dataset is
# gitignored and CI never sees it. This script only publishes what training
# produced.
set -euo pipefail

VERSION="${1:-}"
if [ -z "${VERSION}" ]; then
  echo "usage: $0 <version>   e.g. $0 0.2.0" >&2
  exit 1
fi

TAG="model-v${VERSION}"
MODEL="models/model.json"
METADATA="models/model_meta.json"

for asset in "${MODEL}" "${METADATA}"; do
  if [ ! -f "${asset}" ]; then
    echo "missing ${asset} — run creditboost-train first" >&2
    exit 1
  fi
done

PROVENANCE=$(python -c "import json,sys; print(json.load(open('${METADATA}'))['provenance'])")
if [ "${PROVENANCE}" != "production" ]; then
  echo "refusing to release a '${PROVENANCE}' artifact; retrain with --provenance production" >&2
  exit 1
fi

META_VERSION=$(python -c "import json,sys; print(json.load(open('${METADATA}'))['version'])")
if [ "${META_VERSION}" != "${VERSION}" ]; then
  echo "artifact version is ${META_VERSION} but you asked to release ${VERSION}." >&2
  echo "Bump config.MODEL_VERSION and retrain, or release ${META_VERSION} instead." >&2
  exit 1
fi

echo "==> creating release ${TAG}"
gh release create "${TAG}" \
  --title "Model ${VERSION}" \
  --notes "XGBoost default-risk model ${VERSION}, trained on Home Credit application_train.csv." \
  "${MODEL}" "${METADATA}"

echo "==> rewriting the lockfile"
creditboost-artifact lock --tag "${TAG}"

echo
echo "==> done. Review and commit the lockfile:"
echo "    git diff models/model.lock.json"
echo "    git add models/model.lock.json && git commit -m 'chore: pin model ${TAG}'"
```

Make it executable: `chmod +x scripts/release-model.sh`

- [ ] **Step 6: Verify the script's guard rails without publishing anything**

Run: `./scripts/release-model.sh`
Expected: exits 1 with the usage message.

Run: `./scripts/release-model.sh 9.9.9`
Expected: exits 1 with "artifact version is 0.1.0 but you asked to release 9.9.9" — **before** any `gh` call. Confirm no release was created with `gh release list`.

- [ ] **Step 7: Lint and type-check**

Run: `ruff check . && ruff format --check . && mypy src/`
Expected: all clean.

- [ ] **Step 8: Commit**

```bash
git add src/creditboost/artifact_cli.py scripts/release-model.sh tests/test_artifact_cli.py
git commit -m "feat: add creditboost-artifact lock and the release helper script

Shell owns the gh interaction; Python owns the lockfile's JSON shape, so the
pydantic model stays the only definition of the file format. The script
refuses to publish a fixture artifact or a version that disagrees with the
metadata, before it calls gh."
```

---

### Task 6: Cut the `model-v0.1.0` release and commit the lockfile

**This task must complete before Task 7.** Until the release exists, the build in Task 7 has nothing to fetch, and there would be a commit range in which nothing can build.

This is the one task with a manual, credentialed step. It publishes the artifact that is **already committed** — no retraining, so the shipped model does not change.

**Files:**
- Create: `models/model.lock.json` (committed)

**Interfaces:**
- Consumes: `scripts/release-model.sh` from Task 5.
- Produces: the GitHub Release `model-v0.1.0` with both assets, and a committed lockfile. Task 7's Docker build depends on both.

- [ ] **Step 1: Confirm the committed artifact is the expected one**

Run:

```bash
shasum -a 256 models/model.json models/model_meta.json
```

Expected, exactly:

```
97869896fdb65eacd86b901438adbc2d11d52cc96baad715f9081b703450e4ac  models/model.json
0df0975d33efe7d66386c5a0ed349600e79ebec85eb4b2f9218c6e806c6d4b7d  models/model_meta.json
```

If either differs, **stop**. Someone retrained locally and the working tree is not the artifact that was reviewed and merged. Run `git checkout models/` and re-check.

- [ ] **Step 2: Confirm `gh` is authenticated**

Run: `gh auth status`
Expected: logged in to github.com with `repo` scope.

- [ ] **Step 3: Create the release**

Run: `./scripts/release-model.sh 0.1.0`
Expected: the release is created and the script prints `wrote models/model.lock.json for model-v0.1.0`.

- [ ] **Step 4: Confirm the release assets are publicly downloadable with no credentials**

This is the property the whole design rests on. Run in a directory outside the repo:

```bash
cd "$(mktemp -d)"
curl -fsSL -o model.json \
  https://github.com/edward-sf/CreditBoost.py/releases/download/model-v0.1.0/model.json
shasum -a 256 model.json
```

Expected: `97869896fdb65eacd86b901438adbc2d11d52cc96baad715f9081b703450e4ac`

If this needs a token, the repository is not public and the whole CI approach must be revisited — **stop and report** rather than working around it.

- [ ] **Step 5: Review the lockfile**

Run: `cat models/model.lock.json`

Confirm `release_tag` is `model-v0.1.0`, `asset_base_url` is the real repo URL, and the two digests match Step 1 exactly.

- [ ] **Step 6: Prove fetch-then-verify works against the real release**

Run:

```bash
creditboost-artifact fetch  --dir /tmp/cb-check
creditboost-artifact verify --dir /tmp/cb-check
```

Expected: `fetched model-v0.1.0 into /tmp/cb-check`, then `artifact in /tmp/cb-check verified against model-v0.1.0`.

- [ ] **Step 7: Commit the lockfile**

```bash
git add models/model.lock.json
git commit -m "chore: pin model release model-v0.1.0

Publishes the already-committed, already-reviewed production artifact as a
GitHub Release. The model itself is unchanged; only where its bytes live is.
The committed copy is removed in a later task, once the build no longer needs it."
```

---

### Task 7: Build the image from the release

**Files:**
- Modify: `Dockerfile:9-13` (builder stage), `Dockerfile:21` (the runtime `COPY models`)
- Modify: `.dockerignore`

**Interfaces:**
- Consumes: `creditboost-artifact fetch` and `verify` from Tasks 3-4; the release and lockfile from Task 6.
- Produces: an image whose `/app/models` came from the verified release. Task 8's CI depends on `docker build .` needing no arguments.

- [ ] **Step 1: Modify the builder stage**

In `Dockerfile`, after the existing `RUN pip install --no-cache-dir .` line, add:

```dockerfile
# The model's bytes live in a GitHub Release, not in git. Fetch them here and
# refuse to continue unless they match the committed lockfile: an image
# containing an unverified, fixture-provenance, or ECOA-violating artifact
# cannot be built at all. urllib is used rather than curl so this stage needs
# no apt-get layer.
COPY models/model.lock.json ./models/model.lock.json
RUN creditboost-artifact fetch  --dir /build/models --lockfile /build/models/model.lock.json \
 && creditboost-artifact verify --dir /build/models --lockfile /build/models/model.lock.json
```

The two commands are chained in one `RUN` on purpose: a cached layer holding fetched-but-unverified assets would be a trap.

- [ ] **Step 2: Modify the runtime stage**

Replace this line:

```dockerfile
COPY --chown=appuser:appuser models /app/models
```

with:

```dockerfile
COPY --from=builder --chown=appuser:appuser /build/models /app/models
```

The artifact no longer exists in the build context, so it must come from the builder stage.

- [ ] **Step 3: Update `.dockerignore`**

Add these lines, so a developer's local training output never enters the build context or invalidates the layer cache:

```
models/model.json
models/model_meta.json
```

Do **not** ignore `models/model.lock.json` — the build copies it explicitly.

- [ ] **Step 4: Build the image**

Run: `docker build -t creditboost:m2 .`
Expected: build succeeds. The log must show `fetched model-v0.1.0 into /build/models` and `artifact in /build/models verified against model-v0.1.0`.

- [ ] **Step 5: Run and smoke-test it**

```bash
docker run -d --rm --name cb-m2 -p 8000:8000 creditboost:m2
./scripts/smoke.sh http://localhost:8000
curl -fsS http://localhost:8000/health
docker stop cb-m2
```

Expected: the smoke test passes, and `/health` reports `"provenance":"production"` and `"model_version":"0.1.0"`.

- [ ] **Step 6: Prove the guard actually fails the build**

This is the task's real deliverable, so verify it rather than assuming it. Temporarily corrupt the lockfile:

```bash
python - <<'PY'
import json, pathlib
p = pathlib.Path("models/model.lock.json")
lock = json.loads(p.read_text())
lock["model_sha256"] = "0" * 64
p.write_text(json.dumps(lock, indent=2) + "\n")
PY
docker build -t creditboost:should-fail .
```

Expected: the build **fails**, with a `ChecksumMismatchError` naming `model.json` and printing both digests.

Then restore it and confirm the build works again:

```bash
git checkout models/model.lock.json
docker build -t creditboost:m2 .
```

Expected: success. Do not proceed until both halves behave as described.

- [ ] **Step 7: Confirm the artifact is absent from the build context**

Run: `docker run --rm creditboost:m2 ls /app/models`
Expected: `model.json`, `model_meta.json`, and `model.lock.json`.

- [ ] **Step 8: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat: build the image from the pinned model release

The builder stage fetches and verifies the release assets in a single RUN --
a cached layer of fetched-but-unverified bytes would be a trap -- and the
runtime stage copies from the builder rather than from the build context.
docker build still takes zero arguments; the lockfile carries everything."
```

---

### Task 8: Update CI

**Files:**
- Modify: `.github/workflows/ci.yml` (the `push` job's "Read the model version" step)

**Interfaces:**
- Consumes: the Dockerfile from Task 7.
- Produces: a green pipeline that builds and publishes from the release.

- [ ] **Step 1: Fix the model-version step**

The `push` job currently reads `models/model_meta.json`, which Task 9 deletes. Replace that step's `run:` line with one that reads `config.MODEL_VERSION`:

```yaml
      - name: Read the model version
        id: version
        # models/model_meta.json is no longer committed -- the artifact lives in
        # a GitHub Release. config.MODEL_VERSION is the equivalent source, and
        # creditboost-artifact verify guarantees at build time that the shipped
        # artifact's metadata carries exactly this version.
        run: |
          pip install -e .
          echo "value=$(python -c 'from creditboost import config; print(config.MODEL_VERSION)')" >> "$GITHUB_OUTPUT"
```

The `push` job already has an `actions/setup-python@v5` step before this one, so `pip` is available.

- [ ] **Step 2: Verify the YAML parses**

Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"`
Expected: `ok`

(If `pyyaml` is unavailable, run `pip install pyyaml` first — it is a dev-time check only, do not add it to `pyproject.toml`.)

- [ ] **Step 3: Confirm the version expression matches locally**

Run: `python -c "from creditboost import config; print(config.MODEL_VERSION)"`
Expected: `0.1.0`, matching `models/model_meta.json`'s `version` field before it is deleted.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: read the model version from config, not the committed metadata

models/model_meta.json stops being committed in the next task. verify already
guarantees the shipped artifact's metadata version equals config.MODEL_VERSION,
so the two sources are interchangeable by construction."
```

---

### Task 9: Remove the committed artifact

The payoff. Everything before this made the build independent of `models/*.json`; this deletes them.

**Files:**
- Delete: `models/model.json`, `models/model_meta.json`
- Modify: `.gitignore`
- Modify: `tests/test_train.py:86-112` (remove the two committed-artifact tests)

**Interfaces:**
- Consumes: Tasks 6-8. Do not start this until `docker build .` succeeds without the committed artifact present.
- Produces: a repository whose `models/` directory tracks only `model.lock.json`.

- [ ] **Step 1: Remove the two obsolete tests**

Delete `test_committed_artifact_is_present_and_is_production_provenance` (lines 86-95) and `test_committed_artifact_excludes_gender_and_raw_age` (lines 97-112) from `tests/test_train.py`, which is the end of the file.

Then remove any imports left unused by their deletion. Check the top of the file: `json` and `xgb` are used only by these two tests. Run `ruff check tests/test_train.py` after deleting; it will name any import that is now unused (`F401`).

These assertions are not lost. They live on in `tests/test_artifact_cli.py` — where they test the *guard* rather than one instance — and in `creditboost-artifact verify`, which runs at every build rather than only when someone runs pytest.

- [ ] **Step 2: Run the suite to confirm nothing else depended on the artifact**

Run: `pytest -v`
Expected: PASS. `tests/test_api.py` trains its own fixture artifact into a tmp dir and never needed `models/`; if anything else fails here, it had a hidden dependency worth understanding before deleting the files.

- [ ] **Step 3: Update `.gitignore`**

Append to the end of `.gitignore`:

```gitignore
# The trained model lives in a GitHub Release, not in git: committing it means
# every retrain appends another multi-megabyte copy that history never forgets.
# Only the lockfile pinning the release is tracked.
models/*.json
!models/model.lock.json
```

- [ ] **Step 4: Remove the files from git tracking**

```bash
git rm --cached models/model.json models/model_meta.json
```

The files stay on disk — `artifact.load()` and local `uvicorn` runs still use them — but git stops tracking them. History is deliberately **not** rewritten: the problem being solved is that *future* retrains append copies, and deleting from `HEAD` solves that completely.

- [ ] **Step 5: Confirm git now tracks only the lockfile**

Run: `git ls-files models/`
Expected: exactly one line, `models/model.lock.json`.

Run: `git status --short`
Expected: the two deletions staged, and **no** untracked `models/model.json` — `.gitignore` should be hiding it. If it shows as untracked, the ignore pattern is wrong.

- [ ] **Step 6: Prove a clean clone builds with no artifact on disk**

This is Success Criterion 2, and it is the only way to be sure nothing silently depended on the local files:

```bash
REPO="$(pwd)"
BRANCH="$(git branch --show-current)"
WORK="$(mktemp -d)"

git clone --branch "$BRANCH" "$REPO" "$WORK/clone-check"
cd "$WORK/clone-check"

ls models/            # expect ONLY model.lock.json

docker build -t creditboost:clone-check .
docker run -d --rm --name cb-clone -p 8001:8000 creditboost:clone-check
./scripts/smoke.sh http://localhost:8001
curl -fsS http://localhost:8001/health; echo
docker stop cb-clone

cd "$REPO"
```

Run this from the repo root, and note it clones your **local** branch, so the
commits from Tasks 1-8 must already be committed for it to be a fair test.

Expected: `models/` contains only the lockfile; the build succeeds; `/health` reports `"provenance":"production"`.

- [ ] **Step 7: Lint and run the full suite once more**

Run: `ruff check . && ruff format --check . && mypy src/ && pytest -v`
Expected: all clean, all passing.

- [ ] **Step 8: Commit**

```bash
git add .gitignore tests/test_train.py
git commit -m "feat: remove the committed model artifact from the repo

The bytes now live in the model-v0.1.0 GitHub Release, pinned by
models/model.lock.json. History is deliberately not rewritten: the problem was
that every future retrain appends another copy, which deleting from HEAD
solves completely.

The two committed-artifact tests are removed, not lost. Their assertions live
in creditboost-artifact verify, which runs at every build rather than only
under pytest, and in tests/test_artifact_cli.py, which asserts the guard
rejects a dirty artifact rather than that one artifact happens to be clean."
```

---

### Task 10: Update the documentation

`CLAUDE.md` is the file future work reads first. Leaving it describing a committed artifact would actively mislead.

**Files:**
- Modify: `CLAUDE.md` (Repository state, Roadmap, Architecture, Invariants, Commands)
- Modify: `README.md` (Quickstart, Train, Project layout, Development)

**Interfaces:**
- Consumes: everything above.
- Produces: documentation matching the code.

- [ ] **Step 1: Update `CLAUDE.md`'s Roadmap**

Rewrite the "Milestone 2" section to record it as done, and state what is now unscheduled:

```markdown
**Milestone 2 — model artifact storage.** Done. The trained model's bytes live in a
GitHub Release; `models/model.lock.json` pins the release tag and a sha256 per asset.
The Docker builder fetches and verifies them in a single `RUN`, so an image containing
an unverified, fixture-provenance, or ECOA-violating artifact cannot be built at all.
History was deliberately not rewritten — the problem was future retrains appending
copies, which deleting from `HEAD` solves.

- Design spec: `docs/superpowers/specs/2026-09-01-creditboost-model-artifact-storage-design.md`
- Implementation plan: `docs/superpowers/plans/2026-09-01-creditboost-model-artifact-storage.md`

**Nothing else is scheduled.** SHAP explanations, experiment tracking, the six auxiliary
Home Credit tables, batch prediction, authentication, and automated retraining remain out
of scope and unspecced.
```

- [ ] **Step 2: Update `CLAUDE.md`'s Invariants**

Replace the CI bullet with the amended wording, and add the two new invariants:

```markdown
- **CI never downloads from Kaggle.** It touches only the synthetic fixture and exactly one
  external asset: the checksum-pinned public release the lockfile names. It stays
  credential-free, because the repository is public. Training is a manual step.
- **A bad artifact cannot be built into an image.** `creditboost-artifact verify` runs
  inside the Docker builder and rejects a digest mismatch, a version disagreeing with
  `config.MODEL_VERSION`, a wrong feature order, an ECOA-prohibited feature in either the
  sidecar or the booster's own `feature_names`, or `provenance != "production"`. This is
  structural, not a test that can be skipped.
- **`models/model.lock.json` and `config.MODEL_VERSION` move in the same commit.** `verify`
  enforces it, so `/health` cannot report a version the artifact does not have.
- **Never delete a model release.** Every build pinned to it breaks, including old commits
  that used to build. This is the accepted price of the artifact leaving git.
```

Also amend the "Repository state" section's claim that "a production-trained artifact is committed at `models/model.json`" to say the artifact is released and pinned by `models/model.lock.json`.

- [ ] **Step 3: Update `CLAUDE.md`'s Commands**

Add to the command block:

```bash
creditboost-train --data data/application_train.csv --provenance production
./scripts/release-model.sh 0.2.0    # publish + rewrite models/model.lock.json
creditboost-artifact fetch          # pull the pinned release into models/
creditboost-artifact verify         # check what's on disk against the lockfile
```

Note that a fresh clone has **no** `models/*.json`, so a local `uvicorn` run needs
`creditboost-artifact fetch` first — `pytest` does not, because it trains its own fixture.

- [ ] **Step 4: Update `README.md`**

Four changes:
- **Quickstart / Run:** the "run the API directly without Docker" snippet needs a `creditboost-artifact fetch` line before `uvicorn`, since a clone no longer ships the artifact.
- **Train:** add the release step after training, and drop the sentence "A repo clone already ships with a committed, production-trained artifact".
- **Project layout:** change `models/` from "committed artifact: model.json + model_meta.json" to "model.lock.json — pins the release the artifact is fetched from" and add `artifact_cli.py`, `lockfile.py`, `hashing.py` to the `src/creditboost/` listing.
- **Development:** amend the CI paragraph to say CI downloads exactly one thing, the pinned public release asset, and remains credential-free.

- [ ] **Step 5: Verify the documented commands actually work**

Run each command in the updated docs verbatim from a scratch directory. In particular:

```bash
creditboost-artifact fetch --dir /tmp/cb-doccheck
creditboost-artifact verify --dir /tmp/cb-doccheck
```

Expected: both succeed. A command in `CLAUDE.md` that does not run is worse than no command.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: record Milestone 2 and the amended invariants

CLAUDE.md is the file future work reads first; leaving it describing a
committed artifact would actively mislead."
```

---

## Success Criteria

Verify each before considering the milestone done. Each maps to the spec's criteria of the same number.

- [ ] 1. `git ls-files models/` prints exactly `models/model.lock.json`, and that file is under 500 bytes.
- [ ] 2. A clean `git clone` followed by `docker build .` — **no arguments, no credentials** — produces a working image. (Task 9, Step 6.)
- [ ] 3. `GET /health` on that image reports `"provenance":"production"` and `"model_version":"0.1.0"`.
- [ ] 4. `pytest` passes with no network access; no test references a non-loopback host.
- [ ] 5. `docker build` fails with a legible `ChecksumMismatchError` when the lockfile digest does not match. (Task 7, Step 6.)
- [ ] 6. `docker build` fails with `ProvenanceError` when the release asset carries `provenance: "fixture"`. Covered by `tests/test_artifact_cli.py`; the Docker-level case is the same code path.
- [ ] 7. CI is green end to end on a PR, and publishes to GHCR on merge to `main`.
- [ ] 8. `scripts/release-model.sh 0.1.0` created the release, uploaded both assets, and rewrote the lockfile. (Task 6.)
- [ ] 9. `ruff check . && ruff format --check . && mypy src/` all clean.
