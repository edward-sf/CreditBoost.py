# CreditBoost — Model Artifact Storage

**Date:** 2026-09-01
**Status:** Approved design, ready for implementation planning
**Milestone:** 2 (Phase 2)
**Supersedes:** the artifact-delivery decision in
`2026-08-30-creditboost-thin-slice-design.md` ("Committed to repo, baked into image at build")

## Purpose

Get the trained model's bytes out of git without giving up anything Milestone 1 earned.

Milestone 1 committed `models/model.json` to the repository. `CLAUDE.md` records this as
poor practice for a permanent arrangement and states why: git never forgets, so every
retrain appends another full multi-megabyte copy that cannot be removed without rewriting
history, and it welds the model lifecycle to the code lifecycle. It was accepted for
Milestone 1 because a self-contained clone is what let Tasks 1–10 be verified with no
credentials at all.

Milestone 1 is now proven — CI is green end to end on `main` and the image publishes to
GHCR — so the debt comes due. This milestone moves the bytes to a GitHub Release and
leaves a small, reviewable pointer in git in their place.

## Scope

**In scope:** a committed lockfile pinning a model release by tag and checksum; a CLI to
fetch, verify, and lock artifacts; a release helper script; Dockerfile and CI changes to
build from a downloaded artifact; removal of `models/*.json` from `HEAD`; migration of the
shipped-artifact guarantees from pytest into the build.

**Out of scope, and unchanged by this milestone:** the feature set, the transform, the
model itself, the API surface, risk banding, logging, the runtime image's dependency set,
and anything that runs at container startup. Also out of scope and still unspecced: SHAP
explanations, experiment tracking, a real model registry, the six auxiliary Home Credit
tables, batch prediction, authentication, and automated retraining.

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Storage backend | GitHub Releases | No new infrastructure, no new credentials; the repo is public, so assets download with plain `urllib` and CI stays credential-free |
| Fetch timing | Build time, into the builder stage | Keeps the image self-contained and immutable, adds no runtime network dependency, leaves the boot-time skew gate untouched |
| CI artifact | Always the real release, on every PR | One code path, exercised continuously; a broken URL or checksum fails the PR that caused it rather than surfacing after merge |
| Version pin | A committed `models/model.lock.json` with tag + sha256 per asset | A model bump becomes a reviewable three-line diff; integrity is verified rather than assumed; `config.py` stays about features |
| Shipped-artifact guard | `creditboost-artifact verify`, run inside the Docker build | Structural — an image containing a bad artifact cannot be built, on any machine, not merely one that runs pytest |
| Release creation | `scripts/release-model.sh`, invoked manually after training | Keeps `train.py` offline, dependency-free, and hermetically testable |
| Git history | Delete from `HEAD`; do **not** rewrite history | The problem is future retrains appending copies, which deleting from `HEAD` solves completely; rewriting a public repo's merged history is disruptive out of proportion to reclaiming 1.2MB once |

## Architecture

The model's bytes leave git; a pointer to them stays. Nothing about the train/serve skew
architecture changes.

```
LOCAL (manual, credentialed)        GIT (small, reviewable)       BUILD (public, verified)
────────────────────────────        ───────────────────────       ───────────────────────
creditboost-train                                                 docker build
  ↓ models/model.json (1.2MB)                                        ↓
scripts/release-model.sh 0.2.0                                    artifact fetch
  ↓ gh release create model-v0.2.0                                   ↓
  ↓ upload model.json + meta    →   models/model.lock.json   →    artifact verify
  ↓ artifact lock                   { tag, asset_base_url,          ↓ sha256 match?
                                      model_sha256,                 ↓ version == MODEL_VERSION?
                                      metadata_sha256 }             ↓ provenance == production?
                                  ← reviewed + committed            ↓ feature order / ECOA?
                                                                    ↓ artifact.load() succeeds?
                                                                 COPY into runtime image
```

### What this deliberately does not touch

- **Dependency direction.** `artifact_cli` imports `artifact`, `config`, `schema`, and
  `lockfile` — exactly the set `serve/` already imports. It runs in the builder stage,
  which has only the base package installed. scikit-learn stays out of the runtime image
  and the existing subprocess test that enforces this keeps passing unmodified.
- **The boot-time skew gate.** `artifact.load()` still runs in the FastAPI lifespan
  handler and still exits the process non-zero on mismatch. No new code runs at startup.
- **`config.py`.** It does not learn about releases. `MODEL_VERSION` stays where it is and
  keeps meaning what it means.

`verify` is not a replacement for `load()` but a strictly earlier and stricter partner: it
catches at build time what `load()` would catch at boot time, plus the two things `load()`
cannot know — the checksum and the provenance.

## Components

### `models/model.lock.json`

The pointer, committed. The only file in `models/` that git tracks after this milestone.

```json
{
  "release_tag": "model-v0.1.0",
  "asset_base_url": "https://github.com/edward-sf/CreditBoost.py/releases/download",
  "model_sha256": "…",
  "metadata_sha256": "…",
  "released_at": "2026-09-01T03:38:24+00:00"
}
```

`asset_base_url` lives here rather than in `config.py` for two reasons: it keeps release
coordination out of `config.py`, and it lets a fork repoint at its own releases without
touching code.

There is deliberately **no `model_version` field**. It would be a third place the version
could disagree with itself. Instead `verify` cross-checks the downloaded metadata's
`version` against `config.MODEL_VERSION`, which forces a lockfile bump and a
`MODEL_VERSION` bump into the same commit and keeps `/health` honest.

### `src/creditboost/lockfile.py`

A pure library, roughly sixty lines, with no network access and no CLI:

- `ModelLock` — a pydantic model, sibling to `ModelMetadata` in `schema.py`
- `read(path) -> ModelLock`
- `write(path, release_tag, asset_base_url, model_path, metadata_path) -> None`
- `sha256_of(path) -> str`

Owning the file format in one pydantic model means the shell release script never has to
hand-assemble JSON.

### `creditboost-artifact` (`src/creditboost/artifact_cli.py`)

One console script, three subcommands, all lockfile-driven so they take no arguments in
the common case.

**`fetch`** — reads the lockfile, downloads both assets with `urllib`, writes them to the
model directory. Using `urllib` rather than `curl` is what keeps an `apt-get` layer out of
the builder stage.

**`verify`** — the guard. Cheap explicit checks run first, so failures are legible; then
`artifact.load()` runs as the end-to-end proof:

1. sha256 of each asset matches the lockfile
2. `metadata.version == config.MODEL_VERSION`
3. `metadata.feature_order == config.FEATURE_ORDER`
4. neither `CODE_GENDER` nor `DAYS_BIRTH` appears in `feature_order` **or** in the
   booster's own `feature_names`
5. `provenance == "production"`, relaxable with `--allow-fixture` for local work
6. `artifact.load()` succeeds — which covers xgboost major-version compatibility and
   parseability for free

**`lock`** — writes the lockfile from local files after a release has been uploaded.

### `scripts/release-model.sh`

Orchestrates the GitHub side, which shell is genuinely good at, following the
`scripts/smoke.sh` precedent:

```bash
creditboost-train --data data/application_train.csv --provenance production
./scripts/release-model.sh 0.2.0
```

It runs `gh release create model-v0.2.0`, uploads `model.json` and `model_meta.json`, then
calls `creditboost-artifact lock` to rewrite `models/model.lock.json` for review and
commit. Shell owns `gh`; Python owns the JSON shape.

`train.py` is not modified. It keeps no network dependency, no `gh` dependency, and its
training smoke tests stay hermetic.

### `Dockerfile`

The builder stage gains three lines. The runtime stage is untouched.

```dockerfile
COPY models/model.lock.json ./models/
RUN creditboost-artifact fetch  --dir /build/models
RUN creditboost-artifact verify --dir /build/models
```

The runtime stage's existing `COPY --chown=appuser:appuser models /app/models` reads from
the build context, where the artifact will no longer exist. It changes to read from the
builder stage instead:

```dockerfile
COPY --from=builder --chown=appuser:appuser /build/models /app/models
```

This carries the verified artifact forward and also carries `model.lock.json` into the
image, which is harmless and mildly useful for provenance questions in a running
container.

`docker build .` still takes **zero arguments** — the lockfile carries everything needed.
This preserves today's developer experience exactly.

`.dockerignore` gains `models/model.json` and `models/model_meta.json` so a developer's
local training output never pollutes the build context or invalidates the layer cache.

### `.github/workflows/ci.yml`

- The `build` job is unchanged in structure; the fetch and verify happen inside
  `docker build`. Its smoke test now scores the real production model.
- The `push` job's *"Read the model version"* step currently reads
  `models/model_meta.json`, **a file that will no longer exist in a checkout.** It changes
  to read `config.MODEL_VERSION`, which `verify` now guarantees equals the shipped
  metadata's version.

### `.gitignore`

Gains `models/*.json` with a `!models/model.lock.json` exception.

## Error Handling

New exception types in `lockfile.py` and `artifact_cli.py` mirror `artifact.py`'s existing
style: a base class, specific subclasses, and every failure path raises rather than
returning a questionable result.

| Failure | Behaviour |
|---|---|
| Lockfile missing or malformed | `LockfileError`; build fails immediately |
| Asset 404 | `AssetNotFoundError`; **no retry**; names the tag and points at `scripts/release-model.sh` |
| Transient network error | 3 attempts with backoff, then `AssetDownloadError` |
| Checksum mismatch | `ChecksumMismatchError`; **no retry**; prints expected and actual in full |
| `metadata.version` ≠ `config.MODEL_VERSION` | `VersionMismatchError`; names both and says to bump them together |
| `provenance != "production"` | `ProvenanceError`; this is what stops a fixture model reaching GHCR |
| Feature order, ECOA, xgboost version | Reuses `artifact.py`'s existing error types verbatim |

The retry distinction is deliberate. A transient GitHub blip should not fail an unrelated
PR, but a 404 or a bad checksum is a real defect, and retrying it only delays a truthful
failure by thirty seconds.

**Runtime error handling is entirely unchanged.** No new code runs at container startup.

## Invariant Ledger

### Gained

- An image cannot be built around an unverified, fixture-provenance, or ECOA-violating
  artifact. This is structural — `docker build` fails — not a test that can be skipped.
- `models/model.lock.json` and `config.MODEL_VERSION` move in the same commit; `verify`
  enforces it.

### Amended

- *"CI never downloads from Kaggle"* remains true and remains the point. It gains a
  clause: CI downloads exactly one thing, the checksum-pinned public release asset. CI
  stays **credential-free**, because the repository is public.
- *"The artifact is committed to the repo"* becomes *"only the pointer is committed"*:
  `models/*.json` is gitignored, with `models/model.lock.json` excepted.

### Unchanged

Each of these could plausibly have been broken by this milestone and is not:

- One-way dependency direction; no scikit-learn in the runtime image.
- `features.transform()` as the single shared train/serve transform.
- `artifact.load()` as the boot-time skew gate.
- No applicant financial field is ever logged.
- The test fixture stays synthetic and is never sampled from the real dataset.
- `FEATURE_ORDER` has 21 entries; `REQUEST_FIELDS` has 19.
- `CODE_GENDER` and raw `DAYS_BIRTH` never appear in any feature list.

## Accepted Risks

Recorded rather than solved, so the trade is on the record:

1. **The build now depends on GitHub Releases being available and the asset still
   existing.** Deleting a release breaks every build pinned to it, including old commits
   that used to build. The mitigation is discipline — never delete a model release — not
   code. This is the real price of the milestone, and it is the normal price.

   The blast radius is narrower than "everything breaks", and worth stating precisely:
   CI publishes every image to GHCR tagged by commit sha, so a deleted release costs the
   ability to **rebuild** an old commit, not the ability to **deploy** it — the built
   image is still pullable. Recovery is also possible rather than theoretical: any
   existing image carries `/app/models/model.json`, so a lost release can be
   reconstructed with `docker cp` from a pulled image, and the lockfile's digest proves
   the recovered bytes are the right ones.
2. **The build adds one more host it must reach.** This is a smaller change than it first
   appears, and the first draft of this spec overstated it. `docker build` is *already*
   not offline: the builder stage runs `pip install --no-cache-dir .`, which pulls
   fastapi, uvicorn, xgboost, pandas and numpy from PyPI, on top of pulling the
   `python:3.12-slim` base image from Docker Hub. The change is therefore from two
   external hosts to three — adding `github.com` to `pypi.org` and Docker Hub — not from
   offline to online. `pytest` stays fully offline either way.

## Testing

The governing rule: **`pytest` stays fully hermetic and offline.** No test touches the
network, before or after this change.

### `tests/test_lockfile.py`

Write/read round-trip; `sha256_of` against a known value; rejection of malformed JSON;
rejection of a missing required field.

### `tests/test_artifact_cli.py`

A parametrised table of synthetic artifacts built into `tmp_path`, each asserting that
`verify` fails for exactly one reason:

| Case | Expected |
|---|---|
| Well-formed production artifact | passes |
| Model bytes altered after locking | `ChecksumMismatchError` |
| `provenance: "fixture"` | `ProvenanceError` |
| Same, with `--allow-fixture` | passes |
| `metadata.version` ≠ `config.MODEL_VERSION` | `VersionMismatchError` |
| `feature_order` reordered | `FeatureOrderMismatchError` |
| Metadata clean, but booster `feature_names` contains `CODE_GENDER` | `FeatureOrderMismatchError` |

The last row earns its keep. It replaces the deleted committed-artifact ECOA test and is
strictly stronger: the old test asserted that one particular artifact was clean, whereas
this asserts that the *guard rejects a dirty one* — including the case where the metadata
sidecar looks correct and only the booster's own baked-in `feature_names` are wrong.

### `fetch` tests

Served by an `http.server` instance on an ephemeral port in a fixture rather than by
monkeypatching `urlopen`, so the real code path including retry logic is exercised while
staying entirely on loopback: success, 404-without-retry, and a flaky-then-succeeds case
asserting the backoff actually retries.

### Removed

The two tests at `tests/test_train.py:86-112`
(`test_committed_artifact_is_present_and_is_production_provenance` and
`test_committed_artifact_excludes_gender_and_raw_age`). They read `config.METADATA_PATH`,
which will not exist in a clean checkout.

Their assertions do not vanish. They move into `verify`, where they are enforced at every
build rather than only when someone runs pytest, and into the table above.

Note that the ECOA property was already enforced structurally before this milestone:
`artifact.load()` requires both `metadata.feature_order` and the booster's
`feature_names` to equal `config.FEATURE_ORDER` exactly, and `tests/test_config.py`
guarantees `FEATURE_ORDER` contains neither `CODE_GENDER` nor raw `DAYS_BIRTH`. The
deleted test was defence in depth against the shipped bytes, and `verify` provides that
defence in a stronger form.

### Unchanged and still passing

The subprocess test asserting scikit-learn is never imported by `serve/`; every
`features.py` train/serve parity test; and `tests/test_api.py`, which already trains its
own fixture artifact into a temporary directory and never depended on `models/` at all.

## Sequencing Constraint

The `model-v0.1.0` release must be cut from the **currently committed** artifact before
the build flips over to fetching and before `models/*.json` is deleted from `HEAD`.
Otherwise there is a commit range in which nothing can build.

The implementation plan must order its tasks accordingly: create the release first, land
the lockfile and the CLI second, flip the Dockerfile and CI third, and delete the
committed artifact last.

## Success Criteria

1. `models/model.json` and `models/model_meta.json` are absent from `HEAD`;
   `models/model.lock.json` is present and under 500 bytes.
2. A clean `git clone` followed by `docker build .` — with **no arguments and no
   credentials** — produces a working image.
3. `GET /health` on that image reports `provenance: "production"` and a `model_version`
   equal to `config.MODEL_VERSION`.
4. `pytest` passes with no network access.
5. `docker build` fails, with a legible message naming the reason, when the lockfile
   checksum does not match the release asset.
6. `docker build` fails when the release asset carries `provenance: "fixture"`.
7. CI is green end to end on a PR and publishes to GHCR on merge to `main`.
8. `scripts/release-model.sh` takes a version argument, creates the release, uploads both
   assets, and rewrites the lockfile.
