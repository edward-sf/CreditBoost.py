# CreditBoost — Thin End-to-End Slice

**Date:** 2026-08-30
**Status:** Approved design, ready for implementation planning

## Purpose

Build a deliberately minimal but complete path through all three of CreditBoost's
subsystems: training an XGBoost default-risk model on alternative credit data,
serving it behind a FastAPI endpoint, and shipping it through GitHub Actions to a
container registry.

The slice is minimal on purpose. Each layer is designed to be deepened in its own
later cycle; what this spec guarantees is that the seams between the layers work
end to end before any one layer gets sophisticated. For an MLOps-focused project,
those seams are the product.

## Scope

**In scope:** dataset ingestion, a curated feature transform, model training with
evaluation, a versioned artifact, a FastAPI prediction service, a test suite, a
Docker image, and a CI/CD pipeline that lints, tests, builds, smoke-tests, and
publishes.

**Out of scope** (candidates for later cycles): the six auxiliary Home Credit
tables, SHAP or other per-prediction explanations, a model registry or experiment
tracking, runtime artifact fetching, batch prediction, authentication, and
automated retraining.

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Dataset | Home Credit Default Risk, `application_train.csv` only | Canonical thin-file / alternative-data source; auxiliary tables add scope without adding slice value |
| Artifact delivery | Committed to repo, baked into image at build | Self-contained on clone, immutable image; registry is the obvious cycle-2 upgrade |
| Feature set | 19 request fields → 21 model features | Readable request schema and a small transform; widening later is cheap |
| Response | probability + risk band + model version | Fully testable, no extra dependencies, every response traceable to an artifact |
| Repo structure | Single package, shared feature module | Makes train/serve skew structurally impossible |
| Registry | GHCR | Authenticates with the built-in `GITHUB_TOKEN`; no extra secrets |

## Architecture

```
creditboost/
├── pyproject.toml
├── src/creditboost/
│   ├── config.py       # paths, feature list, category levels, thresholds
│   ├── features.py     # shared transform + FEATURE_ORDER
│   ├── schema.py       # Pydantic request/response/metadata models
│   ├── data.py         # CSV load, validation, stratified split
│   ├── train.py        # CLI: fit, evaluate, write artifact
│   ├── artifact.py     # save/load model + metadata
│   └── serve/
│       ├── app.py      # FastAPI factory, routes
│       └── deps.py     # model loaded once at startup
├── models/
│   ├── model.json
│   └── model_meta.json
├── tests/
│   ├── fixtures/sample.csv
│   ├── test_features.py
│   ├── test_artifact.py
│   ├── test_api.py
│   └── test_train.py
├── Dockerfile
├── .dockerignore
└── .github/workflows/ci.yml
```

### Preventing train/serve skew

This is the central design constraint. `features.py` exports `FEATURE_ORDER` and a
single `transform()` used by both the training CLI and the serving app. Training
copies `FEATURE_ORDER` into `model_meta.json` at fit time. At startup, serving
compares the metadata's order against the code's; on mismatch the process exits
non-zero rather than serving predictions the model was never trained for.

### Dependency direction

`serve/` imports `artifact`, `features`, `schema`, and `config` — never `data` or
`train`. Training dependencies (pandas, scikit-learn) live in an optional
`[train]` extra, so the runtime image installs only what serving needs. A second
`[dev]` extra carries pytest, ruff, and mypy. The package targets
`requires-python = ">=3.11"`, matching the CI matrix; the image pins 3.12.

## Feature Design

**Raw numeric (10):** `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`,
`AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`,
`DAYS_EMPLOYED`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`

**Binary (2):** `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`. Stored as `Y`/`N` strings in
the raw CSV; `transform()` maps them to 1.0 / 0.0, and any other or missing value
to NaN.

**Categorical (6):** `NAME_CONTRACT_TYPE`, `NAME_INCOME_TYPE`,
`NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`,
`OCCUPATION_TYPE`

**Derived (3):** `credit_to_income` (`AMT_CREDIT / AMT_INCOME_TOTAL`),
`annuity_to_income` (`AMT_ANNUITY / AMT_INCOME_TOTAL`), `employed_to_age`
(`DAYS_EMPLOYED / DAYS_BIRTH`). All three are computed inside `transform()` so
they cannot drift between training and serving. Each guards its denominator:
a zero denominator yields NaN rather than raising.

**Order of operations is significant.** The `DAYS_EMPLOYED` sentinel scrub
described below runs *before* the derived ratios are computed. Computing
`employed_to_age` from the raw `365243` sentinel would produce a large positive
ratio that looks like a plausible value rather than the missing datum it is.

Total: **21 model features** from **19 request fields**. `DAYS_BIRTH` is accepted
in the request and consumed to build `employed_to_age`, but the raw value is not
itself a model feature and does not appear in `FEATURE_ORDER`.

### Fairness and regulatory constraints

`CODE_GENDER` is excluded from the feature set. Under the Equal Credit Opportunity
Act and Regulation B, sex is a prohibited basis for credit decisions in the US, so
a model consuming it is a compliance defect rather than a stylistic preference.

Age is protected under the same rule, which makes `DAYS_BIRTH` awkward. Raw age is
therefore excluded as a standalone feature; it enters the model only through the
`employed_to_age` ratio, where it normalizes employment tenure rather than acting
as an independent age signal. This is a documented judgment call, not an
oversight, and it is a reasonable thing to revisit with compliance input.

### Dataset quirks the transform owns

- **`DAYS_EMPLOYED == 365243`** is Home Credit's sentinel for "not employed."
  Left raw it becomes a roughly thousand-year tenure that silently distorts the
  model. `transform()` maps it to NaN.
- **`EXT_SOURCE_1` is missing in over half of rows.** We do not impute. XGBoost
  routes NaN natively, and for thin-file borrowers a missing external score is
  itself signal. These fields are `Optional[float] = None` in the request schema
  and map to NaN in the transform.

### Categorical handling carries no fitted state

Category levels are declared as explicit tuples in `config.py`. `transform()`
builds pandas `Categorical` columns with exactly those levels and XGBoost trains
with `enable_categorical=True`. There is no encoder to serialize and no second
artifact to keep in sync. An unrecognized level becomes NaN rather than raising,
so an unfamiliar occupation code degrades a prediction instead of failing a
request.

## Data Flow

### Training

1. `creditboost-train` loads `application_train.csv` — by default from
   `data/application_train.csv`, overridable by CLI flag. The `data/` directory is
   gitignored; obtaining the file is a manual, credentialed step outside CI. The
   command asserts required columns are present and records the file's SHA-256 for
   provenance.
2. Stratified train/validation split on `TARGET` with a fixed seed. Stratification
   is required: positives are roughly 8% of rows.
3. `features.transform()` produces both matrices.
4. XGBoost fits with `scale_pos_weight` set for the class imbalance and early
   stopping on validation AUC.
5. Evaluation records **ROC-AUC** (primary), **PR-AUC** (the informative metric at
   this prevalence), and **Brier score** (we serve a probability, so calibration is
   part of correctness). Accuracy is deliberately not reported; at an 8% base rate
   it carries no information.
6. If validation ROC-AUC falls below the floor in `config.py` (initially 0.70), the
   script exits non-zero and writes no artifact. A bad retrain cannot produce a
   model, so nothing downstream needs to detect one.
7. `artifact.save()` writes `model.json` and `model_meta.json`.

### Artifact contract

`model_meta.json` contains:

```json
{
  "version": "0.1.0",
  "trained_at": "2026-08-30T12:00:00Z",
  "dataset_sha256": "…",
  "n_train_rows": 245000,
  "feature_order": ["EXT_SOURCE_1", "…"],
  "metrics": {"roc_auc": 0.75, "pr_auc": 0.24, "brier": 0.068},
  "xgboost_version": "2.1.0"
}
```

`version` is a hand-bumped semver constant in `config.py`. The metric values above
are illustrative of the format; actual values are written at training time.

### Serving

A lifespan handler loads the artifact once into module state and runs the startup
checks. Routes:

- `POST /predict` — Pydantic validation → `transform()` → `booster.predict()` →
  probability → band → response carrying `probability`, `risk_band`, and
  `model_version`.
- `GET /health` — liveness plus loaded model version.
- `GET /metadata` — the full metadata block.

### Risk bands

Thresholds live in `config.py`, not in the artifact: where to cut low from medium
is a business policy that changes without retraining. Initial values, chosen
relative to the ~8% base rate: **low** below 0.10, **medium** from 0.10 to below
0.30, **high** at 0.30 and above.

## Error Handling

| Condition | Behavior |
|---|---|
| Malformed or missing request field | 422 from Pydantic; amounts and incomes carry `gt=0` constraints |
| Unrecognized categorical level | Mapped to NaN; request succeeds |
| Missing external scores | Valid input; NaN is meaningful signal |
| Artifact missing, unparseable, or feature-order mismatched | Process exits non-zero at startup; the container never accepts traffic |
| Unhandled exception | 500 with a generic body; full traceback to logs only |

Logging is structured JSON carrying request id, latency, model version, and
predicted band. Applicant financial fields are never logged — that is precisely
the PII that should not accumulate in log aggregation.

## Testing

### Fixture

`tests/fixtures/sample.csv` (~200 rows) is **synthetic**: it matches Home Credit's
schema in column names, dtypes, and plausible ranges, but contains no real rows.
Kaggle's competition terms restrict redistribution, so committing real data to a
public repository is a licensing problem. Generating the fixture also lets us
guarantee the edge cases are present — rows carrying the `365243` sentinel, rows
with all three external scores null, and both target classes.

### Layers

**`test_features.py`** carries the most weight, since everything depends on the
transform being correct:

- `365243` maps to NaN
- nulls are preserved as NaN, not imputed to zero
- derived ratios are correct, with zero denominators yielding NaN
- an unknown categorical level maps to NaN without raising
- output column order equals `FEATURE_ORDER` exactly
- repeated calls on the same input are byte-identical
- **parity:** the same record as an API dict and as a training DataFrame row
  produces identical matrices

**`test_artifact.py`** — save/load round-trip; metadata validates; loading with a
mismatched `feature_order` raises.

**`test_api.py`** — `TestClient` against a small model trained on the fixture in a
session-scoped fixture. Health returns a version; a happy-path predict returns a
probability in [0, 1] with a band matching the configured thresholds; a missing
field and a negative income each return 422; all-null external scores still
returns 200, which is the thin-file borrower this product exists to score.

**`test_train.py`** — slow-marked smoke test that training on the fixture completes
and writes both files with metrics populated.

### Not tested in CI

Model quality. Asserting AUC thresholds in CI is slow and flaky; the AUC floor in
the training script covers it at the point where a bad model would actually be
produced.

## CI/CD

CI is hermetic. The Kaggle download never runs in it — training is a local, manual
step, and CI touches only the generated fixture and the committed artifact. That
keeps it fast and credential-free.

```
pull_request  → lint, test, build
push to main  → lint, test, build, push
```

- **lint** — `ruff check`, `ruff format --check`, `mypy src/`
- **test** — matrix on Python 3.11 and 3.12, pytest with coverage
- **build** — needs lint and test. Builds the image, runs the container, and curls
  `/health` and `/predict` against it. This catches the failure a plain
  `docker build` sails past: an image that builds cleanly without the model in it.
- **push** — needs build. Gated on
  `github.event_name == 'push' && github.ref == 'refs/heads/main'`, with
  `permissions: packages: write` scoped to this job alone so fork PRs never
  receive registry credentials. Publishes to GHCR tagged `latest`,
  `sha-<short>`, and the model version.

Concurrency cancellation on repeated pushes to the same PR; pip caching via
`actions/setup-python` and Docker layer caching via buildx.

### Image

Multi-stage build installing into a virtualenv, `python:3.12-slim` runtime,
non-root user, `HEALTHCHECK` against `/health`, and a `.dockerignore` excluding
`data/`, `tests/`, and `.git`. Only runtime dependencies reach the final layer.

## Success Criteria

1. `pip install -e .[train,dev] && pytest` passes on a fresh clone with no Kaggle
   credentials.
2. `creditboost-train` on the real dataset produces `model.json` and
   `model_meta.json` with validation ROC-AUC at or above 0.70.
3. `docker build` followed by `docker run` yields a container whose `/health`
   reports the model version and whose `/predict` scores a thin-file record — one
   with all external scores null — without error.
4. A pull request runs lint, test, and build without touching registry
   credentials.
5. A merge to main publishes a tagged image to GHCR.
6. The startup feature-order check demonstrably fails the container when the
   artifact and code disagree.
