# CreditBoost

Loan-default risk scoring for thin-file borrowers. A FastAPI service loads a
checksum-pinned XGBoost artifact and returns a default probability, a risk band (`low` / `medium` /
`high`), and the model version for a loan applicant.

**Domains:** `Lending`, `Risk Analytics`

Milestone 1 built a deliberately minimal, complete path through all three subsystems —
training, a served model, and CI/CD — for a fixed 21-feature model trained on the Home
Credit Default Risk dataset. Milestone 2 moved the model's bytes out of git into a GitHub
Release, leaving a checksum-pinned `models/model.lock.json` in their place. Milestone 3
added adverse action reason codes, and removed marital status as a model feature. Milestone 4
measures disparate impact across protected attributes at training time and refuses to write
a model that fails the four-fifths rule. Milestone 5 searches for a less discriminatory
alternative model specification on a split nested inside the training data, applies a 0.01
AUC budget, and records the frontier in every production artifact. See [`CLAUDE.md`](CLAUDE.md)
for the architecture, invariants, and roadmap; see the design spec and implementation plan
under `docs/superpowers/` for the full reasoning.

## Tech stack

- **XGBoost** for the model, served directly via its own `Booster` — no scikit-learn in the
  runtime image. Scikit-learn is used only during training, for metrics.
- **FastAPI** / **uvicorn** for the serving API.
- **Pandas** for the shared train/serve feature transform.
- **Docker**, multi-stage build, non-root user, `HEALTHCHECK`.
- **GitHub Actions** — lint, test, build, and publish to GHCR.

## Quickstart

### Prerequisites

- Python 3.12.
- On macOS, the OpenMP runtime, which `xgboost` needs to import:
  ```bash
  brew install libomp
  ```
- A python.org macOS build ships an empty TLS trust store until you run its
  `Install Certificates.command`; without it `creditboost-artifact fetch` fails with
  `CERTIFICATE_VERIFY_FAILED`. Homebrew and Linux Pythons are unaffected, as is the
  container build.

### Install and test

```bash
git clone <this repo>
cd CreditBoost.py
pip install -e ".[train,dev]"
pytest
```

`pytest -m "not slow"` skips the two training smoke tests if you want a faster loop.

### Train

Training needs the real dataset, which is **not** included in this repo (Kaggle's terms
forbid redistributing it). Download `application_train.csv` from Kaggle's
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) competition
and place it at `data/application_train.csv` (gitignored).

```bash
creditboost-search --data data/application_train.csv     # print the frontier, write nothing
creditboost-train --data data/application_train.csv --provenance production
creditboost-train --data data/application_train.csv --provenance production --search
```

The `search` command is read-only and prints a frontier of model specifications ranked by
their adverse impact ratio at a matched approval rate. The `--search` flag on `train` runs
the same search at training time and adopts a less discriminatory alternative if one clears
the AUC budget; otherwise it trains the baseline and stamps the frontier into the artifact.
The search adds roughly four minutes to a training run. This writes `models/model.json` and
`models/model_meta.json`, refusing to write anything if validation ROC-AUC falls below the
floor in `config.py` or if `provenance` is `production` and no candidate could be scored.

Publish what training produced as a GitHub Release, and rewrite the lockfile that pins it:

```bash
./scripts/release-model.sh 0.2.0
git add models/model.lock.json && git commit -m "chore: pin model model-v0.2.0"
```

The release tag's version must match `config.MODEL_VERSION`; the script refuses to publish
a fixture artifact or a mismatched version before it calls `gh`.

### Run

```bash
docker build -t creditboost:dev .
docker run -d --rm -p 8000:8000 creditboost:dev
./scripts/smoke.sh http://localhost:8000
```

Or run the API directly without Docker. A clone carries no artifact — only the lockfile —
so fetch the pinned release first:

```bash
creditboost-artifact fetch
uvicorn creditboost.serve.app:app --port 8000
```

## API

### `GET /health`

Liveness/readiness check. Returns the loaded model's version and provenance.

```json
{"status": "ok", "model_version": "0.1.0", "provenance": "production"}
```

### `GET /metadata`

The full metadata sidecar: training timestamp, dataset hash, row count, feature order,
validation metrics, xgboost version, provenance, and a `fairness` block. The ratio is the
four-fifths adverse impact ratio measured on the validation split at training time, adverse
meaning the applicant was not auto-approved (`band != "low"`); a model below 0.80 on any
measured attribute cannot be trained.

```json
{
  "fairness": {
    "adverse_definition": "band != low",
    "band_low_max": 0.1,
    "min_group_size": 100,
    "attributes": [
      {
        "attribute": "CODE_GENDER",
        "adverse_impact_ratio": 0.868,
        "groups": [
          {"group": "F", "adverse_rate": 0.222, "n": 40561},
          {"group": "M", "adverse_rate": 0.325, "n": 20940}
        ]
      }
    ]
  }
}
```

### `POST /predict`

Scores one applicant. Only `AMT_INCOME_TOTAL`, `AMT_CREDIT`, and `DAYS_BIRTH` are required;
everything else is optional and degrades to `null` — which XGBoost treats as its own
signal, never imputed — because thin-file borrowers, by definition, have sparse records.

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"AMT_INCOME_TOTAL": 100000, "AMT_CREDIT": 400000, "DAYS_BIRTH": -12000}'
```

```json
{
  "probability": 0.1266,
  "risk_band": "medium",
  "model_version": "0.2.0",
  "reasons": [
    {"code": "EMPLOYMENT_PROFILE", "description": "Employment details were not provided"},
    {"code": "HOUSEHOLD_SIZE", "description": "Household size was not provided"},
    {"code": "LOAN_SIZE", "description": "Loan amount is high relative to income"},
    {"code": "ASSETS", "description": "No asset ownership information was provided"}
  ]
}
```

`reasons` lists the principal factors increasing this applicant's risk, most significant
first, at most four. Where a caller takes adverse action on the score, these are the
specific principal reasons ECOA / Regulation B §1002.9 requires. Only factors that pushed
the score upward appear — a feature that helped the applicant is not a reason for denial —
and the wording distinguishes data that was *unfavourable* from data that was *absent*,
which for a thin-file borrower is the difference between a true statement and a false one.

Protected attributes never appear: sex is not accepted at all, and age and marital status
are accepted but never scored on, so no reason can name them.

`CODE_GENDER` is never accepted, and raw `DAYS_BIRTH` never reaches the model directly —
under ECOA / Regulation B, sex and age are prohibited bases for a US credit decision. Age
enters only through a derived `employed_to_age` ratio.

A container that cannot verify its own artifact against the code's expected feature layout
exits non-zero at startup rather than accepting traffic serving wrong predictions.

## Project layout

```
src/creditboost/
  config.py           # constants: feature lists, paths, risk-band thresholds, AUC floor
  features.py          # the one shared transform, imported by both train.py and serve/
  schema.py             # PredictRequest / PredictResponse / ModelMetadata (pydantic)
  banding.py            # probability -> risk band
  data.py                # dataset loading, validation, train/valid split (train-only)
  artifact.py            # save/load + the train/serve skew gate
  reasons.py             # contributions -> at most four adverse action reasons
  fairness.py            # adverse impact ratios; the training-time gate
  hashing.py             # file_sha256, dependency-free so both sides can import it
  lockfile.py            # the ModelLock pointer: release tag + a sha256 per asset
  artifact_cli.py        # creditboost-artifact: fetch / verify / lock
  train.py                # training CLI (creditboost-train)
  serve/
    app.py                 # FastAPI app: /health, /metadata, /predict
    deps.py                 # process-wide loaded-model state
    logging_config.py        # structured JSON logging for the creditboost logger tree
tests/                        # pytest, one module per src file, plus a synthetic fixture
models/                        # model.lock.json — pins the release the artifact is fetched from
data/                           # gitignored; the real Kaggle CSV goes here
docs/superpowers/                # design spec and implementation plan
.github/workflows/ci.yml          # lint, test, build, smoke-test, publish to GHCR
```

## Development

```bash
ruff check . && ruff format --check .   # CI checks formatting; run before committing
mypy src/
```

CI never downloads from Kaggle — it runs against a synthetic fixture
(`tests/fixtures/sample.csv`) and downloads exactly one external thing: the checksum-pinned
public release asset the lockfile names, which needs no credentials. Training against the
real dataset is a manual, local step.
