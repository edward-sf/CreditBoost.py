# CreditBoost

Loan-default risk scoring for thin-file borrowers. A FastAPI service loads a committed
XGBoost artifact and returns a default probability, a risk band (`low` / `medium` /
`high`), and the model version for a loan applicant.

**Domains:** `Lending`, `Risk Analytics`

This is Milestone 1: a deliberately minimal, complete path through all three subsystems —
training, a served model, and CI/CD — for a fixed 21-feature model trained on the Home
Credit Default Risk dataset. See [`CLAUDE.md`](CLAUDE.md) for the architecture, invariants,
and roadmap; see the design spec and implementation plan under `docs/superpowers/` for the
full reasoning.

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
creditboost-train --data data/application_train.csv --provenance production
```

This writes `models/model.json` and `models/model_meta.json`, refusing to write anything
if validation ROC-AUC falls below the floor in `config.py`. A repo clone already ships with
a committed, production-trained artifact, so this step is only needed to retrain.

### Run

```bash
docker build -t creditboost:dev .
docker run -d --rm -p 8000:8000 creditboost:dev
./scripts/smoke.sh http://localhost:8000
```

Or run the API directly against the committed artifact, without Docker:

```bash
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
validation metrics, xgboost version, and provenance.

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
{"probability": 0.1582, "risk_band": "medium", "model_version": "0.1.0"}
```

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
  train.py                # training CLI (creditboost-train)
  serve/
    app.py                 # FastAPI app: /health, /metadata, /predict
    deps.py                 # process-wide loaded-model state
    logging_config.py        # structured JSON logging for the creditboost logger tree
tests/                        # pytest, one module per src file, plus a synthetic fixture
models/                        # committed artifact: model.json + model_meta.json
data/                           # gitignored; the real Kaggle CSV goes here
docs/superpowers/                # design spec and implementation plan
.github/workflows/ci.yml          # lint, test, build, smoke-test, publish to GHCR
```

## Development

```bash
ruff check . && ruff format --check .   # CI checks formatting; run before committing
mypy src/
```

CI never downloads from Kaggle — it runs only against a synthetic fixture (`tests/fixtures/sample.csv`)
and the committed artifact, so it's hermetic and credential-free. Training against the real
dataset is a manual, local step.
