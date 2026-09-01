# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

**Milestone 1 — the thin end-to-end slice — is implemented, tested, and merged.** The
package lives under `src/creditboost/`, tests under `tests/`, and a production-trained
artifact is committed at `models/model.json` / `models/model_meta.json`.

- Design spec: `docs/superpowers/specs/2026-08-30-creditboost-thin-slice-design.md`
- Implementation plan: `docs/superpowers/plans/2026-08-30-creditboost-thin-slice.md`

Both describe the reasoning behind the decisions recorded in Architecture and Invariants
below — read them to understand *why* the code looks the way it does, not as a description
of work still to do. Any future work on this codebase (bug fixes, Milestone 2 and beyond)
should still follow the plan's task-by-task, test-first structure rather than improvising an
equivalent one.

## Roadmap

**Milestone 1 — thin end-to-end slice.** Done. A deliberately minimal path through all
three subsystems: Home Credit training data, a 21-feature XGBoost model, a committed
artifact baked into a container, a FastAPI `/predict`, and GitHub Actions publishing to
GHCR. All eleven tasks landed; the committed artifact in `models/` is `provenance:
"production"`, trained on the real Kaggle dataset (roc_auc 0.7533).

**Milestone 2 — model artifact storage.** Deliberately **not** specced or planned yet.
Milestone 1 commits the trained model to git, which is poor practice as a permanent
arrangement: git never forgets, so every retrain appends a full multi-megabyte copy that
cannot be removed without rewriting history, and it welds the model lifecycle to the code
lifecycle. It is accepted for Milestone 1 because a self-contained clone is what lets
Tasks 1–10 be verified with no credentials at all, and because adding a registry would mean
the first end-to-end run depends on two unproven things at once.

**Do not spec or begin Milestone 2 until Milestone 1 is proven.** When it starts, GitHub
Releases is the cheapest real upgrade — no new infrastructure, just a release plus a
build-arg download. Nothing else is scheduled: SHAP explanations, experiment tracking, the
six auxiliary Home Credit tables, batch prediction, authentication, and automated
retraining are all out of scope and unspecced.

## Architecture

The design's central constraint is that **train/serve skew is prevented structurally, not
by convention.** A single `features.transform()` in `src/creditboost/features.py` is
imported by both the training CLI and the serving app. Training stamps `FEATURE_ORDER` into
the artifact's metadata sidecar; serving compares it against the code's at startup and the
process exits non-zero on mismatch. A container that cannot score correctly must never
accept traffic.

Dependency direction is one-way: `serve/` imports `artifact`, `features`, `schema`,
`banding`, and `config` — never `data` or `train`. This is what keeps scikit-learn out of
the runtime image, and it is enforced by a subprocess test, not just a comment.

The serving app logs structured JSON to stdout via `serve/logging_config.py`, configuring
only the `creditboost` logger tree (never the root logger or uvicorn's own loggers, and
never `logging.basicConfig()`) so records aren't dropped under uvicorn and aren't double-
logged in uvicorn's plain-text format. `artifact.load()` is the train/serve skew gate: it
checks both the metadata sidecar's `feature_order` *and* the booster's own `feature_names`
baked into `model.json` against `config.FEATURE_ORDER`, because either one can drift
independently of the other.

The project targets Python 3.12 only (`requires-python = ">=3.12"` in `pyproject.toml`,
developed and tested on 3.12.4). `xgboost` is pulled in via environment markers: Linux
(the runtime image and CI) gets `xgboost-cpu`, which skips the ~291MB of CUDA libraries
that CPU-only inference never uses; other platforms (including macOS, which has no
`xgboost-cpu` wheel) get the standard `xgboost` package. On macOS, `xgboost` also needs the
OpenMP runtime, which isn't bundled: run `brew install libomp` before `pip install`, or
`import xgboost` fails.

## Invariants

These are easy to break silently and each has a test guarding it. Do not change one without
understanding why it exists.

- `FEATURE_ORDER` has exactly 21 entries; `REQUEST_FIELDS` has exactly 19.
- **`CODE_GENDER` must never appear** in any feature list, request schema, or transform, and
  **raw `DAYS_BIRTH` must never be a model feature.** Under ECOA / Regulation B, sex and age
  are prohibited bases for credit decisions in the US. Age enters only through the
  `employed_to_age` ratio.
- The `DAYS_EMPLOYED == 365243` sentinel is scrubbed to NaN **before** derived ratios are
  computed. Left raw it becomes a ~1000-year tenure that reads as a plausible value.
- **Missing values are never imputed.** NaN reaches XGBoost intact — for a thin-file
  borrower, a missing external credit score is itself signal. Every optional field in
  `PredictRequest` (e.g. `CNT_CHILDREN`) defaults to `None`, not a business default like
  `0`, so an omitted field degrades to NaN through the same path as an unknown one.
- **No `scale_pos_weight`.** It inflates probabilities away from the true base rate, which
  would decalibrate the score the service bands and reports a Brier score for. This is a
  deliberate deviation from the spec; see the plan's Task 7.
- Risk-band thresholds live in `config.py`, not in the artifact: they are business policy
  that changes without retraining.
- **No applicant financial field may ever be logged.** Logs carry request id, latency, model
  version, and risk band only.
- **CI never downloads from Kaggle.** It touches only the synthetic fixture and the committed
  artifact, which is what keeps it hermetic and credential-free. Training is a manual step.
- The test fixture is synthetic, never sampled from the real dataset — Kaggle's terms
  restrict redistribution.

## Metadata provenance

`model_meta.json` carries a `provenance` field, `"fixture"` or `"production"`. This exists
so Docker and CI can be built and booted against a fixture-trained artifact before the
credentialed training run happens. `/health` and `/metadata` both expose it, so a fixture
model cannot be mistaken for a real one in a running container. This is a deviation from the
spec, recorded in the plan.

## Commands

Requires Python 3.12. On macOS, install the OpenMP runtime first — `xgboost` cannot be
imported without it:

```bash
brew install libomp
```

```bash
pip install -e ".[train,dev]"        # editable install with training and dev extras
pytest                                # full suite
pytest -m "not slow"                  # skip the training smoke tests
pytest tests/test_features.py -v      # one module
pytest tests/test_features.py::test_sentinel_is_scrubbed_before_the_ratio_is_derived -v

ruff check . && ruff format --check . # CI checks formatting; run before committing
mypy src/

creditboost-train --data data/application_train.csv --provenance production
docker build -t creditboost:dev . && docker run -d --rm -p 8000:8000 creditboost:dev
./scripts/smoke.sh http://localhost:8000
```

The training dataset lives at `data/application_train.csv`, is gitignored, and is obtained
manually from Kaggle's Home Credit Default Risk competition.

## Working agreements

- Work runs spec → plan → implement. Creative or architectural work starts with the
  `superpowers:brainstorming` skill, not with code.
- Implementation is TDD, one task at a time, each ending in its own commit.
- The AUC floor in `config.py` is a gate, not a suggestion. If a training run fails it, do
  not lower the floor to force the model through — investigate the data first.
