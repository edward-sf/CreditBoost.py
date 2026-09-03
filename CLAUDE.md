# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

**Milestones 1 through 4 are implemented and tested.** The package lives under
`src/creditboost/`, tests under `tests/`, and the production-trained artifact lives in a
GitHub Release pinned by `models/model.lock.json` — it is not committed.

- Design spec: `docs/superpowers/specs/2026-08-30-creditboost-thin-slice-design.md`
- Implementation plan: `docs/superpowers/plans/2026-08-30-creditboost-thin-slice.md`

Milestones 2 through 4's specs and plans are linked in the Roadmap below, eight documents
in all. Each describes the reasoning behind the decisions recorded in Architecture and
Invariants below — read them to understand *why* the code looks the way it does, not as a
description of work still to do. Any future work on this codebase (bug fixes and beyond)
should still follow the plans' task-by-task, test-first structure rather than improvising
an equivalent one.

## Roadmap

**Milestone 1 — thin end-to-end slice.** Done. A deliberately minimal path through all
three subsystems: Home Credit training data, a 21-feature XGBoost model, a committed
artifact baked into a container, a FastAPI `/predict`, and GitHub Actions publishing to
GHCR. All eleven tasks landed; the artifact it produced is `provenance: "production"`,
trained on the real Kaggle dataset (roc_auc 0.7533). Milestone 2 moved that artifact out
of git, so the "committed artifact" described here is now a released one.

**Milestone 2 — model artifact storage.** Done. The trained model's bytes live in a
GitHub Release; `models/model.lock.json` pins the release tag and a sha256 per asset.
The Docker builder fetches and verifies them in a single `RUN`, so an image containing
an unverified, fixture-provenance, or ECOA-violating artifact cannot be built at all.
History was deliberately not rewritten — the problem was future retrains appending
copies, which deleting from `HEAD` solves.

- Design spec: `docs/superpowers/specs/2026-09-01-creditboost-model-artifact-storage-design.md`
- Implementation plan: `docs/superpowers/plans/2026-09-01-creditboost-model-artifact-storage.md`

**Nothing else was scheduled at the time** — Milestone 3 below was chosen afterwards.
Experiment tracking, the six auxiliary Home Credit tables, batch prediction, authentication,
deployment, prediction persistence, disparate-impact measurement, and automated retraining
remain out of scope and unspecced.

**Milestone 3 — adverse action reason codes.** Done. `/predict` returns up to four
plain-language reasons drawn from a curated catalog in `config.py`, ranked by summed
XGBoost `pred_contribs` contributions grouped into ten concepts. No new runtime dependency.

Writing the spec surfaced a defect in the Milestone 1 feature set: `NAME_FAMILY_STATUS` was
a model feature, and marital status is an enumerated ECOA prohibited basis. It was removed
and the model retrained as `model-v0.2.0`; the field is still accepted, under
`MONITORING_ONLY_FIELDS`, so later disparate-impact work can measure it. The removal cost
0.00023 AUC — 0.75307 against 0.75330.

- Design spec: `docs/superpowers/specs/2026-09-02-creditboost-adverse-action-reason-codes-design.md`
- Implementation plan: `docs/superpowers/plans/2026-09-02-creditboost-adverse-action-reason-codes.md`

**Known open question, deliberately deferred:** `NAME_INCOME_TYPE` carries `Maternity leave`
(a sex proxy) and `Pensioner` (an age proxy) as levels, so an `employment_profile` reason can
implicate a protected characteristic indirectly. Unlike marital status these are levels
rather than a whole prohibited-basis feature, and dropping employment type would cost real
signal. Milestone 5 measured them against the baseline min AIR of 0.8041:
`maternity-to-working` moves it by 0.0000 and `income-type-proxies-dropped` by +0.0002.
They remain in the catalog as `maternity-to-working` and `income-type-proxies-dropped` so
the finding is re-established on every search rather than remembered.

**Milestone 4 — disparate impact measurement.** Done. `creditboost-train` measures an
adverse impact ratio per protected attribute on the validation split, stamps the report into
the artifact, and refuses to write a model below 0.80. `model-v0.3.0` measures sex 0.868,
marital status 0.818, age 0.810.

The four-fifths rule had to be applied to the right outcome to mean anything. With adverse
defined as the `high` band, an applicant under 62 is denied at 9.19× the rate of one aged 62
or over and the test returns 0.974 — a gate built that way would pass any model and
manufacture documented assurance it had not established.

- Design spec: `docs/superpowers/specs/2026-09-02-creditboost-disparate-impact-design.md`
- Implementation plan: `docs/superpowers/plans/2026-09-02-creditboost-disparate-impact.md`

**Milestone 5 — less discriminatory alternative search.** Done. `creditboost-train --search`
ranks a catalog of model specifications at a matched approval rate on a split nested inside
the training data, applies a 0.01 AUC budget and a 0.01 noise guard, and stamps the whole
frontier into the artifact. `creditboost-artifact verify` refuses a production artifact that
carries no such record.

Measurement shaped every decision. Bootstrapping `model-v0.3.0` put the age ratio at 0.8100
with sd 0.0046 and 1% of resamples already below the floor — it passed within noise, not
comfortably. A design-time probe over sixteen small perturbations spanned 0.8058 to 0.8146,
entirely inside that noise, and showed why a narrow search of fine-tunings would not move
the needle. The shipped catalog also contains sixteen candidates, but the real-data frontier
is different in character: best AUC is `min-child-weight-50` at 0.7506 (admitting AUC ≥
0.7406), `external-scores-only` reaches min AIR 0.8595 but is *excluded by the AUC budget*
at a cost of 0.028, and `no-employment` sits inside the budget at 0.8087 but is *rejected
by the noise guard* — a mere +0.0046 over baseline. Both guards firing, for different
reasons, shows the rule discriminates rather than rubber-stamps. The band threshold, by
contrast, moves the ratio further than every model variant combined, which is exactly why
it is excluded as a search axis rather than exploited.

`model-v0.4.0` carries the identical fairness ratios as `model-v0.3.0` — sex 0.8684, age
0.8100, marital status 0.8179 — because ranking reads only the training split and the final
fit was unchanged; the model's sha256 digest did not budge. The milestone ships **no fairness
improvement**: age remains at 0.8100 against the 0.80 floor. What is new is that every
production model now carries recorded evidence it was searched, a negative result that
establishes business necessity. The one real finding: removing the external bureau scores
makes fairness markedly worse (0.8041 to 0.7023), locating the disparity in the
application-form features, not the bureau scores.

- Design spec: `docs/superpowers/specs/2026-09-02-creditboost-less-discriminatory-alternative-design.md`
- Implementation plan: `docs/superpowers/plans/2026-09-02-creditboost-less-discriminatory-alternative.md`

**Still open:** remediation that actually moves the ratio, intersectional analysis, fairness
of the reason codes across groups, deployment, prediction persistence, experiment tracking,
the six auxiliary Home Credit tables, batch prediction, authentication.

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

- `FEATURE_ORDER` has exactly 20 entries; `REQUEST_FIELDS` has exactly 19.
- **No member of `config.PROTECTED_ATTRIBUTES` is ever a model feature.** ECOA,
  15 U.S.C. §1691(a)(1), names the prohibited bases in one sentence: race, color, religion,
  national origin, sex or marital status, or age. `CODE_GENDER` is never accepted at all;
  `DAYS_BIRTH` and `NAME_FAMILY_STATUS` are accepted and never modelled. The rule is about
  features, not reads — the transform still reads `DAYS_BIRTH` to derive `employed_to_age`,
  which Regulation B allows in an empirically derived, statistically sound scoring system.
  Marital status has no equivalent allowance, which is why it is not a feature.
- **Every shipped model has measured disparate impact.** `ModelMetadata.fairness` is
  required, so an artifact without it cannot be loaded, served, or built into an image.
  `creditboost-train` computes an adverse impact ratio per attribute in
  `FAIRNESS_ATTRIBUTES` and refuses to write a model below
  `MIN_ADVERSE_IMPACT_RATIO` (0.80). There is no override flag.
- **The adverse outcome is `band != "low"`, and the ratio is `min/max` over favourable
  rates.** Not `band == "high"`: at a 97% approval rate the ratio on approvals compresses
  toward 1.0 and cannot discriminate — a 9.19× age disparity passes at 0.974. Reg B treats
  credit on substantially different terms as adverse action. Inverting the ratio would make
  every failing model read as passing; a test guards the direction.
- **"Not measured" never reads as "passed."** Exactly one of `adverse_impact_ratio` and
  `unmeasured_reason` is set, and `failing_attributes` never returns an unmeasured one.
- **`FAIRNESS_ATTRIBUTES` is a requirement, `PROTECTED_ATTRIBUTES` a prohibition.** They
  overlap and must not be merged: one says what must be present in training data to measure
  outcomes, the other what may never be a model feature. `CODE_GENDER` is in both — required
  in training data, never accepted from a caller.
- **When risk-band policy moves, fairness must be re-measured.** The ratio is measured at
  `RISK_BAND_LOW_MAX`, which is policy that changes without retraining, so the report records
  the threshold it was measured under. This one is documentation rather than a test:
  enforcing it would revoke the invariant that band thresholds change without retraining.
- **`MONITORING_ONLY_FIELDS` are accepted but never transformed.** Reg B §1002.13 requires
  collecting certain protected attributes precisely so fair-lending monitoring is possible.
  An attribute a service refuses to accept cannot be collected retroactively; a feature can
  always be restored by retraining.
- **`REASON_CONCEPTS` partitions `FEATURE_ORDER` exactly**, and **no reason text names or
  implies age, sex, or marital status.** Both have tests. A disclosure that reintroduces a
  protected basis in prose would undo the feature work it accompanies.
- **At most four reasons, and only from positive contributions.** A feature that helped the
  applicant is not a reason for denial.
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
- **Model releases are not meant to be deleted.** Deleting one breaks rebuilds of every
  commit pinned to it. It is recoverable rather than fatal — CI publishes every image to
  GHCR tagged by commit sha, so the image stays pullable and the artifact can be recovered
  with `docker cp` from it, with the lockfile's digest proving the recovered bytes are
  right — but the recovery is a chore, so don't.
- **The one-way dependency rule is a contract, not a convention.** `[tool.importlinter]` in
  `pyproject.toml` forbids `serve/`, `artifact_cli`, and the shared modules from reaching
  `data.py` or `train.py`; CI runs `lint-imports`. This is what keeps scikit-learn out of
  the runtime image and lets the artifact CLI import cleanly in the Docker builder, where
  scikit-learn is absent.
- The test fixture is synthetic, never sampled from the real dataset — Kaggle's terms
  restrict redistribution.
- **Every production model was selected by a recorded search.** `creditboost-artifact verify`
  requires `ModelMetadata.selection` when `provenance == "production"`, so an image
  containing an unsearched production artifact cannot be built.
- **Candidates are ranked at matched approval rate, never at a fixed threshold.** At a fixed
  threshold the ratio reports leniency, not fairness: a single-feature model reads 0.984 at
  the band threshold and 0.930 at a matched rate. A test guards it.
- **The band threshold is never a search axis.** It moves the ratio further than any model
  variant, and it is risk appetite rather than fairness. `CandidateSpec` has no field that
  can express one, and a test pins that field set.
- **Selection happens strictly inside the training split.** `search.rank` receives the
  training frame only, so the validation split the artifact reports on cannot participate.
- **The search's matched-threshold ratio never reaches the artifact.** `ModelMetadata.fairness`
  is always what `fairness.evaluate` produced at `RISK_BAND_LOW_MAX`.
- **There is one implementation of the ratio arithmetic.** `fairness.adverse_impact_ratios`
  serves both the gate and the search; the `min/max` direction has one place to be wrong.
- **A negative search result is recorded, not discarded.** An artifact that dropped it could
  not distinguish "looked and stayed" from "never looked".
- **A non-baseline winner is a code change, not an automatic one.** `--search` writes nothing
  and says what to change, because a winner's features, levels or parameters must be
  reflected in `config.py`, `features.py` and `train.PARAMS` or the skew gate describes a
  model that no longer exists.

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
lint-imports                          # enforce the one-way dependency rule

creditboost-search --data data/application_train.csv     # print the frontier, write nothing
creditboost-train --data data/application_train.csv --provenance production
creditboost-train --data data/application_train.csv --provenance production --search
./scripts/release-model.sh 0.2.0      # publish + rewrite models/model.lock.json
creditboost-artifact fetch            # pull the pinned release into models/
creditboost-artifact verify           # check what's on disk against the lockfile
docker build -t creditboost:dev . && docker run -d --rm -p 8000:8000 creditboost:dev
./scripts/smoke.sh http://localhost:8000
```

A fresh clone carries **no** `models/*.json`, so a local `uvicorn` run needs
`creditboost-artifact fetch` first. `pytest` does not — it trains its own fixture.

The training dataset lives at `data/application_train.csv`, is gitignored, and is obtained
manually from Kaggle's Home Credit Default Risk competition.

## Working agreements

- Work runs spec → plan → implement. Creative or architectural work starts with the
  `superpowers:brainstorming` skill, not with code.
- Implementation is TDD, one task at a time, each ending in its own commit.
- The AUC floor in `config.py` is a gate, not a suggestion. If a training run fails it, do
  not lower the floor to force the model through — investigate the data first.
