# CreditBoost — Disparate Impact Measurement

**Date:** 2026-09-02
**Status:** Approved design, ready for implementation planning
**Milestone:** 4 (Phase 4)
**Completes:** the credit-domain pairing begun in
`2026-09-02-creditboost-adverse-action-reason-codes-design.md`

## Purpose

Measure whether the model treats protected groups differently, and refuse to ship one that does.

Milestone 3 established that no protected attribute is a model feature. That is a statement
about inputs, and it is not the same as fairness: facially neutral features act as proxies,
which is why disparate impact is a distinct legal doctrine from disparate treatment. A model
can satisfy every input rule in `PROTECTED_ATTRIBUTES` and still deny one group at several
times the rate of another.

Nothing in the codebase has ever checked. This milestone measures it on every training run,
records the result in the artifact, and gates on it — so a model whose outcomes diverge
across groups cannot be written, let alone released.

## Scope

**In scope:** an adverse impact ratio computed per protected attribute on the validation
split; a training-time gate; a `FairnessReport` stamped into `ModelMetadata`; the fixture and
loader changes needed to make the attributes available; a retrain and release carrying the
new metadata.

**Out of scope, and unchanged:** the feature set, the transform, the model's
hyperparameters, the reason catalog, the risk bands and their thresholds, the serving
request and response schemas, the lockfile and release machinery, and the Docker build.

**Out of scope and still unspecced:** remediation of any disparity found (reweighting,
threshold adjustment by group, adversarial debiasing), fairness of the *reason codes*
themselves across groups, intersectional analysis, deployment, prediction persistence,
experiment tracking, the six auxiliary Home Credit tables, batch prediction, and
authentication.

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Where it runs | Inside `creditboost-train`, on the validation split | The protected attributes are already in the frame at exactly the moment validation predictions are made; no data plumbing is required |
| Enforcement | A gate — training refuses to write a failing model | Matches the AUC floor and the codebase's preference for structural guarantees over discipline |
| Metric | Adverse impact ratio, four-fifths rule, threshold 0.80 | It is what an examiner computes, which makes the number legible to the audience that matters |
| Adverse outcome | `band != "low"` — not auto-approved | AIR on approval cannot discriminate at a 97% approval rate; see Architecture. Reg B defines adverse action to include credit granted on substantially different terms, so this is grounded rather than a convenience |
| Policy dependence | Record `band_low_max` beside the ratio; do not enforce | Preserves the Milestone 1 invariant that band thresholds change without retraining; the figure becomes self-describing instead |
| Small groups | Excluded below `MIN_FAIRNESS_GROUP_SIZE` (100), and recorded as unmeasured | A rate estimated from twenty rows is too noisy to gate a release on |
| Unmeasured attributes | Recorded with a reason, and never treated as passing | "Not measured" reading as "passed" is the failure mode that would make the whole report worthless |
| Override | None | Consistent with the AUC floor: a model that cannot pass is a conversation about the approach, not a constant to edit |
| Metadata field | Required, not optional | An optional field makes "every shipped model has measured fairness" a hope rather than a guarantee |

## Architecture

### Why the conventional four-fifths test fails here

The four-fifths rule comes from UGESP, written for employment selection, where selection
rates are moderate. It compares the *favourable* outcome rate between groups:

```
AIR = min favourable_rate / max favourable_rate      pass if >= 0.80
```

The shipped model approves 97.3% of the validation set. At that rate the arithmetic
compresses every ratio toward 1.0, and the test stops being able to detect anything. Measured
against the shipped `model-v0.2.0` with adverse defined as the `high` band:

| Attribute | Adverse-rate disparity | AIR | Verdict |
|---|---|---|---|
| Age (62+ vs under 62) | **9.19×** | 0.974 | passes |
| Marital status | 4.10× | 0.971 | passes |
| Sex | 1.97× | 0.980 | passes |

An applicant under 62 is denied at more than nine times the rate of one aged 62 or over, and
the four-fifths test returns 0.974. A gate built on that definition would pass essentially any
model — which is worse than having no gate, because it manufactures documented assurance in
precisely the artifact a compliance reader would trust.

### The definition that restores sensitivity

Defining the adverse outcome as **not banded low** — the applicant is not auto-approved —
raises the adverse rate to 10–33% and makes the ratio informative again, while keeping both
the literal four-fifths rule and its standard 0.80 threshold:

| Attribute | Adverse rate range | AIR | Margin |
|---|---|---|---|
| Sex | 0.222 – 0.325 | 0.868 | comfortable |
| Marital status | 0.165 – 0.317 | 0.818 | comfortable |
| Age (62+ vs under 62) | 0.099 – 0.271 | **0.810** | one point |

This is grounded rather than convenient. Regulation B defines adverse action to include "a
refusal to grant credit in substantially the amount or on substantially the terms
requested" — unfavourable terms are adverse action, not only refusal. A `medium` band is a
materially worse outcome than `low`.

The age margin of 0.010 is a property of the design, not a flaw. A gate that cannot fail
proves nothing, and this one plausibly will on some future retrain. The answer then is
investigation, not adjustment of the constant.

### What this does and does not claim

An adverse impact ratio below 0.80 establishes a *prima facie* case that a creditor may
rebut with business necessity. It is evidence, not proof, and this milestone's gate is
therefore a deliberately conservative engineering control rather than a legal conclusion.
Equally, the disparities above are observed outcome differences: some part of the age gap is
genuine repayment behaviour, and the model uses no age feature at all — any age effect
arrives through proxies, which is exactly what disparate-impact analysis exists to surface.

### Policy dependence, and why it is recorded rather than enforced

The ratio is measured at `RISK_BAND_LOW_MAX`. Milestone 1 established that band thresholds
are business policy living in `config.py` precisely so they can change without retraining.
Those two facts collide: edit the threshold and the stored ratio silently describes a policy
no longer in force.

Enforcing agreement — having `creditboost-artifact verify` refuse when the config's bands
differ from those recorded, in the manner of the feature-order skew gate — would resolve it
structurally but revoke the Milestone 1 invariant, since bands could then no longer move
without a retrain. The measurement therefore records `band_low_max` alongside the ratio, so
the figure is self-describing and cannot be misread, and the resulting rule is a documented
discipline: **when band policy moves, fairness must be re-measured.** It is a deliberate trade: enforcement
here would cost an invariant that matters more.

### Module boundary

`src/creditboost/fairness.py` imports `config` and `banding` only — never `data` or `train`,
so it joins the `[tool.importlinter]` contract like `reasons.py`. It takes a frame and an
array of probabilities and returns a `FairnessReport`; the gate decision is a separate
predicate over that report. Keeping computation and enforcement apart means the gate can be
tested against constructed reports without training anything.

## Components

### `src/creditboost/fairness.py`

```python
def evaluate(
    frame: pd.DataFrame,
    probabilities: Sequence[float],
    min_group_size: int = config.MIN_FAIRNESS_GROUP_SIZE,
) -> FairnessReport: ...

def failing_attributes(report: FairnessReport) -> list[AttributeFairness]: ...
```

`min_group_size` is a parameter rather than read directly from config so unit tests and the
fixture integration test can lower it; `train.py` always passes the config value.
`failing_attributes` returns those whose ratio is below `config.MIN_ADVERSE_IMPACT_RATIO`;
an unmeasured attribute is never returned, because it has established nothing either way.

Age is bucketed at the ECOA line — 62 and over versus under 62 — rather than by quantile,
because that is the boundary the statute draws.

### `src/creditboost/schema.py`

Gains `GroupRate`, `AttributeFairness`, `FairnessReport`, and a required `fairness` field on
`ModelMetadata`:

```python
class GroupRate(BaseModel):
    group: str
    adverse_rate: float
    n: int

class AttributeFairness(BaseModel):
    attribute: str
    adverse_impact_ratio: float | None = None    # None means not measured
    unmeasured_reason: str | None = None
    groups: list[GroupRate]

class FairnessReport(BaseModel):
    adverse_definition: str        # "band != low"
    band_low_max: float            # the policy the measurement was taken under
    min_group_size: int
    attributes: list[AttributeFairness]
```

Exactly one of `adverse_impact_ratio` and `unmeasured_reason` is set on any attribute.

Only `band_low_max` is recorded. The adverse definition depends solely on the low boundary,
and recording `band_medium_max` would imply a relevance it does not have.

`/metadata` returns `metadata.model_dump()`, so the report becomes publicly visible with no
endpoint change. That is intended: a service that gates on fairness should be willing to
publish what it measured. The report contains aggregate group rates only — no applicant data.

### `src/creditboost/config.py`

Gains `FAIRNESS_ATTRIBUTES = ("CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS")`,
`MIN_FAIRNESS_GROUP_SIZE = 100`, `MIN_ADVERSE_IMPACT_RATIO = 0.80`, and
`ECOA_PROTECTED_AGE = 62`. `MODEL_VERSION` becomes `0.3.0`.

`FAIRNESS_ATTRIBUTES` overlaps `PROTECTED_ATTRIBUTES` but is not the same list and must not be
merged with it: `PROTECTED_ATTRIBUTES` states what may never be a feature, while
`FAIRNESS_ATTRIBUTES` states what must be present in training data to measure outcomes. One
is a prohibition, the other a requirement.

### `src/creditboost/data.py`

`load_training_frame` adds `FAIRNESS_ATTRIBUTES` to its required-column check. Training
cannot gate on fairness without the attributes, so their absence is an error rather than a
silent skip. `CODE_GENDER` thereby becomes required in training data while remaining
something the service never accepts from a caller.

### `src/creditboost/train.py`

After computing metrics and before writing anything, evaluates fairness on the validation
frame and refuses on failure — the same position and shape as the AUC floor. The report is
passed into `ModelMetadata`.

### `tests/fixtures/generate_fixture.py`

Gains a third column category. The generator currently emits model features and
monitoring-only request fields; it must now also emit analysis-only columns that are never
accepted from callers, of which `CODE_GENDER` is the first. Its final selection becomes
`REQUEST_FIELDS + (FAIRNESS_ATTRIBUTES - REQUEST_FIELDS) + TARGET`.

The new draw is taken after the target computation so that no existing draw shifts position
in the random stream. This must be **verified** against the previous fixture rather than
asserted: the same claim was made during Milestone 3 and was wrong, because removing an entry
from `CATEGORICAL_LEVELS` had already shortened the stream earlier.

### The retrain

Nothing that affects training changes in this milestone — same features, same
hyperparameters, same seeded split — so the retrain is in substance a metadata refresh, and
the ratios measured against `model-v0.2.0` should reproduce. That makes the gate's outcome
predictable in advance rather than a gamble: age at 0.810 is expected to clear 0.80 by the
same one-point margin.

`ModelMetadata.fairness` is required, so `model-v0.2.0` will no longer load and
`creditboost-artifact verify` will reject it inside the Docker build. This reopens the
build-red window Milestone 3 had, and it closes the same way: retrain, `release-model.sh
0.3.0`, commit the lockfile with the `MODEL_VERSION` bump.

## Error Handling

| Failure | Behaviour |
|---|---|
| An attribute's AIR is below 0.80 | `creditboost-train` writes nothing and exits non-zero, naming the attribute, its ratio, the threshold, and every group's adverse rate and size |
| Fewer than two groups meet `min_group_size` | The attribute is recorded unmeasured with a reason; it never satisfies the gate and never fails it |
| A `FAIRNESS_ATTRIBUTES` column is missing from the data | `MissingColumnsError` from `load_training_frame`, naming the column |
| One group is entirely adverse (favourable rate 0) | The ratio is `min/max`, so the numerator is 0 and the ratio is 0.0 — measured, and it fails the gate, correctly |
| *Every* group is entirely adverse (max favourable rate 0) | The only genuine division by zero. Recorded unmeasured, with the reason that no applicant in any group received the favourable outcome — a degenerate model the gate cannot speak to |
| Both `adverse_impact_ratio` and `unmeasured_reason` set, or neither | Cannot occur at runtime; a test fails the build first |

## Invariant Ledger

### Gained

- **Every shipped model has measured disparate impact.** `ModelMetadata.fairness` is
  required, so an artifact without it cannot be loaded, served, or built into an image.
- **A model whose adverse impact ratio falls below 0.80 on any measured attribute cannot be
  written.** No override flag exists.
- **An unmeasured attribute never counts as passing.** Exactly one of
  `adverse_impact_ratio` and `unmeasured_reason` is set.
- **The adverse outcome is `band != "low"`**, and the ratio is `min/max` over favourable
  rates. Inverting it would make a failing model read as passing.
- **`FAIRNESS_ATTRIBUTES` are required in training data**, including `CODE_GENDER`, which
  the service still never accepts from a caller.

### Amended

- `MODEL_VERSION` becomes `0.3.0`, and the artifact metadata schema gains a required field.
- The training data contract widens: `load_training_frame` requires the fairness attributes.

### Unchanged

- No member of `PROTECTED_ATTRIBUTES` is ever a model feature. This milestone measures
  outcomes; it does not add an input.
- Risk-band thresholds remain business policy in `config.py`, changing without retraining.
- Missing values are never imputed; no `scale_pos_weight`.
- No applicant financial field is ever logged, and nothing new is logged here.
- CI never downloads from Kaggle; training remains manual, local and credentialed.
- The one-way dependency rule, now covering `fairness.py`.

## Accepted Risks

**The age margin is one point.** 0.810 against a 0.80 floor means a future retrain could
trip the gate on noise as easily as on a real regression. Accepted deliberately: a gate with
a comfortable margin everywhere would be one that never fires. The response to a failure is
to investigate the model, not to widen the threshold.

**The policy-dependence rule is documentation, not a test.** Changing `RISK_BAND_LOW_MAX`
makes a stored ratio stale, and nothing detects it. Accepted to preserve the Milestone 1
invariant that band thresholds change without retraining; the recorded `band_low_max` at
least makes the staleness discoverable by inspection.

**Fixture-scale measurement is thin.** At 200 rows with a minimum group size of 100, marital
status has roughly forty rows per level and will be recorded unmeasured. The integration test
therefore proves the plumbing and the skip path; the gate's behaviour is proven by unit tests
over constructed reports.

**A single validation split is a single estimate.** The ratios carry no confidence interval,
so a value near the threshold is not distinguishable from noise by the gate alone. Recording
each group's `n` alongside its rate is what lets a reader judge that for themselves.

## Testing

### `tests/test_fairness.py`

Constructed frames, no model:

- The ratio is `min/max` over favourable rates, never `max/min`. Inverted, a model at 0.81
  reads as 1.23 and every model passes — silent and total, so this is the load-bearing test.
- Adverse means `band != "low"`: an applicant banded `medium` counts as adverse. A test
  exercising only `high` would pass under the insensitive definition this design rejects.
- An exact ratio against a hand-computed disparity.
- Groups below `min_group_size` are excluded; an attribute left with fewer than two is
  recorded unmeasured with a reason, and `failing_attributes` never returns it.
- Age buckets at 62, not at a quantile.
- Boundary: exactly 0.80 passes; just below fails.

### `tests/test_schema.py`

`FairnessReport` round-trips through JSON; an attribute with both ratio and reason, or
neither, is rejected.

### `tests/test_train.py`

Training on the fixture produces metadata carrying a report whose `band_low_max` equals
`config.RISK_BAND_LOW_MAX` and whose `adverse_definition` is recorded. A rigged report below
the threshold causes `main` to write nothing and return non-zero.

### `tests/test_fixture.py`

The fixture carries every `FAIRNESS_ATTRIBUTES` column, and its pre-existing columns are
unchanged by the addition of `CODE_GENDER`.

## Sequencing Constraint

`fairness.py` and its schema must exist before `train.py` can call them, and the retrain must
follow the gate, since the gate determines whether the retrain can produce a shippable model.
The build is red from the moment `ModelMetadata.fairness` becomes required until
`model-v0.3.0` is released.

## Success Criteria

1. `creditboost-train` computes an adverse impact ratio per attribute in
   `FAIRNESS_ATTRIBUTES` and refuses to write a model failing any measured one.
2. The ratio is computed as `min/max` over favourable rates with adverse defined as
   `band != "low"`, proven by a test that fails when the direction is inverted.
3. An attribute with fewer than two qualifying groups is recorded unmeasured and never
   satisfies the gate.
4. `ModelMetadata.fairness` is required; an artifact lacking it cannot load.
5. The report records `band_low_max`, `adverse_definition`, `min_group_size`, and each
   group's rate and size.
6. `model-v0.3.0` is trained, released, pinned, and verified inside `docker build`, with
   `MODEL_VERSION` at `0.3.0`.
7. The retrained model clears both the AUC floor and the 0.80 ratio on every measured
   attribute, with the achieved figures recorded.
8. `/metadata` exposes the report with no endpoint change.
9. The fixture carries `CODE_GENDER`, and its pre-existing columns are verified unchanged.
10. No new runtime dependency; `ruff`, `mypy`, `lint-imports` and the full suite are clean.
