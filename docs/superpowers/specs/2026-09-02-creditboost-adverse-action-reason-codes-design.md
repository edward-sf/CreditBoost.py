# CreditBoost — Adverse Action Reason Codes

**Date:** 2026-09-02
**Status:** Approved design, ready for implementation planning
**Milestone:** 3 (Phase 3)
**Amends:** the feature set fixed in `2026-08-30-creditboost-thin-slice-design.md`
(21 features including `NAME_FAMILY_STATUS`)

## Purpose

Tell an applicant *why*.

The service scores an applicant and returns a probability, a risk band, and a model
version. It never says what drove the number. Under ECOA / Regulation B §1002.9, a
creditor taking adverse action must disclose the **specific principal reasons** for it —
so a scoring service that cannot name its reasons cannot support the decision it is used
to make. This milestone adds them.

Building the explanation also forced a question the earlier milestones never had to ask:
if the service must disclose the actual principal reasons, what happens when one of them
is a prohibited basis? That question surfaced a real defect in the Milestone 1 feature
set, which this milestone also fixes.

## Scope

**In scope:** removal of `NAME_FAMILY_STATUS` as a model feature and the retrain and
release that follow; a named `PROTECTED_ATTRIBUTES` set with a structural guard; a
concept-grouped reason catalog; per-prediction reason codes computed from XGBoost's
`pred_contribs`; a new `reasons` field on `PredictResponse`.

**Out of scope, and unchanged:** the risk bands and their thresholds, the training
pipeline's structure, the artifact format and its metadata sidecar, the lockfile and
release machinery, the Docker build, CI, and the runtime dependency set — no new runtime
dependency is added.

**Out of scope and still unspecced:** disparate-impact measurement, SHAP explanations
beyond the built-in tree contributions, experiment tracking, the six auxiliary Home Credit
tables, batch prediction, authentication, deployment, prediction persistence, and
automated retraining.

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Attribution method | XGBoost `pred_contribs=True` | Exact tree SHAP for this model class, already in the runtime image; adding the `shap` package would violate the small-image constraint for no gain |
| Delivery | Always, in the `/predict` response | One round trip, one response shape; the caller decides when reasons matter, which is right because the caller — not this service — takes the adverse action |
| Reason form | Curated code plus plain-language text | An adverse action notice is prose for an applicant; a signed float is neither meaningful to them nor safe to publish |
| Granularity | Concepts, grouped then ranked | Three `EXT_SOURCE` columns are one idea; ranking per feature can fill all four slots with one cause stated four ways |
| Catalog location | `config.py` | Business policy that changes without retraining — the same argument that puts the band thresholds there |
| Missing data | Distinct wording per concept | "No external credit score on file" and "external credit score is low" are different statements, and only one is true of a thin-file borrower |
| Reason count | At most 4 | Reg B's commentary treats more than four principal reasons as unhelpful to the applicant |
| Direction | Positive contributions only | A feature that helped the applicant is not a reason for denial |
| `NAME_FAMILY_STATUS` | Dropped as a feature, retained as a field | Marital status is an enumerated prohibited basis; retaining the field preserves fair-lending measurement without permitting scoring |

## Architecture

### The prohibited-basis defect

ECOA, 15 U.S.C. §1691(a)(1), names the prohibited bases in a single sentence: *"race,
color, religion, national origin, sex or marital status, or age."* The codebase is
scrupulous about two of them — `CODE_GENDER` is banned outright and age enters only
through the `employed_to_age` ratio — and then uses `NAME_FAMILY_STATUS` as model feature
16, with `Widow` and `Separated` among its scoring levels.

Explainability is what makes this untenable rather than merely untidy. Reason codes
disclose the *actual* principal reasons. If marital status materially drives a score there
are two outcomes and both are unacceptable: disclose it, and the service states on the
record that a prohibited basis drove the decision; suppress it, and the notice is
untruthful, because the reasons disclosed are not the principal ones.

Regulation B does permit *collecting* marital status in limited circumstances, and
§1002.13 goes further — it **requires** collecting certain protected attributes on
dwelling-secured applications specifically so fair-lending monitoring is possible. Collect
for monitoring, quarantine from scoring, is therefore the regulator's own pattern rather
than an invention of this design. The precise boundary of permissible use is a question
for counsel; this design takes the conservative side of it.

The fix is to name the rule the codebase was already following unevenly:

```python
PROTECTED_ATTRIBUTES = ("CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS")
```

The invariant is **no protected attribute is itself a model feature** — deliberately not
"the transform never reads one," because the transform still reads `DAYS_BIRTH` to derive
`employed_to_age`. That carve-out stands on firmer ground than the marital-status feature
did: Regulation B gives age an explicit allowance within an empirically derived,
demonstrably and statistically sound credit scoring system, subject to the constraint that
an elderly applicant is never assigned a negative factor for age. Marital status has no
equivalent allowance.

The three members sit in three different states, and the invariant covers all of them
uniformly:

| Attribute | Accepted in request? | A model feature? |
|---|---|---|
| `CODE_GENDER` | Never | Never |
| `DAYS_BIRTH` | Yes, required | No — only via `employed_to_age` |
| `NAME_FAMILY_STATUS` | Yes, optional | No, as of this milestone |

`FEATURE_ORDER` goes from 21 entries to 20. `REQUEST_FIELDS` stays at 19: the field is
still accepted, so a caller sending it is not broken, and live traffic retains the
attribute that later disparate-impact work will need. Dropping the field would have been
the one genuinely irreversible change in this milestone — a model feature can be restored
by retraining against a dataset that still has the column, but an attribute a service
refused to accept cannot be collected retroactively.

### How a reason is produced

`booster.predict(dmatrix, pred_contribs=True)` returns one row of `n_features + 1` values:
a signed contribution per feature in the model's margin (log-odds) space, plus the bias in
the final column. Positive contributions push the applicant toward higher default
probability, which is to say toward adverse action. Only sign and rank are used; no
number reaches the response.

The pipeline, in order:

1. Map each feature to its concept and **sum contributions within the concept.**
2. Discard concepts whose total is not positive.
3. Rank the remainder by total, descending.
4. Take at most `MAX_REASONS` (4).
5. For each, choose wording: the **absent** text when every feature in that concept was
   missing after the transform, otherwise the **unfavourable** text.

Missingness is read from the **transformed** frame, not the raw request. This is both the
correct notion — it is what the model actually saw — and a free correctness win: the
`DAYS_EMPLOYED == 365243` sentinel is scrubbed to NaN before the transform returns, so an
unemployed applicant reads as *"no employment history on record"* rather than *"length of
current employment is short."*

The all-missing rule was chosen over following the largest single contributor's state.
Following the dominant contributor is more precise and more surprising: an applicant with
two of three bureau scores on file could be told no score is on file because the absent
one happened to dominate. Simplicity and truthfulness agree here.

### Module boundary

`src/creditboost/reasons.py` is pure. It imports `config` and the standard library — no
xgboost, no pandas — and exposes:

```python
def principal_reasons(
    contributions: Mapping[str, float],
    missing: Mapping[str, bool],
) -> list[Reason]: ...
```

Keeping the array handling at the edge, in `serve/app.py`, means the ranking, grouping and
wording logic is testable with plain dictionaries and no model at all. The module joins
the `[tool.importlinter]` contract's `source_modules`, so it can never reach training code.

### What this deliberately does not touch

The risk bands, their thresholds, and `banding.py`. The artifact format, `ModelMetadata`,
and the skew gate in `artifact.load()`. The lockfile, the release script, and the Docker
build — a retrain flows through the Milestone 2 machinery unchanged, which is the first
real exercise of that path. Logging: reason codes are not financial fields, but they are
applicant disclosures, and the existing invariant's spirit is that nothing about the
applicant accumulates in log aggregation, so nothing new is logged.

## Components

### `src/creditboost/config.py`

Gains `PROTECTED_ATTRIBUTES`, `REASON_CONCEPTS`, `REASON_TEXT`, and `MAX_REASONS = 4`.
Loses `NAME_FAMILY_STATUS` from `FEATURE_ORDER` and from `CATEGORICAL_LEVELS`; keeps it in
`REQUEST_FIELDS`.

`REASON_CONCEPTS` maps a concept id to the features expressing it. It must partition
`FEATURE_ORDER` exactly:

| Concept | Features |
|---|---|
| `external_credit` | `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` |
| `loan_size` | `AMT_CREDIT`, `AMT_GOODS_PRICE`, `credit_to_income` |
| `repayment_burden` | `AMT_ANNUITY`, `annuity_to_income` |
| `employment` | `DAYS_EMPLOYED`, `employed_to_age` |
| `employment_profile` | `NAME_INCOME_TYPE`, `OCCUPATION_TYPE` |
| `assets` | `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`, `NAME_HOUSING_TYPE` |
| `household` | `CNT_CHILDREN`, `CNT_FAM_MEMBERS` |
| `income` | `AMT_INCOME_TOTAL` |
| `education` | `NAME_EDUCATION_TYPE` |
| `product` | `NAME_CONTRACT_TYPE` |

Twenty features, ten concepts, no overlaps. Grouping closes the age-disclosure trap as a
side effect: `employed_to_age` can never surface on its own, because it is only ever
reported as `employment`.

`REASON_TEXT` maps a concept id to its code and one or two texts. All ten concepts need an
entry, enforced by test. Three are given here as the pattern; the remaining seven are
authored in the catalog task of the implementation plan, under one binding constraint —
**no reason text may name or imply age, sex, or marital status** — with the wording read
back against that constraint before the task closes:

| Concept | Code | Unfavourable | Absent |
|---|---|---|---|
| `external_credit` | `EXTERNAL_CREDIT` | Credit scores from external bureaus are low | No external credit score on file |
| `employment` | `EMPLOYMENT_TENURE` | Length of current employment is short | No employment history on record |
| `loan_size` | `LOAN_SIZE` | Loan amount is high relative to income | *(unreachable)* |

Concepts built only from required request fields — `income`, `loan_size` — can never be
fully absent, so their absent text is unreachable. The catalog makes it optional, and a
test asserts the two definitions agree rather than letting dead text drift.

### `src/creditboost/reasons.py`

New, pure, described under Module boundary above. Defines `principal_reasons` and the
concept-summing and wording-selection logic. `Reason` itself lives in `schema.py` with the
other API contracts, and `reasons.py` imports it — the type is part of the response
contract, so it belongs where the contract is defined.

### `src/creditboost/schema.py`

`PredictRequest` is unchanged — `NAME_FAMILY_STATUS` remains an accepted optional field.
`PredictResponse` gains `reasons: list[Reason]`, ordered most significant first, at most
four, possibly empty.

The field is named `reasons` rather than `principal_reasons`. "Principal reasons" is
Regulation B's term of art and attaches only once a creditor takes adverse action; this
service scores and does not decide. The field's documentation states the relationship
precisely: these are the factors increasing the applicant's risk, ordered by contribution,
and they are the specific principal reasons §1002.9 requires **if** the caller denies on
this score.

### `src/creditboost/serve/app.py`

The `/predict` handler keeps its existing `booster.predict(...)` call for the probability
and adds a second call with `pred_contribs=True` for the contributions. Two calls, chosen
deliberately: deriving the probability from the contribution sum instead
(`sigmoid(sum(contributions))`) is one call rather than two, but it makes the service's
primary output a byproduct of the explanation path, where a numeric drift would be subtle
and would land on the number that matters most. The single-call form is a measured
optimisation available later if latency demands it; the margin-reconstruction test below
is what would license it.

The handler then builds the contribution and missingness mappings and calls
`principal_reasons`. Logging is unchanged.

### `src/creditboost/features.py`

Unchanged in structure. `NAME_FAMILY_STATUS` disappears from its output only because it
disappears from `config.CATEGORICAL_LEVELS` — no code change beyond that.

### The retrain

`creditboost-train --data data/application_train.csv --provenance production` against the
20-feature transform, then `./scripts/release-model.sh 0.2.0`, then a `MODEL_VERSION` bump
to `0.2.0` and the lockfile commit. `creditboost-artifact verify` enforces that the
lockfile and `MODEL_VERSION` move together, so this cannot half-land.

The dataset is not present on the development machine and must be re-downloaded from
Kaggle first. The AUC floor in `config.py` remains a gate: if the retrain fails it, that is
a finding to investigate, not a number to lower.

## Error Handling

| Failure | Behaviour |
|---|---|
| `pred_contribs` array is the wrong width | Raise at request time rather than emit reasons derived from a misread array; a silently misindexed array yields entirely plausible, entirely wrong reasons |
| A feature is absent from `REASON_CONCEPTS` | Cannot occur at runtime — the partition test fails the build first |
| No concept has a positive total | Return an empty `reasons` list; a valid state, not an error |
| A concept has no applicable text | Cannot occur at runtime — the catalog consistency test fails the build first |
| Fewer than four positive concepts | Return what exists; the cap is a maximum, not a quota |

The first row is the one that matters. The remainder are made unreachable by tests rather
than handled defensively at runtime, following the codebase's existing preference for
structural guarantees over runtime checks.

## Invariant Ledger

### Gained

- **No protected attribute is a model feature.** `PROTECTED_ATTRIBUTES` and
  `FEATURE_ORDER` are disjoint, enforced by test. Supersedes the narrower rule that named
  only `CODE_GENDER` and raw `DAYS_BIRTH`.
- **`REASON_CONCEPTS` partitions `FEATURE_ORDER` exactly.** Every feature belongs to
  exactly one concept; no orphans, no unknown names. This is the invariant that rots
  silently when the feature list changes, so it is the one most worth a test.
- **A reason is never a feature that helped the applicant.** Only positive contributions
  are eligible.
- **At most four reasons**, ordered by concept total.
- **Reason wording is truthful about missingness.** Absent text is used only when every
  contributing feature in that concept was missing after the transform.

### Amended

- `FEATURE_ORDER` has exactly **20** entries, not 21. `REQUEST_FIELDS` stays at 19.
- The ECOA invariant now names **marital status** alongside sex and age, and is expressed
  through `PROTECTED_ATTRIBUTES` rather than as prose naming two fields.

### Unchanged

- Missing values are never imputed.
- No `scale_pos_weight`.
- Risk-band thresholds live in `config.py`, not the artifact.
- No applicant financial field is ever logged, and nothing new is logged here.
- CI never downloads from Kaggle; training remains a manual, local, credentialed step.
- The one-way dependency rule, now covering `reasons.py`.
- `models/model.lock.json` and `config.MODEL_VERSION` move in the same commit.

## Accepted Risks

**The retrain may lose accuracy.** `NAME_FAMILY_STATUS` is expected to be weak next to the
`EXT_SOURCE` features, but no number is asserted here — the retrain measures it against
the 0.7533 baseline, and `model-v0.1.0` stays downloadable, so reverting is a three-line
lockfile diff. Accepted because a prohibited-basis feature is not a trade to be made
against AUC.

**Reason codes expose model behaviour.** Ranked concepts are a coarse, deliberately
non-numeric view, and an applicant's right to know why they were denied is the point of
the exercise. Publishing signed contributions would be a different and larger exposure;
this design does not.

**`NAME_INCOME_TYPE` carries proxy levels.** `Maternity leave` is a sex proxy and
`Pensioner` an age proxy inside an otherwise legitimate feature, so an
`employment_profile` reason can implicate a protected characteristic through the back
door. Unlike marital status this is a level rather than a whole prohibited-basis feature,
and dropping employment type wholesale would cost real signal. Recorded here as a named
open question for the disparate-impact milestone rather than resolved by guesswork now.
`CNT_CHILDREN` carries a lighter version of the same concern: Regulation B forbids
assumptions about income interruption from childbearing, and dependent count is adjacent
to that.

**Latency rises on every prediction.** `pred_contribs` does more work than a plain
`predict`, and reasons are returned on every call rather than only adverse ones. Expected
to be small for a model of this size, but measured during implementation rather than
assumed; the `latency_ms` log field already provides the instrument.

## Testing

### `tests/test_reasons.py`

The pure module, driven by plain dictionaries with no model involved: concept summing,
positive-only filtering, ranking by concept total rather than per-feature maximum, the
four-reason cap, fewer-than-four inputs, an empty result when nothing is positive, and
absent-versus-unfavourable wording selection including the partially-missing case.

### `tests/test_config.py` (extended)

The structural invariants: `PROTECTED_ATTRIBUTES` disjoint from `FEATURE_ORDER`;
`REASON_CONCEPTS` partitions `FEATURE_ORDER` exactly; every concept has an entry in
`REASON_TEXT`; concepts composed only of required fields have no unreachable absent text;
`FEATURE_ORDER` has 20 entries and `REQUEST_FIELDS` 19.

### `tests/test_api.py` (extended)

End to end: a response carries at most four reasons drawn from the catalog; the same
request yields the same reasons; an applicant with the `365243` sentinel is told *no
employment history on record*; `NAME_FAMILY_STATUS` is still accepted and provably absent
from the feature matrix.

### The array-shape test

Contributions plus the bias column reconstruct the model's margin. An off-by-one in a
`pred_contribs` row produces reasons that are entirely plausible and entirely wrong — a
failure with no natural symptom, and therefore the one that most needs an explicit test.

### Removed

Nothing. No existing test asserts `NAME_FAMILY_STATUS` is a feature; the count assertions
in `test_config.py` are amended rather than deleted.

## Sequencing Constraint

The feature removal, retrain, and release must land **before** the reason catalog, because
the concept map must partition the final feature list. Writing the catalog first would
mean writing it twice.

## Success Criteria

1. `FEATURE_ORDER` has 20 entries and does not contain any member of
   `PROTECTED_ATTRIBUTES`; `REQUEST_FIELDS` still has 19.
2. `REASON_CONCEPTS` partitions `FEATURE_ORDER` exactly, enforced by a failing-first test.
3. A retrained `model-v0.2.0` is released and pinned, `MODEL_VERSION` is `0.2.0`, and
   `creditboost-artifact verify` passes against it inside the Docker build.
4. The retrain clears the AUC floor, with the achieved figure recorded against the 0.7533
   baseline.
5. `POST /predict` returns at most four reasons, ordered by concept total, each a code and
   plain-language text drawn from the catalog.
6. A concept whose contribution total is not positive never appears in a response.
7. An applicant with no external credit score on file is told exactly that, and never that
   their score is low.
8. An applicant with the `DAYS_EMPLOYED` sentinel is told no employment history is on
   record, and never that their employment is short.
9. No reason names or implies age, sex, or marital status.
10. The runtime image gains no new dependency, and `ruff`, `mypy`, `lint-imports` and the
    full suite are clean.
