# CreditBoost — Less Discriminatory Alternative Search

**Date:** 2026-09-02
**Status:** Approved design, ready for implementation planning
**Milestone:** 5 (Phase 5)
**Follows:** `2026-09-02-creditboost-disparate-impact-design.md`

## Purpose

Milestone 4 measures disparate impact and refuses to ship a model that fails the four-fifths
rule. It leaves the codebase with exactly one response to a failing gate: stop. There is no
procedure for finding a model that does better, and the working agreements forbid the only
other move available — lowering the floor.

That gap is not hypothetical. Bootstrapping the shipped `model-v0.3.0` over its own
validation split puts the age ratio at 0.810 with a standard deviation of 0.0046 and a 95%
interval of 0.802 – 0.819. One per cent of resamples already fall below the floor; for
marital status it is 2.3%. **The shipped model does not comfortably pass the gate — it
passes within noise.**

This milestone gives the project a repeatable, recorded procedure for searching for a less
discriminatory alternative, and makes the search's result part of the artifact.

The choice of technique is not arbitrary. Disparate impact is a burden-shifting doctrine:
once a *prima facie* disparity is shown, a creditor may rebut with business necessity, and
that rebuttal fails if a less discriminatory alternative achieving comparable performance
exists. Searching for one is the obligation. A recorded search — including one that finds
nothing — is the evidence that the obligation was met.

## Scope

**In scope:** a candidate search over model specifications; a matched-approval-rate ranking
rule; a nested split that keeps selection out of the validation data; a `SearchReport`
stamped into `ModelMetadata` and required of production artifacts by
`creditboost-artifact verify`; a `creditboost-search` inspection command and a
`creditboost-train --search` path; a retrain and release carrying the new metadata.

**Out of scope, and unchanged:** the transform, the risk bands and their thresholds, the
reason catalog, the serving request and response schemas, `fairness.py`'s measurement and
its gate, the lockfile and release machinery, and the Docker build's existing checks.

**Out of scope and still unspecced:** intersectional analysis, fairness of the reason codes
across groups, remediation techniques that use a protected attribute during fitting (see
Rejected Alternatives), deployment, prediction persistence, experiment tracking, the six
auxiliary Home Credit tables, batch prediction, and authentication.

**Correction to Milestone 4.** That spec's out-of-scope list named "threshold adjustment by
group" as a candidate remediation. It should not have. Setting a different band cutoff by
sex, age, or marital status is not a remedy for disparate impact — it is disparate
treatment, the use of a prohibited basis in the decision itself, ECOA §1691(a)(1). It is
excluded here on the merits rather than deferred.

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Technique | Search over model specifications for a less discriminatory alternative | The one remediation that never lets a protected attribute touch the model, and the one the burden-shifting doctrine actually asks for |
| Where selection happens | Nested inside the training split | The validation split must never participate in selection, or the winner's reported ratio is optimistically biased |
| Ranking metric | Minimum AIR across attributes, at **matched approval rate** | At a fixed threshold the search selects leniency and calls it fairness; measured below |
| Band threshold as a search axis | Excluded, absolutely | It dominates every model effect, and tuning it for its fairness number is Milestone 4's gaming trap through a different door |
| Selection rule | Highest min-AIR within `MAX_AUC_SACRIFICE` (0.01) of the best candidate's AUC | Encodes "comparable performance" as a bounded, stated number rather than a judgment made per-release |
| Noise guard | The winner must beat baseline by `MIN_AIR_IMPROVEMENT` (0.01) or baseline is kept | Grounded in the measured sd of ≈0.005; without it the search churns the shipped model on noise |
| Negative results | Recorded, not discarded | A search that found nothing is the business-necessity evidence. Discarding it would leave the artifact unable to distinguish "searched and found nothing" from "never looked" |
| Metadata field | Optional in the schema, required by `verify` for `provenance: "production"` | Mirrors the existing provenance gate; fixture training stays unaffected while production artifacts cannot skip it |
| Reweighting | Rejected | Collides with the standing `scale_pos_weight` invariant on calibration grounds; see Rejected Alternatives |

## Evidence

Every figure below was measured against the shipped `model-v0.3.0` pipeline on the real
Home Credit data before this design was written. The baseline reproduces the artifact's
recorded ratios exactly — sex 0.868, age 0.810, marital status 0.818 — which confirms the
pipeline is deterministic and the measurements comparable.

### The measurement is precise enough to search on

300 bootstrap resamples of the validation split at a fixed model:

| Attribute | AIR | sd | 95% interval | P(AIR < 0.80) |
|---|---|---|---|---|
| `CODE_GENDER` | 0.8684 | 0.0045 | 0.859 – 0.878 | 0.0% |
| `DAYS_BIRTH` | 0.8100 | 0.0046 | 0.802 – 0.819 | 1.0% |
| `NAME_FAMILY_STATUS` | 0.8171 | 0.0086 | 0.801 – 0.833 | 2.3% |

A difference of 0.02 between candidates is signal; a difference of 0.005 is not. This is
what makes both the search and its noise guard quantitative rather than intuitive.

### Small perturbations are not a remediation lever

Sixteen candidates — feature ablations, `NAME_INCOME_TYPE` level collapses, and
regularization variants — spanned a minimum AIR of **0.8058 to 0.8146**, a range entirely
inside the noise band. Notably, the `NAME_INCOME_TYPE` proxy levels deferred since Milestone
3 move the ratio by 0.0001:

| Candidate | min AIR |
|---|---|
| baseline | 0.8095 |
| `Maternity leave` → `Working` | 0.8095 |
| both proxy levels → NaN | 0.8096 |
| drop `NAME_INCOME_TYPE` entirely | 0.8108 |

The `Maternity leave` / `Pensioner` question is therefore real as a *disclosure* matter and
close to irrelevant as a disparate-impact one. This milestone records that finding and does
not act on it.

### The band threshold dominates, which is why it is excluded

Re-banding the same predictions, no retraining:

| `RISK_BAND_LOW_MAX` | auto-approved | default rate among approved | min AIR |
|---|---|---|---|
| 0.08 | 66.2% | 3.89% | 0.7653 |
| **0.10 (shipped)** | **74.3%** | **4.40%** | **0.8100** |
| 0.12 | 80.2% | 4.96% | 0.8432 |
| 0.15 | 86.4% | 5.54% | 0.8764 |
| 0.30 | 97.3% | 7.25% | 0.9711 |

A two-point config edit buys more ratio than every model variant combined. The bottom row
also reproduces Milestone 4's rejected definition precisely: at 97.3% approval the age
ratio reads 0.974.

This is the single most dangerous finding in the milestone. The threshold is risk appetite —
the improvement is bought by extending credit to more people, and the default rate among
approved applicants rises with it. Choosing it for its fairness number would manufacture
documented assurance rather than establish it, which is the exact failure Milestone 4 was
written to avoid. **It is therefore never a search axis, and this is an invariant.**

### A real frontier exists, at a real price

Comparing candidates at matched approval rate:

| Model | AUC | min AIR |
|---|---|---|
| full model | 0.7531 | 0.8100 |
| `EXT_SOURCE_*` only | 0.7249 | 0.8664 |
| stump (depth 1) | 0.7054 | 0.8843 |
| money features only | 0.6293 | 0.8744 |
| **no `EXT_SOURCE_*`** | 0.6748 | **0.7220** |

Giving up 0.028 AUC buys +0.056 AIR — ten times the noise. And the last row is the
substantive finding about this dataset: **removing the external bureau scores makes fairness
markedly worse.** The disparity lives in the application-form features, not in the bureau
scores. There is genuinely something here to search for, and it is not where an intuition
about "opaque third-party scores" would look.

### Why ranking must use matched approval rate

The same probe at a *fixed* threshold rather than a matched rate:

| Model | approval rate at 0.10 | min AIR at 0.10 | min AIR at matched rate |
|---|---|---|---|
| `EXT_SOURCE_2` only | 95.5% | 0.9844 | 0.9301 |

At a fixed threshold this candidate looks like the fairest model in the field. It is not; it
is merely the most lenient, and the ratio is reporting its approval rate. A search ranking
on that number would reliably select whichever candidate approves most people. This is
Milestone 4's central insight — measure the wrong outcome and the metric stops
discriminating — reappearing one level up, and the design must answer it in the same way.

## Architecture

### The nested split

The obvious structure, a three-way train/select/confirm split, is rejected. It would halve
the validation split used for the shipped measurement, raising its noise from sd ≈ 0.005 to
≈ 0.007 and making Milestone 5's reported ratios not directly comparable with Milestone 4's.

Instead, selection nests strictly inside the training portion:

```
frame ──split(0.8/0.2)──> train (0.8)                    valid (0.2)  [untouched]
                            │                                 │
                            └─split(0.75/0.25)─> search-train, select
                                                        │
                                    candidates ranked ──┘
                                                        │
                            winner retrained on the full train (0.8)
                                                        │
                            gated and measured on valid (0.2) ────────┘
```

The validation split never participates in selection, so the winner's reported ratio carries
no selection bias, and the shipped numbers are measured exactly as they were in Milestone 4 —
same split, same size, same seed. `train.py`'s existing measurement and gate are untouched.

The cost is that candidates train on 0.6 of the data rather than 0.8, so their absolute AUCs
are slightly pessimistic. Only their *ranking* is consumed, and all candidates share the
handicap. The residual risk — that a candidate's rank could change at full data size — is
accepted and recorded below.

### Matched approval rate, and the wall between the two measurements

Within the search, the baseline candidate's approval rate on `select` at `RISK_BAND_LOW_MAX`
defines a target rate. Every other candidate's adverse threshold is the quantile of its own
predictions achieving that same rate. Candidates are then compared on equal terms, and a
candidate cannot win by being lenient.

**This matched threshold is used only for ranking and never leaves the search.** The winner's
shipped `FairnessReport` is produced by the existing `fairness.evaluate` at the real
`RISK_BAND_LOW_MAX`, unchanged. The search's internal number and the artifact's reported
number are computed differently, on different data, for different purposes, and must never
be conflated — a test enforces that the reported report is the one `fairness.evaluate`
produced.

### The candidate space

A `CandidateSpec` varies three things, none of which is a protected attribute:

- **feature drops** — columns removed from the transform's output;
- **level collapses** — a mapping applied to a categorical column before transformation,
  which is how the `NAME_INCOME_TYPE` proxy-level questions enter the search as measurements
  rather than judgments;
- **hyperparameter overrides** — depth, regularization, and the like.

The space must be wide enough to contain real trade-offs. The evidence section shows that a
space of small perturbations contains nothing but noise; the initial catalog therefore
includes the large feature-subset candidates that produced the frontier, not only ablations.

The catalog lives in `search.py` rather than `config.py`. `config.py` is imported by the
serving app, and the candidate space is training-only; putting it in `config.py` would carry
it into the runtime image for no reason.

### Selection

Among candidates whose AUC is within `MAX_AUC_SACRIFICE` of the best candidate's, take the
highest minimum AIR. Then apply the noise guard: unless the winner beats the baseline
candidate's min AIR by more than `MIN_AIR_IMPROVEMENT`, the baseline is selected instead.
Ties break by AUC, then by declaration order, so the result is deterministic under a fixed
seed.

`MAX_AUC_SACRIFICE = 0.01` is where "comparable performance" is written down. On the
measured frontier it is a tight budget: the 0.028 AUC that the `EXT_SOURCE_*`-only model
costs falls outside it, so **the likely outcome of the first run is that the baseline is
selected and a negative result is recorded.** That is an accepted and intended outcome, not
a failure of the milestone — the frontier is recorded in the artifact regardless of who
wins, which is what makes a later decision to spend more AUC an informed and documented one
rather than an improvisation.

### Why the search is structural

`ModelMetadata.selection` is optional in the schema, because fixture training must keep
working without a four-minute search. `creditboost-artifact verify` requires it when
`provenance == "production"`, in the same place and shape as the existing provenance and
ECOA-feature checks — which run inside the Docker builder, so an image containing a
production artifact that was never searched cannot be built. The guarantee is structural
rather than a test that can be skipped.

## Components

### `src/creditboost/search.py`

Training-side only; added to the `[tool.importlinter]` contract alongside `data.py` and
`train.py` so `serve/` cannot reach it.

```python
@dataclass(frozen=True)
class CandidateSpec:
    name: str
    drops: tuple[str, ...] = ()
    collapses: Mapping[str, Mapping[str, str | None]] = field(default_factory=dict)
    params: Mapping[str, object] = field(default_factory=dict)

CANDIDATES: tuple[CandidateSpec, ...]        # baseline first, always

def apply(spec: CandidateSpec, frame: pd.DataFrame) -> pd.DataFrame: ...
def rank(train_frame: pd.DataFrame, seed: int = config.RANDOM_SEED) -> SearchReport: ...
def select(report: SearchReport,
           auc_budget: float = config.MAX_AUC_SACRIFICE,
           min_improvement: float = config.MIN_AIR_IMPROVEMENT) -> str: ...
```

`rank` performs the nested split and trains every candidate; `select` is a pure function over
a report. Keeping them apart means the selection rule — the part with the subtle failure
modes — is unit-testable against constructed frontiers without training anything, exactly as
`failing_attributes` is in Milestone 4.

### `src/creditboost/schema.py`

```python
class CandidateResult(BaseModel):
    name: str
    roc_auc: float | None = None                  # None only when failed_reason is set
    min_adverse_impact_ratio: float | None = None
    adverse_impact_ratios: dict[str, float] = {}
    n_features: int
    failed_reason: str | None = None

class SearchReport(BaseModel):
    baseline: str
    selected: str
    auc_budget: float
    min_air_improvement: float
    target_approval_rate: float     # the matched rate candidates were ranked at
    ranking_basis: str              # "matched approval rate on the selection split"
    candidates: list[CandidateResult]
```

A failed candidate carries `failed_reason` and no scores; a scored candidate carries scores
and no reason. As with `AttributeFairness` in Milestone 4, exactly one of the two states
holds, and a test enforces it — a failed candidate scoring a default of 0.0 would drag the
recorded frontier downward and misrepresent what the search found.

`ModelMetadata` gains `selection: SearchReport | None = None`. `/metadata` returns
`metadata.model_dump()`, so the frontier becomes publicly visible with no endpoint change —
intended, for the same reason the fairness report is: it contains aggregate model statistics
and no applicant data, and a service that claims to have searched should publish the search.

### `src/creditboost/config.py`

Gains `MAX_AUC_SACRIFICE = 0.01`, `MIN_AIR_IMPROVEMENT = 0.01`, and `SELECTION_SIZE = 0.25`
(the selection fraction *of the training split*, not of the frame). `MODEL_VERSION` becomes
`0.4.0`.

### `src/creditboost/train.py`

Gains `--search`. When passed, `rank` and `select` run against the training split before
fitting; the winner's spec is applied to the final fit; the report is stamped into
`ModelMetadata.selection`. Without it, behaviour is exactly as today. The existing AUC floor
and fairness gate run afterwards on the winner, unchanged and in the same place.

### `creditboost-search`

A read-only console script that runs `rank`, prints the frontier and what `select` would
choose, and writes nothing. It exists so the frontier can be inspected before a release
without any possibility of producing an artifact as a side effect.

Search and training deliberately do **not** communicate through an intermediate file. A
`search.json` on disk could be stamped onto a model it did not select; `--search` makes the
search and the fit it justifies atomic.

### `src/creditboost/artifact_cli.py`

`verify` gains one check: `provenance == "production"` requires `selection` to be present and
`selection.selected` to name a candidate present in `selection.candidates`.

## Error Handling

| Failure | Behaviour |
|---|---|
| A candidate fails to train (e.g. its drops leave no features) | Recorded in the frontier with `failed_reason`, excluded from selection. Never silently skipped — a missing candidate would misrepresent the breadth of the search |
| No candidate is within the AUC budget | Cannot occur: the best candidate is always within the budget of itself |
| The winner ties the baseline within `MIN_AIR_IMPROVEMENT` | Baseline is selected; the report records it, so "we looked and stayed" is distinguishable from "we never looked" |
| An attribute is unmeasured on the selection split | The candidate's min AIR is taken over measured attributes only; a candidate with none measured is recorded as failed rather than scoring 1.0 |
| The winner fails the AUC floor or the fairness gate on the validation split | `creditboost-train` writes nothing and exits non-zero, unchanged from Milestone 4. Selection does not weaken any existing gate |
| `verify` on a production artifact with no `selection` | Rejected, inside the Docker builder |

## Invariant Ledger

### Gained

- **Every production model was selected by a recorded search.** `verify` requires
  `ModelMetadata.selection` when `provenance == "production"`, so an unsearched production
  artifact cannot be built into an image.
- **Candidates are ranked at matched approval rate, never at a fixed threshold.** Ranking at
  a fixed threshold selects leniency and reports it as fairness.
- **The band threshold is never a search axis.** It dominates every model effect, and
  selecting it for its fairness number manufactures assurance instead of establishing it.
- **Selection happens strictly inside the training split.** The validation split never
  participates, so the shipped ratio carries no selection bias.
- **The search's internal matched-threshold ratio never appears in the shipped fairness
  report,** which remains whatever `fairness.evaluate` produced at `RISK_BAND_LOW_MAX`.
- **A negative search result is recorded, not discarded.** The frontier is stamped whether or
  not the baseline wins.
- **No protected attribute influences fitting.** The search varies features, levels, and
  hyperparameters only.

### Amended

- `MODEL_VERSION` becomes `0.4.0`; artifact metadata gains an optional field that `verify`
  makes mandatory in production.
- Milestone 4's out-of-scope list is corrected: per-group thresholds are rejected on the
  merits, not deferred.
- The `[tool.importlinter]` contract extends to `search.py`.

### Unchanged

- No member of `PROTECTED_ATTRIBUTES` is ever a model feature.
- `fairness.py`, the 0.80 floor, the adverse definition, and the AUC floor are untouched.
- Risk-band thresholds remain business policy in `config.py`, changing without retraining.
- Missing values are never imputed; no `scale_pos_weight`.
- No applicant financial field is ever logged.
- CI never downloads from Kaggle; training and search remain manual, local, and credentialed.

## Rejected Alternatives

**Pre-processing reweighting** (Kamiran–Calders sample weights from the protected × target
joint distribution). Rejected primarily on repo-internal grounds rather than legal ones: the
codebase refuses `scale_pos_weight` because it "inflates probabilities away from the true
base rate," which would decalibrate the score the service bands and reports a Brier score
for. Group reweighting does the same thing by the same mechanism. Adopting it means revoking
a standing invariant, and the trade is not worth it.

**Per-group thresholds.** Rejected on the merits. Varying the band cutoff by a prohibited
basis is disparate treatment under ECOA §1691(a)(1) — it is the harm, not the remedy.

**Adversarial debiasing.** Requires a differentiable model. XGBoost is not one, so this means
replacing the model architecture and the runtime dependency set for a technique whose legal
standing in US credit is far less settled than an LDA search.

**A three-way train/select/confirm split.** Rejected in favour of the nested split, which
achieves the same isolation without shrinking the shipped measurement or breaking
comparability with Milestone 4.

**K-fold measurement of every candidate.** Multiplies the search by K to solve a problem the
untouched validation split already solves, and complicates the one-artifact-per-training
story.

## Accepted Risks

**The milestone will probably ship no fairness improvement.** Under a 0.01 AUC budget the
measured frontier contains no qualifying alternative, so the expected outcome is that the
baseline is selected and a negative result recorded. This is the deliberate consequence of
choosing a tight budget, and the recorded frontier is the deliverable that makes a future
decision to spend more AUC an informed one.

**Candidate ranking is measured on less data than the final fit.** Candidates train on 0.6
of the frame and are ranked on a selection split carved from training data, where the noise
is larger than the 0.005 measured on the full validation split. A candidate's rank could in
principle differ at full data size. The noise guard is calibrated for this, and the winner is
always re-measured on the untouched validation split before any gate is applied.

**The candidate space is a human artifact.** A search proves only that no alternative was
found *in the space searched*. The space is declared in one list, recorded in the artifact by
name and count, and can be widened; it cannot be complete. The honest claim the artifact
supports is "these candidates were tried," not "no better model exists."

**Search adds roughly four minutes to a release.** Measured at ~3–4 seconds per candidate;
a fifty-candidate search is about three and a half minutes on ordinary hardware. Accepted:
it runs once per release, on a manual step that is already credentialed and local.

**The threshold finding is recorded but not defended structurally.** Nothing stops a future
change to `RISK_BAND_LOW_MAX` from improving every ratio at the cost of risk appetite. That
remains Milestone 4's documented discipline — when band policy moves, fairness must be
re-measured — and this milestone strengthens the argument for it without changing the
mechanism, because enforcing agreement would still cost the Milestone 1 invariant.

## Testing

### `tests/test_search.py`

Constructed frontiers, no training, for the selection rule:

- **A leniency-only candidate does not win.** The load-bearing test of this milestone, and
  the direct analogue of Milestone 4's `min/max` direction test: a candidate that is fairer
  only because it approves more people must lose to a genuinely fairer one.
- The AUC budget is respected at its boundary: exactly at `best − MAX_AUC_SACRIFICE` is
  eligible, just below is not.
- The noise guard keeps the baseline when the improvement is at or below
  `MIN_AIR_IMPROVEMENT`, and yields when it is above.
- A failed candidate is recorded and never selected; a candidate with no measured attribute
  is failed rather than scored 1.0.
- Ties break deterministically; two runs at the same seed produce identical frontiers.
- `apply` performs drops and level collapses correctly, and never removes a column the
  transform needs to derive another.

### `tests/test_train.py`

- `--search` stamps a `SearchReport` whose `selected` names a candidate in `candidates`.
- The stamped `FairnessReport` is the one `fairness.evaluate` produced at
  `config.RISK_BAND_LOW_MAX` — not the search's internal matched-threshold figure.
- Without `--search`, metadata carries no selection report and behaviour is unchanged.
- The validation split is not used during ranking.

### `tests/test_artifact_cli.py`

`verify` rejects a `provenance: "production"` artifact with no selection report, and one
whose `selected` names an absent candidate.

### `tests/test_schema.py`

`SearchReport` round-trips through JSON; `ModelMetadata` still loads without one.

### Import contract

`lint-imports` proves `serve/` and `artifact_cli` cannot reach `search.py`.

## Sequencing Constraint

`schema.py` must carry `SearchReport` before `search.py` can return one, and `search.py`
must exist before `train.py --search` can call it. `verify`'s new check must land with or
after the retrain, since it invalidates every production artifact lacking a selection
report — `model-v0.3.0` included. The build is red from the moment that check lands until
`model-v0.4.0` is released, which is the same window Milestones 3 and 4 opened and it closes
the same way: retrain, `release-model.sh 0.4.0`, commit the lockfile with the
`MODEL_VERSION` bump in the same commit.

## Success Criteria

1. `creditboost-search` prints a frontier and writes nothing.
2. `creditboost-train --search` ranks candidates on a split nested inside the training data,
   trains the selected candidate, and stamps a `SearchReport` into the artifact.
3. Candidates are ranked at matched approval rate, proven by a test that fails when a
   leniency-only candidate is allowed to win.
4. The selection rule honours the AUC budget and the noise guard, proven at both boundaries.
5. The shipped `FairnessReport` is `fairness.evaluate`'s output at `RISK_BAND_LOW_MAX`, and a
   test proves the search's internal figure never reaches it.
6. The frontier is recorded whether or not the baseline wins.
7. `creditboost-artifact verify` rejects a production artifact with no selection report,
   inside the Docker builder.
8. `model-v0.4.0` is trained with `--search`, released, pinned, and verified, with
   `MODEL_VERSION` at `0.4.0` in the same commit as the lockfile.
9. The shipped model clears the AUC floor and the 0.80 ratio on every measured attribute,
   with the achieved figures and the search's frontier both recorded.
10. `/metadata` exposes the search report with no endpoint change.
11. No new runtime dependency; `ruff`, `mypy`, `lint-imports` and the full suite are clean.
