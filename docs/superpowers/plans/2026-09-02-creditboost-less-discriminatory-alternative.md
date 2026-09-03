# Less Discriminatory Alternative Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the project a repeatable, recorded search for a less discriminatory model, and make the search's result part of every production artifact.

**Architecture:** A new training-side module `search.py` declares a catalog of `CandidateSpec`s that vary features, categorical levels, and hyperparameters — never a protected attribute and never the band threshold. `rank()` splits the *training* frame again, fits every candidate, and scores each at a **matched approval rate** so leniency cannot masquerade as fairness. `select()` is a pure function that applies an AUC budget and a noise guard. The frontier is stamped into `ModelMetadata.selection` and required of production artifacts by `creditboost-artifact verify`. The validation split never participates in selection.

**Tech Stack:** Python 3.12, pandas, numpy, pydantic v2, xgboost, scikit-learn (training only), pytest, import-linter.

**Spec:** `docs/superpowers/specs/2026-09-02-creditboost-less-discriminatory-alternative-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Python 3.12 only.** On macOS, `brew install libomp` before `pip install`.
- **No new runtime dependency.** pandas, numpy, pydantic and xgboost are already present.
- **Candidates are ranked at matched approval rate, never at a fixed threshold.** Ranking at a fixed threshold selects whichever candidate approves the most people and reports that as fairness. This is the single most dangerous error in the milestone.
- **The band threshold is never a search axis.** `CandidateSpec` has no field that can express one, and a test pins that.
- **Selection happens strictly inside the training split.** `rank()` receives the training frame only; it is structurally incapable of touching validation data.
- **The search's internal matched-threshold ratio never reaches the artifact.** `ModelMetadata.fairness` remains exactly what `fairness.evaluate` produced at `config.RISK_BAND_LOW_MAX`.
- **One implementation of the ratio arithmetic.** The search reuses `fairness.py`'s grouping and `min/max` direction rather than reimplementing it — two implementations would mean the direction invariant has one test and two ways to break.
- **No candidate may name a member of `config.PROTECTED_ATTRIBUTES`.** A test enforces it.
- **`MIN_ADVERSE_IMPACT_RATIO` (0.80) and `MIN_VALIDATION_AUC` (0.70) are untouched.** This milestone adds a search; it weakens no existing gate.
- **`ruff` line-length 100**, rules `["E", "F", "I", "UP", "B"]`. Run `ruff check . && ruff format --check . && mypy src/ && lint-imports` before every commit.
- CI never downloads from Kaggle. Training and search stay manual, local and credentialed.

## Two Deviations From the Spec

Both were found while working out exact interfaces. Neither changes the design's intent.

**1. `select()` takes candidates, not a report.** The spec sketched `select(report) -> str`, but `SearchReport` carries a `selected` field, so a report cannot exist before selection has happened. `select()` therefore takes `Sequence[CandidateResult]` plus the baseline name, and `rank()`'s caller builds the `SearchReport` afterwards. This keeps `select` pure and trivially testable, which was the spec's actual goal.

**2. A non-baseline winner refuses to write, and says what to change.** The spec did not address the interaction between a winning candidate and the train/serve skew gate. It is load-bearing:

- A winner with `drops` has fewer features than `config.FEATURE_ORDER`, so `artifact.load()` rejects it and serving's transform emits columns the model never saw.
- A winner with `collapses` was trained on levels that serving would still pass through unmapped — train/serve skew by another name.
- A winner with `params` differs from `train.PARAMS`, so the run is not reproducible from the code.

In every case, adopting a candidate is a **reviewed code change**, not something a training run may do to itself. So `creditboost-train --search` writes nothing when the winner is not the baseline; it prints the frontier and the change required. The operator makes that change — which promotes the winner to be the code's baseline — and re-runs, at which point baseline wins and the artifact records the frontier.

This mirrors the AUC floor and the fairness gate exactly: a failing condition is a conversation, not an automatic action. It also makes the loop converge — the code always encodes the best known spec, and the search's job is to say when that stops being true.

## Build-Red Window

**Task 7 breaks `docker build`, and Task 8 is the only thing that fixes it.**

Task 7 makes `verify` reject a production artifact with no selection report. The pinned `model-v0.3.0` has none, so `creditboost-artifact verify` fails inside the Docker builder — correctly.

- Between Task 7's commit and Task 8's, `docker build .` fails. Expected, and documented in Task 7's commit message.
- Task 8 requires `data/application_train.csv` (present on the development machine) and `gh` credentials.
- **The branch must not merge until Task 8 lands.**

The final fit is unaffected by ranking — same train split, same `PARAMS`, same seed — so `model-v0.4.0`'s ratios should reproduce `model-v0.3.0`'s exactly: sex 0.868, marital status 0.818, age 0.810. Under a 0.01 AUC budget the measured frontier contains no qualifying alternative, so **the expected outcome is that baseline wins and a negative result is recorded.** That is the designed outcome, not a failure.

## File Structure

| File | Responsibility |
|---|---|
| `src/creditboost/schema.py` | **Modify.** Add `CandidateResult`, `SearchReport`; add optional `selection` to `ModelMetadata`. |
| `src/creditboost/fairness.py` | **Modify.** Extract `adverse_impact_ratios(frame, adverse, ...)` from `evaluate`, so the search reuses one implementation of the grouping and the `min/max` direction. |
| `src/creditboost/search.py` | **Create.** `CandidateSpec`, `CANDIDATES`, `apply`, `matched_adverse_mask`, `rank`, `select`. Training-side: imports `data.split`. |
| `src/creditboost/config.py` | **Modify.** Add `MAX_AUC_SACRIFICE`, `MIN_AIR_IMPROVEMENT`, `SELECTION_SIZE`; later bump `MODEL_VERSION` to `0.4.0`. |
| `src/creditboost/train.py` | **Modify.** `fit` accepts a spec; `main` gains `--search`; refuses to write on a non-baseline winner. |
| `src/creditboost/search_cli.py` | **Create.** The read-only `creditboost-search` command. |
| `src/creditboost/artifact_cli.py` | **Modify.** `verify` requires `selection` when provenance is `production`. |
| `pyproject.toml` | **Modify.** Add `creditboost.search` to the import-linter forbidden list; register the `creditboost-search` script. |
| `tests/conftest.py` | **Modify.** Add `a_search_report()`. |
| `tests/test_search.py` | **Create.** The catalog, `apply`, the matched mask, `select`, and `rank`. |
| `tests/test_schema.py`, `tests/test_fairness.py`, `tests/test_train.py`, `tests/test_api.py`, `tests/test_artifact_cli.py` | **Modify.** |
| `models/model.lock.json` | **Modify.** Repointed at `model-v0.4.0`. |
| `CLAUDE.md`, `README.md` | **Modify.** Roadmap, invariant ledger, commands. |

---

### Task 1: The search schema

**Files:**
- Modify: `src/creditboost/schema.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `CandidateResult(name: str, n_features: int, roc_auc: float | None, min_adverse_impact_ratio: float | None, adverse_impact_ratios: dict[str, float], failed_reason: str | None)`; `SearchReport(baseline: str, selected: str, auc_budget: float, min_air_improvement: float, target_approval_rate: float, ranking_basis: str, candidates: list[CandidateResult])`; `ModelMetadata.selection: SearchReport | None`; `conftest.a_search_report()`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_schema.py`:

```python
import pytest
from pydantic import ValidationError

from creditboost.schema import CandidateResult, ModelMetadata, SearchReport


def a_candidate(name="baseline", auc=0.75, air=0.81):
    return CandidateResult(
        name=name,
        n_features=20,
        roc_auc=auc,
        min_adverse_impact_ratio=air,
        adverse_impact_ratios={"CODE_GENDER": 0.87, "DAYS_BIRTH": air},
    )


def a_report(candidates=None, baseline="baseline", selected="baseline"):
    candidates = candidates if candidates is not None else [a_candidate()]
    return SearchReport(
        baseline=baseline,
        selected=selected,
        auc_budget=0.01,
        min_air_improvement=0.01,
        target_approval_rate=0.743,
        ranking_basis="matched approval rate on the selection split",
        candidates=candidates,
    )


def test_search_report_round_trips_through_json():
    report = a_report()
    assert SearchReport.model_validate_json(report.model_dump_json()) == report


def test_a_failed_candidate_carries_a_reason_and_no_scores():
    result = CandidateResult(name="empty", n_features=0, failed_reason="no features remain")
    assert result.roc_auc is None
    assert result.min_adverse_impact_ratio is None
    assert result.adverse_impact_ratios == {}


def test_a_candidate_cannot_be_both_scored_and_failed():
    with pytest.raises(ValidationError):
        CandidateResult(
            name="x",
            n_features=3,
            roc_auc=0.7,
            min_adverse_impact_ratio=0.9,
            failed_reason="boom",
        )


def test_a_candidate_must_be_either_scored_or_failed():
    # A default of 0.0 would silently drag the recorded frontier downward and
    # misrepresent what the search found.
    with pytest.raises(ValidationError):
        CandidateResult(name="x", n_features=3)


def test_selected_must_name_a_candidate_in_the_frontier():
    with pytest.raises(ValidationError):
        a_report(selected="a-candidate-that-was-never-run")


def test_baseline_must_name_a_candidate_in_the_frontier():
    with pytest.raises(ValidationError):
        a_report(baseline="not-in-the-list", selected="baseline")


def test_model_metadata_loads_without_a_selection_report():
    from tests.conftest import a_passing_fairness_report

    metadata = ModelMetadata(
        version="0.4.0",
        trained_at="2026-09-02T00:00:00+00:00",
        dataset_sha256="0" * 64,
        n_train_rows=100,
        feature_order=["EXT_SOURCE_1"],
        metrics={"roc_auc": 0.75},
        xgboost_version="3.4.1",
        provenance="fixture",
        fairness=a_passing_fairness_report(),
    )
    assert metadata.selection is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_schema.py -k "candidate or search_report or selection" -v`
Expected: FAIL with `ImportError: cannot import name 'CandidateResult'`.

- [ ] **Step 3: Implement the schema**

In `src/creditboost/schema.py`, add after `FairnessReport`:

```python
class CandidateResult(BaseModel):
    """One model specification's score in the search.

    Exactly one of the two states holds, for the same reason AttributeFairness
    has the rule: a candidate that could not be trained has established nothing,
    and a default score of 0.0 would drag the recorded frontier downward and
    misrepresent the breadth of the search.
    """

    name: str
    n_features: int = Field(ge=0)
    roc_auc: float | None = Field(default=None, ge=0, le=1)
    min_adverse_impact_ratio: float | None = Field(default=None, ge=0, le=1)
    adverse_impact_ratios: dict[str, float] = Field(default_factory=dict)
    failed_reason: str | None = None

    @model_validator(mode="after")
    def exactly_one_outcome(self) -> CandidateResult:
        scored = self.roc_auc is not None and self.min_adverse_impact_ratio is not None
        failed = self.failed_reason is not None
        if scored == failed:
            raise ValueError(
                f"candidate {self.name!r} must be either scored (roc_auc and "
                "min_adverse_impact_ratio) or failed (failed_reason), never both "
                "and never neither"
            )
        return self


class SearchReport(BaseModel):
    """The less discriminatory alternative search that produced this model.

    Recorded whether or not it changed anything. A search that found nothing is
    the evidence that a search was conducted, and an artifact that discarded it
    could not distinguish "looked and stayed" from "never looked".

    target_approval_rate and ranking_basis record how candidates were compared.
    That comparison is internal to the search and is NOT the artifact's fairness
    report, which is measured separately at config.RISK_BAND_LOW_MAX.
    """

    baseline: str
    selected: str
    auc_budget: float = Field(ge=0)
    min_air_improvement: float = Field(ge=0)
    target_approval_rate: float = Field(ge=0, le=1)
    ranking_basis: str
    candidates: list[CandidateResult]

    @model_validator(mode="after")
    def names_resolve(self) -> SearchReport:
        known = {candidate.name for candidate in self.candidates}
        for label, name in (("baseline", self.baseline), ("selected", self.selected)):
            if name not in known:
                raise ValueError(
                    f"{label} {name!r} does not name a candidate in this report; "
                    f"known candidates are {sorted(known)}"
                )
        return self
```

Then add the field to `ModelMetadata`, after `fairness`:

```python
    # Optional in the schema so fixture training need not run a four-minute
    # search. creditboost-artifact verify requires it when provenance is
    # "production", which is the gate that runs inside the Docker builder.
    selection: SearchReport | None = None
```

- [ ] **Step 4: Add the conftest helper**

In `tests/conftest.py`:

```python
def a_search_report():
    """A minimal report in which the baseline won, for tests that need a valid
    production ModelMetadata but are not about the search."""
    from creditboost.schema import CandidateResult, SearchReport

    return SearchReport(
        baseline="baseline",
        selected="baseline",
        auc_budget=0.01,
        min_air_improvement=0.01,
        target_approval_rate=0.74,
        ranking_basis="matched approval rate on the selection split",
        candidates=[
            CandidateResult(
                name="baseline",
                n_features=20,
                roc_auc=0.75,
                min_adverse_impact_ratio=0.81,
                adverse_impact_ratios={"CODE_GENDER": 0.87, "DAYS_BIRTH": 0.81},
            )
        ],
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_schema.py -v`
Expected: PASS, including every pre-existing test.

- [ ] **Step 6: Run the full suite and the linters**

Run: `.venv/bin/python -m pytest -m "not slow" -q && ruff check . && ruff format --check . && mypy src/ && lint-imports`
Expected: all green. `ModelMetadata.selection` defaults to `None`, so no existing construction site needs changing.

- [ ] **Step 7: Commit**

```bash
git add src/creditboost/schema.py tests/conftest.py tests/test_schema.py
git commit -m "feat: add the search report schema

CandidateResult and SearchReport, with the same exactly-one-outcome rule
AttributeFairness carries: a candidate that could not be trained has
established nothing, and a default score would misrepresent the frontier.

selection is optional on ModelMetadata so fixture training need not run a
search; verify will require it for production artifacts.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: One implementation of the ratio arithmetic

**Files:**
- Modify: `src/creditboost/fairness.py`
- Test: `tests/test_fairness.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `fairness.adverse_impact_ratios(frame: pd.DataFrame, adverse: Sequence[bool], min_group_size: int = config.MIN_FAIRNESS_GROUP_SIZE) -> list[AttributeFairness]`. `evaluate` keeps its exact signature and behaviour.

This is a behaviour-preserving refactor. The search must score candidates at a matched threshold rather than at `RISK_BAND_LOW_MAX`, and it must not reimplement the age bucketing, the minimum-group-size rule, or the `min/max` direction to do it. Two implementations would mean the most dangerous invariant in the codebase has one test and two ways to break.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fairness.py`:

```python
from creditboost.fairness import adverse_impact_ratios, evaluate


def test_adverse_impact_ratios_is_driven_by_the_mask_it_is_given():
    """Same frame, two different adverse masks, two different ratios. This is
    what lets the search score a candidate at a threshold of its own choosing
    without reimplementing any of the grouping or the min/max direction."""
    frame = a_frame_with_two_sexes(n_per_group=200)

    everyone_adverse = [True] * len(frame)
    nobody_adverse = [False] * len(frame)

    all_adverse = adverse_impact_ratios(frame, everyone_adverse, min_group_size=10)
    none_adverse = adverse_impact_ratios(frame, nobody_adverse, min_group_size=10)

    sex_all = next(a for a in all_adverse if a.attribute == "CODE_GENDER")
    sex_none = next(a for a in none_adverse if a.attribute == "CODE_GENDER")

    # Everyone adverse is the degenerate case: no favourable outcome anywhere.
    assert sex_all.adverse_impact_ratio is None
    assert sex_all.unmeasured_reason is not None
    # Nobody adverse is perfect parity.
    assert sex_none.adverse_impact_ratio == 1.0


def test_evaluate_delegates_to_adverse_impact_ratios():
    """evaluate is exactly adverse_impact_ratios over a band-derived mask, so a
    caller passing the equivalent mask gets an identical answer."""
    from creditboost.banding import risk_band

    frame = a_frame_with_two_sexes(n_per_group=200)
    probabilities = [0.05 + 0.4 * (i % 3 == 0) for i in range(len(frame))]

    via_evaluate = evaluate(frame, probabilities, min_group_size=10).attributes
    mask = [risk_band(p) != "low" for p in probabilities]
    via_mask = adverse_impact_ratios(frame, mask, min_group_size=10)

    assert via_evaluate == via_mask
```

Add this frame builder near the top of `tests/test_fairness.py` if an equivalent
helper is not already present; reuse the existing one if it is.

```python
def a_frame_with_two_sexes(n_per_group: int) -> pd.DataFrame:
    """Half F, half M, all aged 40, marital status constant. Only CODE_GENDER
    has two eligible groups, which keeps the assertions about one attribute."""
    n = n_per_group * 2
    return pd.DataFrame(
        {
            "CODE_GENDER": ["F"] * n_per_group + ["M"] * n_per_group,
            "DAYS_BIRTH": [-40 * 365] * n,
            "NAME_FAMILY_STATUS": ["Married"] * n,
        }
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fairness.py -k "mask or delegates" -v`
Expected: FAIL with `ImportError: cannot import name 'adverse_impact_ratios'`.

- [ ] **Step 3: Perform the refactor**

In `src/creditboost/fairness.py`, replace the body of `evaluate` with a delegation, and move the loop into the new function. The loop body is unchanged — only the source of `adverse` moves:

```python
def adverse_impact_ratios(
    frame: pd.DataFrame,
    adverse: Sequence[bool],
    min_group_size: int = config.MIN_FAIRNESS_GROUP_SIZE,
) -> list[AttributeFairness]:
    """Adverse impact ratio per protected attribute, given an explicit adverse
    mask.

    The mask is a parameter so the alternative search can score a candidate at a
    threshold matched to another candidate's approval rate, without
    reimplementing the age bucketing, the minimum group size, or -- above all --
    the min/max direction. There is one implementation of that arithmetic.
    """
    adverse_series = pd.Series(list(adverse), index=frame.index)

    attributes: list[AttributeFairness] = []
    for attribute in config.FAIRNESS_ATTRIBUTES:
        table = pd.DataFrame({"group": _groups(frame, attribute), "adverse": adverse_series})
        table = table.dropna(subset=["group"])
        summary = table.groupby("group")["adverse"].agg(["mean", "size"])
        eligible = summary[summary["size"] >= min_group_size]

        group_rates = [
            GroupRate(group=str(name), adverse_rate=float(row["mean"]), n=int(row["size"]))
            for name, row in eligible.iterrows()
        ]

        if len(eligible) < 2:
            attributes.append(
                AttributeFairness(
                    attribute=attribute,
                    unmeasured_reason=(
                        f"fewer than two groups reached the minimum size of {min_group_size}"
                    ),
                    groups=group_rates,
                )
            )
            continue

        favourable = 1.0 - eligible["mean"]
        if favourable.max() <= 0.0:
            attributes.append(
                AttributeFairness(
                    attribute=attribute,
                    unmeasured_reason=("no applicant in any group received the favourable outcome"),
                    groups=group_rates,
                )
            )
            continue

        # min over max, never the reverse. Inverted, a failing model reads as
        # passing and the gate silently permits everything.
        attributes.append(
            AttributeFairness(
                attribute=attribute,
                adverse_impact_ratio=float(favourable.min() / favourable.max()),
                groups=group_rates,
            )
        )

    return attributes


def evaluate(
    frame: pd.DataFrame,
    probabilities: Sequence[float],
    min_group_size: int = config.MIN_FAIRNESS_GROUP_SIZE,
) -> FairnessReport:
    """Adverse impact ratio per protected attribute, over the rows in `frame`.

    min_group_size is a parameter rather than read from config directly so tests
    can lower it; train.py takes the configured default.
    """
    adverse = [risk_band(float(p)) != "low" for p in probabilities]
    return FairnessReport(
        adverse_definition=ADVERSE_DEFINITION,
        band_low_max=config.RISK_BAND_LOW_MAX,
        min_group_size=min_group_size,
        attributes=adverse_impact_ratios(frame, adverse, min_group_size),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fairness.py -v`
Expected: PASS. Every pre-existing test in the module must pass unchanged — that is what makes this refactor safe.

- [ ] **Step 5: Run the full suite and the linters**

Run: `.venv/bin/python -m pytest -m "not slow" -q && ruff check . && ruff format --check . && mypy src/ && lint-imports`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/creditboost/fairness.py tests/test_fairness.py
git commit -m "refactor: expose adverse_impact_ratios over an explicit mask

The search must score candidates at a threshold matched to a reference
approval rate rather than at RISK_BAND_LOW_MAX, and it must not
reimplement the age bucketing, the group size rule, or the min/max
direction to do it. Two implementations would leave the most dangerous
invariant in the codebase with one test and two ways to break.

evaluate keeps its signature and is now a band-derived mask over the same
function; the existing tests pass unchanged, which is the proof.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: The candidate catalog and `apply`

**Files:**
- Create: `src/creditboost/search.py`
- Modify: `pyproject.toml`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `search.CandidateSpec(name, drops, collapses, params)`; `search.CANDIDATES: tuple[CandidateSpec, ...]` with the baseline first; `search.BASELINE: CandidateSpec`; `search.apply(spec, frame) -> pd.DataFrame`; `search.UnknownFeatureError`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_search.py`:

```python
"""The alternative search: the catalog, the spec transform, the matched mask,
the selection rule, and the ranking."""

import dataclasses

import pandas as pd
import pytest

from creditboost import config
from creditboost.features import transform
from creditboost.search import BASELINE, CANDIDATES, CandidateSpec, UnknownFeatureError, apply


def a_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [100_000.0, 200_000.0, 150_000.0],
            "AMT_CREDIT": [400_000.0, 500_000.0, 300_000.0],
            "AMT_ANNUITY": [20_000.0, 25_000.0, 15_000.0],
            "DAYS_BIRTH": [-15_000, -20_000, -12_000],
            "DAYS_EMPLOYED": [-2_000, -3_000, -500],
            "NAME_INCOME_TYPE": ["Working", "Pensioner", "Maternity leave"],
        }
    )


def test_the_baseline_is_first_and_changes_nothing():
    frame = a_raw_frame()
    assert CANDIDATES[0] is BASELINE
    pd.testing.assert_frame_equal(apply(BASELINE, frame), transform(frame))


def test_apply_drops_only_the_named_columns():
    frame = a_raw_frame()
    spec = CandidateSpec(name="x", drops=("CNT_CHILDREN",))
    result = apply(spec, frame)
    assert "CNT_CHILDREN" not in result.columns
    assert len(result.columns) == len(config.FEATURE_ORDER) - 1


def test_dropping_a_source_column_leaves_its_derived_feature_intact():
    """Drops apply to the transform's OUTPUT, so a derived feature is already
    computed by the time its source is removed. employed_to_age must survive
    DAYS_EMPLOYED being dropped, and keep its value."""
    frame = a_raw_frame()
    spec = CandidateSpec(name="x", drops=("DAYS_EMPLOYED",))
    result = apply(spec, frame)
    assert "DAYS_EMPLOYED" not in result.columns
    assert "employed_to_age" in result.columns
    pd.testing.assert_series_equal(
        result["employed_to_age"], transform(frame)["employed_to_age"]
    )


def test_apply_collapses_a_level_before_the_transform():
    frame = a_raw_frame()
    spec = CandidateSpec(
        name="x", collapses={"NAME_INCOME_TYPE": {"Maternity leave": "Working"}}
    )
    result = apply(spec, frame)
    assert list(result["NAME_INCOME_TYPE"].astype("string")) == [
        "Working",
        "Pensioner",
        "Working",
    ]


def test_a_collapse_target_of_none_becomes_missing():
    frame = a_raw_frame()
    spec = CandidateSpec(name="x", collapses={"NAME_INCOME_TYPE": {"Pensioner": None}})
    result = apply(spec, frame)
    assert result["NAME_INCOME_TYPE"].isna().tolist() == [False, True, False]


def test_apply_rejects_a_drop_naming_an_unknown_column():
    spec = CandidateSpec(name="x", drops=("NOT_A_FEATURE",))
    with pytest.raises(UnknownFeatureError, match="NOT_A_FEATURE"):
        apply(spec, a_raw_frame())


def test_candidate_names_are_unique():
    names = [spec.name for spec in CANDIDATES]
    assert len(names) == len(set(names))


def test_no_candidate_names_a_protected_attribute():
    """ECOA: a protected attribute is never a model feature, so no candidate can
    drop one (it was never there) or collapse its levels."""
    for spec in CANDIDATES:
        for prohibited in config.PROTECTED_ATTRIBUTES:
            assert prohibited not in spec.drops
            assert prohibited not in spec.collapses


def test_every_candidate_drop_names_a_real_feature():
    for spec in CANDIDATES:
        for column in spec.drops:
            assert column in config.FEATURE_ORDER, f"{spec.name} drops unknown {column}"


def test_a_candidate_cannot_vary_the_band_threshold():
    """The band threshold dominates every model effect, so selecting it for its
    fairness number would manufacture assurance rather than establish it. A
    CandidateSpec has no field that can express one, and this pins that."""
    assert {field.name for field in dataclasses.fields(CandidateSpec)} == {
        "name",
        "drops",
        "collapses",
        "params",
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'creditboost.search'`.

- [ ] **Step 3: Create the module**

Create `src/creditboost/search.py`:

```python
"""The less discriminatory alternative search.

Milestone 4 measures disparate impact and refuses to ship a model that fails the
four-fifths rule. It leaves exactly one response to a failing gate: stop. This
module supplies the other one -- a recorded search for a model that does better.

Disparate impact is a burden-shifting doctrine: business necessity rebuts a
prima facie case only if no less discriminatory alternative achieving comparable
performance exists. Searching for one is the obligation, and a recorded search,
including one that finds nothing, is the evidence the obligation was met.

Two rules run through everything here:

  * Candidates are compared at a MATCHED APPROVAL RATE, never at a fixed
    threshold. At a fixed threshold the ratio reports how lenient a candidate is
    rather than how fair, and the search would reliably pick whichever model
    approves the most people.

  * The band threshold is never a search axis. It dominates every model effect
    -- moving RISK_BAND_LOW_MAX from 0.10 to 0.12 buys more ratio than every
    model variant measured -- and it is risk appetite, not fairness. Selecting it
    for its fairness number is precisely the failure Milestone 4 was written to
    avoid. CandidateSpec has no field that can express one.

This module is training-side: it imports data.split, so serve/ must never reach
it. The import-linter contract in pyproject.toml enforces that.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from . import config
from .features import transform


class UnknownFeatureError(ValueError):
    """A candidate names a column that is not a model feature."""


@dataclass(frozen=True)
class CandidateSpec:
    """One model specification to try.

    Three axes, none of which is a protected attribute and none of which is the
    band threshold:

      drops      -- columns removed from the transform's output. They apply
                    AFTER the transform, so removing DAYS_EMPLOYED does not
                    disturb employed_to_age, which is already computed.
      collapses  -- {column: {level: replacement_or_None}}, applied to the raw
                    frame BEFORE the transform, because the transform builds a
                    Categorical from the declared levels. This is how the
                    NAME_INCOME_TYPE proxy-level questions enter the search as
                    measurements rather than judgments.
      params     -- xgboost parameter overrides.
    """

    name: str
    drops: tuple[str, ...] = ()
    collapses: Mapping[str, Mapping[str, str | None]] = field(default_factory=dict)
    params: Mapping[str, object] = field(default_factory=dict)


BASELINE = CandidateSpec(name="baseline")

# The catalog must be wide enough to contain real trade-offs. A space of small
# perturbations was measured and contains nothing: sixteen ablation and
# regularization variants spanned a minimum AIR of 0.8058 to 0.8146, entirely
# inside the sampling noise. The large feature-subset candidates are the ones
# that produced a frontier, so they are here on purpose, not as padding.
#
# The two NAME_INCOME_TYPE candidates carry the proxy-level question deferred
# since Milestone 3: "Maternity leave" is a sex proxy and "Pensioner" an age
# proxy. Measured, they move the ratio by 0.0001. They stay in the catalog so
# that finding is re-established on every search rather than remembered.
CANDIDATES: tuple[CandidateSpec, ...] = (
    BASELINE,
    CandidateSpec(
        name="maternity-to-working",
        collapses={"NAME_INCOME_TYPE": {"Maternity leave": "Working"}},
    ),
    CandidateSpec(
        name="income-type-proxies-dropped",
        collapses={"NAME_INCOME_TYPE": {"Maternity leave": None, "Pensioner": None}},
    ),
    CandidateSpec(name="no-income-type", drops=("NAME_INCOME_TYPE",)),
    CandidateSpec(name="no-occupation", drops=("OCCUPATION_TYPE",)),
    CandidateSpec(name="no-housing-type", drops=("NAME_HOUSING_TYPE",)),
    CandidateSpec(name="no-household", drops=("CNT_CHILDREN", "CNT_FAM_MEMBERS")),
    CandidateSpec(name="no-employed-to-age", drops=("employed_to_age",)),
    CandidateSpec(name="no-employment", drops=("DAYS_EMPLOYED", "employed_to_age")),
    # The large subsets. external-scores-only was measured at min AIR 0.866
    # against the baseline's 0.810, at a cost of 0.028 AUC -- the only candidate
    # so far to move the ratio by more than the noise.
    CandidateSpec(
        name="external-scores-only",
        drops=tuple(
            column for column in config.FEATURE_ORDER if not column.startswith("EXT_SOURCE")
        ),
    ),
    CandidateSpec(
        name="no-external-scores",
        drops=tuple(
            column for column in config.FEATURE_ORDER if column.startswith("EXT_SOURCE")
        ),
    ),
    CandidateSpec(name="depth-3", params={"max_depth": 3}),
    CandidateSpec(name="depth-4", params={"max_depth": 4}),
    CandidateSpec(name="min-child-weight-50", params={"min_child_weight": 50}),
    CandidateSpec(name="lambda-10", params={"lambda": 10}),
    CandidateSpec(
        name="income-type-proxies-dropped-depth-3",
        collapses={"NAME_INCOME_TYPE": {"Maternity leave": None, "Pensioner": None}},
        params={"max_depth": 3},
    ),
)


def apply(spec: CandidateSpec, frame: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw frame into this candidate's feature matrix.

    Collapses run before the transform and drops after it, which is what keeps
    a derived feature intact when its source column is dropped.
    """
    unknown = [column for column in spec.drops if column not in config.FEATURE_ORDER]
    if unknown:
        raise UnknownFeatureError(
            f"candidate {spec.name!r} drops {unknown}, which are not model "
            f"features. Known features: {list(config.FEATURE_ORDER)}"
        )

    raw = frame
    for column, mapping in spec.collapses.items():
        if column in raw.columns:
            raw = raw.assign(**{column: raw[column].replace(dict(mapping))})

    out = transform(raw)
    return out[[column for column in out.columns if column not in spec.drops]]
```

- [ ] **Step 4: Add the import-linter rule**

In `pyproject.toml`, add `"creditboost.search"` to `forbidden_modules`:

```toml
forbidden_modules = [
    "creditboost.data",
    "creditboost.train",
    "creditboost.search",
]
```

`search.py` imports `data.split` in Task 5, so it is training-side like `train.py`. Adding it explicitly means the contract still holds if that import ever goes away.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_search.py -v && lint-imports`
Expected: PASS, and the import contract is kept.

- [ ] **Step 6: Run the full suite and the linters**

Run: `.venv/bin/python -m pytest -m "not slow" -q && ruff check . && ruff format --check . && mypy src/ && lint-imports`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/creditboost/search.py tests/test_search.py pyproject.toml
git commit -m "feat: add the candidate catalog and spec transform

CandidateSpec varies features, categorical levels and hyperparameters --
never a protected attribute, and never the band threshold, which it has no
field to express. A test pins that field set, because the threshold
dominates every model effect and selecting it for its fairness number
would manufacture assurance rather than establish it.

Collapses run before the transform and drops after it, so a derived
feature survives its source column being dropped.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: The selection rule

**Files:**
- Modify: `src/creditboost/search.py`
- Modify: `src/creditboost/config.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `schema.CandidateResult` (Task 1); `search.CANDIDATES` (Task 3).
- Produces: `search.select(candidates: Sequence[CandidateResult], baseline: str, auc_budget: float = config.MAX_AUC_SACRIFICE, min_improvement: float = config.MIN_AIR_IMPROVEMENT) -> str`; `config.MAX_AUC_SACRIFICE = 0.01`; `config.MIN_AIR_IMPROVEMENT = 0.01`; `search.BaselineMissingError`.

`select` is a pure function over already-scored candidates. Every candidate reaching it was scored at the same matched approval rate, so the comparison is like-for-like by construction.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_search.py`:

```python
from creditboost.schema import CandidateResult
from creditboost.search import BaselineMissingError, select


def scored(name, auc, air):
    return CandidateResult(
        name=name,
        n_features=20,
        roc_auc=auc,
        min_adverse_impact_ratio=air,
        adverse_impact_ratios={"DAYS_BIRTH": air},
    )


def failed(name, reason="did not train"):
    return CandidateResult(name=name, n_features=0, failed_reason=reason)


def test_the_fairest_candidate_within_the_budget_wins():
    candidates = [
        scored("baseline", auc=0.750, air=0.810),
        scored("fairer", auc=0.745, air=0.860),
    ]
    assert select(candidates, baseline="baseline", auc_budget=0.01) == "fairer"


def test_a_candidate_outside_the_auc_budget_cannot_win():
    candidates = [
        scored("baseline", auc=0.750, air=0.810),
        scored("much-fairer-but-worse", auc=0.700, air=0.950),
    ]
    assert select(candidates, baseline="baseline", auc_budget=0.01) == "baseline"


def test_the_budget_boundary_is_inclusive():
    """Exactly at best - budget is eligible; a hair below is not."""
    inside = [
        scored("baseline", auc=0.750, air=0.810),
        scored("edge", auc=0.740, air=0.900),
    ]
    assert select(inside, baseline="baseline", auc_budget=0.01) == "edge"

    outside = [
        scored("baseline", auc=0.750, air=0.810),
        scored("edge", auc=0.7399, air=0.900),
    ]
    assert select(outside, baseline="baseline", auc_budget=0.01) == "baseline"


def test_the_noise_guard_keeps_the_baseline_on_an_improvement_within_noise():
    """AIR's measured sd is about 0.005 on the full validation split and larger
    on the smaller selection split. Without this guard the search churns the
    shipped model on noise."""
    candidates = [
        scored("baseline", auc=0.750, air=0.810),
        scored("marginally-fairer", auc=0.750, air=0.815),
    ]
    assert select(candidates, baseline="baseline", min_improvement=0.01) == "baseline"


def test_the_noise_guard_boundary_is_exclusive():
    """An improvement of exactly min_improvement keeps the baseline. Written in
    exactly-representable binary fractions so the assertion tests the rule and
    not the float arithmetic: 0.75 - 0.5 is exactly 0.25."""
    candidates = [
        scored("baseline", auc=0.750, air=0.5),
        scored("exactly-at-the-guard", auc=0.750, air=0.75),
    ]
    assert select(candidates, baseline="baseline", min_improvement=0.25) == "baseline"

    candidates = [
        scored("baseline", auc=0.750, air=0.5),
        scored("past-the-guard", auc=0.750, air=0.9),
    ]
    assert select(candidates, baseline="baseline", min_improvement=0.25) == "past-the-guard"


def test_the_noise_guard_yields_to_an_improvement_above_it():
    candidates = [
        scored("baseline", auc=0.750, air=0.810),
        scored("really-fairer", auc=0.750, air=0.8201),
    ]
    assert select(candidates, baseline="baseline", min_improvement=0.01) == "really-fairer"


def test_a_failed_candidate_is_never_selected():
    candidates = [scored("baseline", auc=0.750, air=0.810), failed("broken")]
    assert select(candidates, baseline="baseline") == "baseline"


def test_ties_break_by_auc_then_by_declaration_order():
    candidates = [
        scored("baseline", auc=0.700, air=0.810),
        scored("first", auc=0.750, air=0.900),
        scored("second", auc=0.750, air=0.900),
        scored("third", auc=0.749, air=0.900),
    ]
    assert select(candidates, baseline="baseline", auc_budget=0.01) == "first"


def test_selection_is_deterministic():
    candidates = [
        scored("baseline", auc=0.750, air=0.810),
        scored("a", auc=0.749, air=0.900),
        scored("b", auc=0.749, air=0.900),
    ]
    assert len({select(candidates, baseline="baseline") for _ in range(20)}) == 1


def test_a_missing_or_failed_baseline_is_an_error():
    with pytest.raises(BaselineMissingError):
        select([scored("other", auc=0.75, air=0.81)], baseline="baseline")
    with pytest.raises(BaselineMissingError):
        select([failed("baseline")], baseline="baseline")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_search.py -k "select or budget or noise or ties or baseline" -v`
Expected: FAIL with `ImportError: cannot import name 'select'`.

- [ ] **Step 3: Add the config constants**

In `src/creditboost/config.py`, at the end of the disparate impact section:

```python
# --- Alternative search ----------------------------------------------------

# How much validation ROC-AUC the search may trade away for a fairer model.
# This is where "comparable performance" is written down: the burden-shifting
# test asks whether a LESS DISCRIMINATORY ALTERNATIVE ACHIEVING COMPARABLE
# PERFORMANCE exists, and a number stated once is more honest than a judgment
# made per release. It is a tight budget on the measured frontier -- the
# external-scores-only model costs 0.028 AUC and falls outside it.
MAX_AUC_SACRIFICE = 0.01

# A winning candidate must beat the baseline's minimum adverse impact ratio by
# more than this, or the baseline is kept. Grounded in measurement: 300
# bootstrap resamples of the validation split put AIR's standard deviation at
# about 0.005, and the selection split is smaller still. Without this guard the
# search churns the shipped model on noise.
MIN_AIR_IMPROVEMENT = 0.01
```

- [ ] **Step 4: Implement `select`**

In `src/creditboost/search.py`, add the import of `Sequence` and `CandidateResult`, then:

```python
class BaselineMissingError(ValueError):
    """The baseline candidate is absent from the frontier, or failed to train."""


def select(
    candidates: Sequence[CandidateResult],
    baseline: str,
    auc_budget: float = config.MAX_AUC_SACRIFICE,
    min_improvement: float = config.MIN_AIR_IMPROVEMENT,
) -> str:
    """The name of the candidate to adopt.

    Among candidates whose ROC-AUC is within `auc_budget` of the best scored
    candidate, take the highest minimum adverse impact ratio; then keep the
    baseline unless the winner beats it by more than `min_improvement`.

    Every candidate here was scored at the same matched approval rate, so the
    comparison is like-for-like by construction. Ties break by AUC and then by
    declaration order, which makes the result deterministic under a fixed seed.
    """
    scored = [c for c in candidates if c.failed_reason is None]
    order = {candidate.name: position for position, candidate in enumerate(candidates)}

    base = next((c for c in scored if c.name == baseline), None)
    if base is None:
        raise BaselineMissingError(
            f"baseline {baseline!r} is not among the scored candidates. The "
            "baseline must always train: without it there is nothing to "
            "compare an alternative against."
        )

    best_auc = max(c.roc_auc for c in scored)  # type: ignore[type-var]
    eligible = [c for c in scored if c.roc_auc >= best_auc - auc_budget]  # type: ignore[operator]

    winner = min(
        eligible,
        key=lambda c: (-c.min_adverse_impact_ratio, -c.roc_auc, order[c.name]),  # type: ignore[operator]
    )

    improvement = winner.min_adverse_impact_ratio - base.min_adverse_impact_ratio  # type: ignore[operator]
    if improvement <= min_improvement:
        return base.name
    return winner.name
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_search.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite and the linters**

Run: `.venv/bin/python -m pytest -m "not slow" -q && ruff check . && ruff format --check . && mypy src/ && lint-imports`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/creditboost/search.py src/creditboost/config.py tests/test_search.py
git commit -m "feat: add the selection rule

Highest min AIR within MAX_AUC_SACRIFICE (0.01) of the best candidate,
then a noise guard: the winner must beat the baseline by more than
MIN_AIR_IMPROVEMENT (0.01) or the baseline is kept.

Both constants are measured rather than chosen. AIR's bootstrap standard
deviation is about 0.005 on the full validation split, so an improvement
of 0.01 or less is not distinguishable from noise, and a search without
the guard would churn the shipped model on it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Ranking at a matched approval rate

**Files:**
- Modify: `src/creditboost/search.py`
- Modify: `src/creditboost/config.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `search.apply`, `search.CANDIDATES` (Task 3); `fairness.adverse_impact_ratios` (Task 2); `schema.CandidateResult` (Task 1).
- Produces: `search.matched_adverse_mask(probabilities: np.ndarray, approval_rate: float) -> np.ndarray`; `search.Ranking(target_approval_rate: float, candidates: list[CandidateResult])`; `search.rank(train_frame: pd.DataFrame, seed: int = config.RANDOM_SEED, min_group_size: int = config.MIN_FAIRNESS_GROUP_SIZE) -> Ranking`; `config.SELECTION_SIZE = 0.25`.

`rank` takes the **training** frame only. It is structurally incapable of touching the validation split, which is what keeps selection bias out of the shipped ratio.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_search.py`:

```python
import numpy as np

from creditboost.search import Ranking, matched_adverse_mask, rank


def test_the_matched_mask_hits_the_requested_approval_rate():
    probabilities = np.linspace(0.0, 1.0, 1000)
    mask = matched_adverse_mask(probabilities, approval_rate=0.75)
    assert abs((~mask).mean() - 0.75) < 0.01


def test_a_more_lenient_candidate_gains_nothing_from_being_lenient():
    """THE load-bearing test of this milestone.

    Candidate B's probabilities are candidate A's, halved. B ranks applicants
    identically -- it is the same model wearing a lower scale -- so it must
    score identically. At a FIXED threshold it would not: B approves everyone
    and its ratio would read near 1.0, and a search ranking on that number
    would reliably select whichever candidate approves the most people.
    """
    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 0.4, size=1000)
    b = a * 0.5

    target = float((a <= config.RISK_BAND_LOW_MAX).mean())

    mask_a = matched_adverse_mask(a, target)
    mask_b = matched_adverse_mask(b, target)

    assert np.array_equal(mask_a, mask_b)

    # And the naive alternative really would have differed, which is what makes
    # this test meaningful rather than vacuous.
    assert not np.array_equal(a > config.RISK_BAND_LOW_MAX, b > config.RISK_BAND_LOW_MAX)


def test_the_matched_mask_is_invariant_to_any_monotone_rescaling():
    rng = np.random.default_rng(1)
    probabilities = rng.uniform(0.01, 0.99, size=500)
    rescaled = probabilities**2  # strictly increasing on (0, 1)
    assert np.array_equal(
        matched_adverse_mask(probabilities, 0.6), matched_adverse_mask(rescaled, 0.6)
    )


@pytest.mark.slow
def test_rank_scores_every_candidate_and_puts_the_baseline_first(fixture_path):
    frame = pd.read_csv(fixture_path)
    ranking = rank(frame, min_group_size=10)

    assert isinstance(ranking, Ranking)
    assert [c.name for c in ranking.candidates][0] == BASELINE.name
    assert len(ranking.candidates) == len(CANDIDATES)
    assert 0.0 <= ranking.target_approval_rate <= 1.0
    for candidate in ranking.candidates:
        assert (candidate.failed_reason is None) != (candidate.roc_auc is None)


@pytest.mark.slow
def test_rank_is_deterministic(fixture_path):
    frame = pd.read_csv(fixture_path)
    first = rank(frame, min_group_size=10)
    second = rank(frame, min_group_size=10)
    assert first == second
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_search.py -k "matched or lenient or monotone" -v`
Expected: FAIL with `ImportError: cannot import name 'matched_adverse_mask'`.

- [ ] **Step 3: Add the config constant**

In `src/creditboost/config.py`, in the alternative search section:

```python
# The selection split, as a fraction OF THE TRAINING SPLIT -- not of the frame.
# Selection nests inside training so the validation split never participates in
# it: a winner chosen on validation data would carry an optimistically biased
# ratio into the artifact. The cost is that candidates train on 0.6 of the
# frame rather than 0.8, which makes their absolute AUCs slightly pessimistic;
# only their ranking is consumed, and every candidate shares the handicap.
SELECTION_SIZE = 0.25
```

- [ ] **Step 4: Implement the ranking**

In `src/creditboost/search.py`, add the remaining imports (`numpy as np`, `xgboost as xgb`, `roc_auc_score`, `data.split`, `fairness.adverse_impact_ratios`) and:

```python
RANKING_BASIS = "matched approval rate on the selection split"


@dataclass(frozen=True)
class Ranking:
    """A scored frontier, before any selection has been made."""

    target_approval_rate: float
    candidates: list[CandidateResult]


def matched_adverse_mask(probabilities: np.ndarray, approval_rate: float) -> np.ndarray:
    """The adverse mask that approves `approval_rate` of these applicants.

    Comparing candidates at a fixed threshold measures how lenient each one is,
    not how fair. A candidate whose probabilities are simply lower approves more
    people and its ratio drifts toward 1.0 for that reason alone -- measured, a
    single-feature model read 0.984 at a fixed threshold and 0.930 at a matched
    rate. Taking a quantile of each candidate's own predictions removes the
    scale entirely, so the comparison is of who is ranked adversely, not of how
    many.
    """
    threshold = float(np.quantile(probabilities, approval_rate))
    return probabilities > threshold


def _fit(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    score_features: pd.DataFrame,
    score_labels: pd.Series,
    params: Mapping[str, object],
) -> np.ndarray:
    """Train one candidate and return its predictions on the scoring frame.

    Early stopping watches the SCORING frame, not the training one. Watching the
    training frame would never stop early -- training loss keeps improving -- so
    every candidate would run the full NUM_BOOST_ROUND and be ranked on an
    overfit tail. This mirrors exactly what train.fit does with its validation
    split, one level down.

    train.py's constants are imported lazily because train.py imports this
    module.
    """
    from .train import EARLY_STOPPING_ROUNDS, NUM_BOOST_ROUND, PARAMS

    dtrain = xgb.DMatrix(train_features, label=train_labels, enable_categorical=True)
    dscore = xgb.DMatrix(score_features, label=score_labels, enable_categorical=True)
    booster = xgb.train(
        {**PARAMS, **params},
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dscore, "select")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    booster = booster[: booster.best_iteration + 1]
    return np.asarray(booster.predict(dscore))


def rank(
    train_frame: pd.DataFrame,
    seed: int = config.RANDOM_SEED,
    min_group_size: int = config.MIN_FAIRNESS_GROUP_SIZE,
) -> Ranking:
    """Score every candidate on a split nested inside the training frame.

    This function receives the TRAINING frame and nothing else. It is therefore
    structurally incapable of touching the validation split, which is what keeps
    selection bias out of the ratio the artifact reports.
    """
    inner_train, selection = split(train_frame, seed=seed, validation_size=config.SELECTION_SIZE)
    labels = inner_train[config.TARGET_COLUMN]
    truth = selection[config.TARGET_COLUMN]

    baseline_probabilities = _fit(
        apply(BASELINE, inner_train),
        labels,
        apply(BASELINE, selection),
        truth,
        BASELINE.params,
    )
    target_approval_rate = float(
        (baseline_probabilities <= config.RISK_BAND_LOW_MAX).mean()
    )

    results: list[CandidateResult] = []
    for spec in CANDIDATES:
        try:
            features = apply(spec, inner_train)
            if features.empty or not len(features.columns):
                raise ValueError("no features remain after drops")
            probabilities = (
                baseline_probabilities
                if spec is BASELINE
                else _fit(features, labels, apply(spec, selection), truth, spec.params)
            )
            mask = matched_adverse_mask(probabilities, target_approval_rate)
            attributes = adverse_impact_ratios(selection, mask.tolist(), min_group_size)
            ratios = {
                a.attribute: a.adverse_impact_ratio
                for a in attributes
                if a.adverse_impact_ratio is not None
            }
            if not ratios:
                raise ValueError(
                    "no protected attribute could be measured on the selection split"
                )
            results.append(
                CandidateResult(
                    name=spec.name,
                    n_features=len(features.columns),
                    roc_auc=float(roc_auc_score(truth, probabilities)),
                    min_adverse_impact_ratio=min(ratios.values()),
                    adverse_impact_ratios=ratios,
                )
            )
        except Exception as error:  # noqa: BLE001 - recorded, never swallowed
            # A candidate that could not be scored is recorded, not skipped. A
            # missing candidate would misrepresent the breadth of the search.
            results.append(
                CandidateResult(name=spec.name, n_features=0, failed_reason=str(error))
            )

    return Ranking(target_approval_rate=target_approval_rate, candidates=results)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_search.py -v`
Expected: PASS, including the two `slow` tests.

- [ ] **Step 6: Run the full suite and the linters**

Run: `.venv/bin/python -m pytest -q && ruff check . && ruff format --check . && mypy src/ && lint-imports`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/creditboost/search.py src/creditboost/config.py tests/test_search.py
git commit -m "feat: rank candidates at a matched approval rate

rank() splits the training frame again and scores every candidate at a
threshold matched to the baseline's approval rate, so a candidate cannot
win by being lenient. Measured, a single-feature model reads 0.984 at a
fixed threshold and 0.930 at a matched rate; a search ranking on the
former would reliably pick whichever model approves the most people.

rank() receives the training frame only, so it is structurally incapable
of touching the validation split the artifact reports its ratio on.

A candidate that cannot be scored is recorded with a reason rather than
skipped: a missing candidate misrepresents the breadth of the search.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Wire the search into training, and add `creditboost-search`

**Files:**
- Modify: `src/creditboost/train.py`
- Create: `src/creditboost/search_cli.py`
- Modify: `pyproject.toml`
- Test: `tests/test_train.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `search.rank`, `search.select`, `search.Ranking`, `search.CANDIDATES`, `search.BASELINE`, `search.RANKING_BASIS`, `search.apply`; `schema.SearchReport`.
- Produces: `train.fit(train_frame, valid_frame, spec=search.BASELINE)`; `creditboost-train --search`; the `creditboost-search` console script.

**Why a non-baseline winner refuses to write.** Adopting a candidate is a reviewed code change, never something a training run does to itself: a winner with `drops` disagrees with `config.FEATURE_ORDER` and the skew gate rejects it; a winner with `collapses` was trained on levels serving would pass through unmapped; a winner with `params` is not reproducible from `train.PARAMS`. So `--search` prints the frontier and the change required, and writes nothing. The operator promotes the winner to the code's baseline and re-runs.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_train.py`:

```python
from creditboost import search
from creditboost.schema import ModelMetadata


def test_fit_accepts_a_candidate_spec(fixture_path):
    """A spec with drops produces a model over fewer features."""
    frame = load_training_frame(fixture_path)
    train_frame, valid_frame = split(frame)
    spec = search.CandidateSpec(name="fewer", drops=("CNT_CHILDREN",))
    booster, _ = fit(train_frame, valid_frame, spec=spec)
    assert "CNT_CHILDREN" not in (booster.feature_names or [])


@pytest.mark.slow
def test_search_stamps_a_report_naming_a_real_candidate(tmp_path, fixture_path):
    model_out = tmp_path / "model.json"
    metadata_out = tmp_path / "model_meta.json"
    code = main(
        [
            "--data", str(fixture_path),
            "--model-out", str(model_out),
            "--metadata-out", str(metadata_out),
            "--provenance", "fixture",
            "--search",
        ]
    )
    assert code == 0
    metadata = ModelMetadata.model_validate_json(metadata_out.read_text())
    assert metadata.selection is not None
    names = {c.name for c in metadata.selection.candidates}
    assert metadata.selection.selected in names
    assert metadata.selection.baseline == search.BASELINE.name
    assert metadata.selection.ranking_basis == search.RANKING_BASIS


@pytest.mark.slow
def test_the_stamped_fairness_report_is_the_one_measured_at_the_band_threshold(
    tmp_path, fixture_path
):
    """The search scores candidates at a matched threshold of its own. That
    number must never become the artifact's fairness report, which is measured
    at config.RISK_BAND_LOW_MAX on the validation split."""
    model_out = tmp_path / "model.json"
    metadata_out = tmp_path / "model_meta.json"
    main(
        [
            "--data", str(fixture_path),
            "--model-out", str(model_out),
            "--metadata-out", str(metadata_out),
            "--provenance", "fixture",
            "--search",
        ]
    )
    metadata = ModelMetadata.model_validate_json(metadata_out.read_text())
    assert metadata.fairness.band_low_max == config.RISK_BAND_LOW_MAX
    assert metadata.fairness.adverse_definition == "band != low"


def test_without_search_no_selection_report_is_written(tmp_path, fixture_path):
    model_out = tmp_path / "model.json"
    metadata_out = tmp_path / "model_meta.json"
    code = main(
        [
            "--data", str(fixture_path),
            "--model-out", str(model_out),
            "--metadata-out", str(metadata_out),
            "--provenance", "fixture",
        ]
    )
    assert code == 0
    metadata = ModelMetadata.model_validate_json(metadata_out.read_text())
    assert metadata.selection is None


def test_a_non_baseline_winner_writes_nothing(tmp_path, fixture_path, monkeypatch, capsys):
    """Adopting a candidate is a reviewed code change. Training must refuse and
    say what to change, not silently ship a model whose feature set disagrees
    with the code."""
    model_out = tmp_path / "model.json"
    metadata_out = tmp_path / "model_meta.json"

    monkeypatch.setattr(
        "creditboost.train.search.rank",
        lambda frame, **kwargs: search.Ranking(
            target_approval_rate=0.75,
            candidates=[
                CandidateResult(
                    name=search.BASELINE.name,
                    n_features=20,
                    roc_auc=0.75,
                    min_adverse_impact_ratio=0.81,
                    adverse_impact_ratios={"DAYS_BIRTH": 0.81},
                ),
                CandidateResult(
                    name="no-occupation",
                    n_features=19,
                    roc_auc=0.75,
                    min_adverse_impact_ratio=0.95,
                    adverse_impact_ratios={"DAYS_BIRTH": 0.95},
                ),
            ],
        ),
    )

    code = main(
        [
            "--data", str(fixture_path),
            "--model-out", str(model_out),
            "--metadata-out", str(metadata_out),
            "--provenance", "fixture",
            "--search",
        ]
    )
    assert code == 1
    assert not model_out.exists()
    assert not metadata_out.exists()
    assert "no-occupation" in capsys.readouterr().err
```

Add `from creditboost.schema import CandidateResult` to the module's imports, and
`load_training_frame`, `split` and `fit` if the module does not already import them.

One more test in `tests/test_train.py`, proving the split isolation is real and
not merely intended:

```python
def test_ranking_never_sees_the_validation_split(fixture_path, monkeypatch):
    """rank() takes the training frame and nothing else. If a later refactor
    ever hands it the whole frame, this fails."""
    frame = load_training_frame(fixture_path)
    train_frame, valid_frame = split(frame)
    seen = {}

    def spy(passed_frame, **kwargs):
        seen["n"] = len(passed_frame)
        return search.Ranking(
            target_approval_rate=0.75,
            candidates=[
                CandidateResult(
                    name=search.BASELINE.name,
                    n_features=20,
                    roc_auc=0.75,
                    min_adverse_impact_ratio=0.81,
                    adverse_impact_ratios={"DAYS_BIRTH": 0.81},
                )
            ],
        )

    monkeypatch.setattr("creditboost.train.search.rank", spy)
    main([...])  # the same argument list as the test above, with --search

    assert seen["n"] == len(train_frame)
    assert seen["n"] != len(frame)
```

And in `tests/test_api.py`, mirroring the existing
`test_metadata_exposes_the_fairness_report`:

```python
def test_metadata_exposes_the_search_report(client):
    """The endpoint dumps the whole ModelMetadata, so the frontier arrives with
    no endpoint change. It contains aggregate model statistics only, never
    applicant data, and a service claiming to have searched should publish the
    search."""
    body = client.get("/metadata").json()
    assert "selection" in body
```

The `artifact_paths` fixture in `tests/test_api.py` builds a fixture-provenance
`ModelMetadata` with no selection report, so `body["selection"]` is `None` there;
the assertion is that the key is present and the endpoint did not change shape.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_train.py -k "spec or search or selection or non_baseline" -v`
Expected: FAIL — `fit()` takes no `spec`, and `main` has no `--search`.

- [ ] **Step 3: Teach `fit` about specs**

In `src/creditboost/train.py`, replace `_matrix` and adjust `fit`:

```python
from . import config, search
from .search import BASELINE, CandidateSpec


def _matrix(frame: pd.DataFrame, spec: CandidateSpec = BASELINE) -> xgb.DMatrix:
    return xgb.DMatrix(
        search.apply(spec, frame),
        label=frame[config.TARGET_COLUMN],
        enable_categorical=True,
    )


def fit(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    spec: CandidateSpec = BASELINE,
) -> tuple[xgb.Booster, dict[str, float]]:
```

Inside `fit`, build the matrices with the spec and merge its params:

```python
    dtrain, dvalid = _matrix(train_frame, spec), _matrix(valid_frame, spec)

    booster = xgb.train(
        {**PARAMS, **spec.params},
        dtrain,
        ...
    )
```

`search.apply(BASELINE, frame)` is exactly `transform(frame)`, so the default path is unchanged — the existing tests prove it.

- [ ] **Step 4: Add `--search` to `main`**

After the argument parser gains:

```python
    parser.add_argument(
        "--search",
        action="store_true",
        help=(
            "Search for a less discriminatory alternative before training, and "
            "record the frontier in the artifact."
        ),
    )
```

Insert this block after the split and before `fit` is called:

```python
    selection = None
    spec = search.BASELINE
    if args.search:
        logger.info("searching %d candidates for a less discriminatory alternative", len(search.CANDIDATES))
        ranking = search.rank(train_frame)
        for candidate in ranking.candidates:
            if candidate.failed_reason is None:
                logger.info(
                    "  %-36s auc %.4f  min AIR %.4f",
                    candidate.name,
                    candidate.roc_auc,
                    candidate.min_adverse_impact_ratio,
                )
            else:
                logger.info("  %-36s failed: %s", candidate.name, candidate.failed_reason)

        chosen = search.select(ranking.candidates, baseline=search.BASELINE.name)
        if chosen != search.BASELINE.name:
            logger.error(
                "the search selected %r over the baseline; no artifact written. "
                "Adopting a candidate is a reviewed code change, not something a "
                "training run may do to itself: its feature set, categorical "
                "levels or hyperparameters must be reflected in config.py, "
                "features.py and train.PARAMS, or the train/serve skew gate "
                "would be describing a model that no longer exists. Promote %r "
                "to the baseline in search.CANDIDATES, make the matching code "
                "change, and re-run.",
                chosen,
                chosen,
            )
            return 1

        selection = SearchReport(
            baseline=search.BASELINE.name,
            selected=chosen,
            auc_budget=config.MAX_AUC_SACRIFICE,
            min_air_improvement=config.MIN_AIR_IMPROVEMENT,
            target_approval_rate=ranking.target_approval_rate,
            ranking_basis=search.RANKING_BASIS,
            candidates=ranking.candidates,
        )
```

Pass `spec` to `fit(train_frame, valid_frame, spec)` and `selection=selection` into `ModelMetadata`. Import `SearchReport` from `.schema`.

- [ ] **Step 5: Create the read-only CLI**

Create `src/creditboost/search_cli.py`:

```python
"""The creditboost-search command.

Read-only by design: it prints the frontier and what the selection rule would
choose, and writes nothing. Search and training deliberately do not communicate
through a file on disk -- a stored frontier could be stamped onto a model it did
not select -- so this command exists purely for inspection, and
`creditboost-train --search` is the path that produces an artifact.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import config, search
from .data import load_training_frame, split

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="creditboost-search")
    parser.add_argument("--data", type=Path, default=config.DEFAULT_DATA_PATH)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    frame = load_training_frame(args.data)
    train_frame, _ = split(frame)
    ranking = search.rank(train_frame)

    print(f"ranked at a matched approval rate of {ranking.target_approval_rate:.4f}")
    print(f"{'candidate':38s} {'AUC':>8s} {'min AIR':>9s}  features")
    for candidate in ranking.candidates:
        if candidate.failed_reason is None:
            print(
                f"{candidate.name:38s} {candidate.roc_auc:8.4f} "
                f"{candidate.min_adverse_impact_ratio:9.4f}  {candidate.n_features}"
            )
        else:
            print(f"{candidate.name:38s} {'--':>8s} {'--':>9s}  {candidate.failed_reason}")

    chosen = search.select(ranking.candidates, baseline=search.BASELINE.name)
    print(f"\nselection rule would choose: {chosen}")
    if chosen == search.BASELINE.name:
        print("no less discriminatory alternative was found within the AUC budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

In `pyproject.toml`, register it:

```toml
[project.scripts]
creditboost-train = "creditboost.train:main"
creditboost-artifact = "creditboost.artifact_cli:main"
creditboost-search = "creditboost.search_cli:main"
```

`search_cli` imports `data`, so it is training-side. It is not in the import-linter `source_modules` list, and must not be added to it.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_train.py -v`
Expected: PASS.

- [ ] **Step 7: Run the full suite and the linters**

Run: `.venv/bin/python -m pytest -q && ruff check . && ruff format --check . && mypy src/ && lint-imports`
Expected: all green. Re-run `pip install -e ".[train,dev]"` if `creditboost-search` is not on PATH.

- [ ] **Step 8: Commit**

```bash
git add src/creditboost/train.py src/creditboost/search_cli.py pyproject.toml tests/test_train.py tests/test_api.py
git commit -m "feat: run the search from training, and add creditboost-search

creditboost-train --search ranks candidates on a split nested inside the
training data and stamps the frontier into the artifact. The shipped
fairness report is still whatever fairness.evaluate produced at
RISK_BAND_LOW_MAX on the validation split; the search's matched-threshold
figure never reaches it, and a test proves that.

A non-baseline winner writes nothing and says what to change. Adopting a
candidate means a feature set, level mapping or parameter change that
config.py, features.py and train.PARAMS must reflect, or the train/serve
skew gate describes a model that no longer exists. That is a reviewed
code change, not something a training run does to itself.

creditboost-search prints the same frontier and writes nothing, so it can
be inspected before a release with no chance of producing an artifact.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Require the search of production artifacts

**Files:**
- Modify: `src/creditboost/artifact_cli.py`
- Test: `tests/test_artifact_cli.py` (extend `make_artifact` at line 23)

**Interfaces:**
- Consumes: `ModelMetadata.selection` (Task 1).
- Produces: `artifact_cli.SelectionError`.

**This task opens the build-red window.** `model-v0.3.0` has no selection report, so `docker build` fails from this commit until Task 8 lands.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_artifact_cli.py`, following the module's existing pattern for
building an artifact directory and lockfile:

First extend the module's existing `make_artifact` helper (currently at
`tests/test_artifact_cli.py:23`) with a `selection` keyword. Default it to
`a_search_report()` so every pre-existing production-provenance test keeps
passing unchanged, and add `from tests.conftest import a_search_report` beside
the existing `a_passing_fairness_report` import:

```python
def make_artifact(
    directory: Path,
    *,
    provenance: str = "production",
    version: str | None = None,
    metadata_feature_order: list[str] | None = None,
    booster_feature_names: list[str] | None = None,
    selection: SearchReport | None = _UNSET,
) -> tuple[Path, Path]:
```

`_UNSET` is a module-level sentinel (`_UNSET: Any = object()`) so that
`selection=None` can mean "write an artifact with no selection report", which is
exactly what the new test needs, while the default still supplies one:

```python
        provenance=provenance,  # type: ignore[arg-type]
        fairness=a_passing_fairness_report(),
        selection=a_search_report() if selection is _UNSET else selection,
    )
```

Then add the tests:

```python
from creditboost.artifact_cli import SelectionError


def test_verify_rejects_a_production_artifact_that_was_never_searched(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "models"
    make_artifact(directory, selection=None)
    with pytest.raises(SelectionError, match="never searched"):
        verify_artifact(directory, lock_for(directory, tmp_path))


def test_verify_accepts_a_production_artifact_carrying_a_search(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "models"
    make_artifact(directory)
    verify_artifact(directory, lock_for(directory, tmp_path))


def test_verify_still_accepts_a_fixture_artifact_without_a_search(
    tmp_path: Path,
) -> None:
    """A fixture artifact need not have been searched: the four-minute search is
    a production concern, and the provenance gate already keeps fixture models
    out of published images."""
    directory = tmp_path / "models"
    make_artifact(directory, provenance="fixture", selection=None)
    verify_artifact(directory, lock_for(directory, tmp_path), allow_fixture=True)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_artifact_cli.py -k "search" -v`
Expected: FAIL with `ImportError: cannot import name 'SelectionError'`.

- [ ] **Step 3: Implement the check**

In `src/creditboost/artifact_cli.py`, add the error class beside the others:

```python
class SelectionError(ArtifactError):
    """A production artifact carries no record of an alternative search."""
```

And in `verify_artifact`, immediately after the provenance check:

```python
    if metadata.provenance == "production" and metadata.selection is None:
        raise SelectionError(
            "artifact provenance is 'production' but it carries no selection "
            "report: this model was never searched for a less discriminatory "
            "alternative. Disparate impact is a burden-shifting doctrine, and "
            "business necessity rebuts a prima facie case only if no such "
            "alternative exists -- an unsearched model cannot support that "
            "claim. Retrain with `creditboost-train --search`."
        )
```

The check keys on `provenance` rather than on `--allow-fixture`, so it holds even
when a local experiment passes that flag with a production artifact.

`SearchReport`'s own validator already guarantees that `selected` and `baseline`
name candidates present in the frontier, so an artifact carrying an incoherent
report cannot be parsed at all — that half of the spec's check is structural and
needs no code here.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_artifact_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Confirm the red window is open, deliberately**

Run: `.venv/bin/python -m creditboost.artifact_cli verify` against the pinned artifact if `models/` is populated.
Expected: FAIL with `SelectionError`. This is correct — `model-v0.3.0` was never searched. Task 8 closes it.

- [ ] **Step 6: Run the full suite and the linters**

Run: `.venv/bin/python -m pytest -q && ruff check . && ruff format --check . && mypy src/ && lint-imports`
Expected: all green. The suite does not build the Docker image, so it stays green while `docker build` does not.

- [ ] **Step 7: Commit**

```bash
git add src/creditboost/artifact_cli.py tests/test_artifact_cli.py
git commit -m "feat: require a recorded search of production artifacts

verify now rejects a production artifact with no selection report, inside
the Docker builder, so an image containing an unsearched production model
cannot be built. Business necessity rebuts a prima facie disparate impact
case only if no less discriminatory alternative exists, and a model that
was never searched cannot support that claim.

BUILD RED: model-v0.3.0 carries no selection report, so docker build
fails from this commit until model-v0.4.0 is released. Do not merge this
branch before that task lands.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Retrain, release `model-v0.4.0`, close the window

**Files:**
- Modify: `src/creditboost/config.py` (`MODEL_VERSION`)
- Modify: `models/model.lock.json`

**Interfaces:**
- Consumes: everything above.
- Produces: a released, pinned, verified `model-v0.4.0`.

Requires `data/application_train.csv` and `gh` credentials. `MODEL_VERSION` and the
lockfile must move in the **same commit** — `verify` enforces it, so that `/health`
cannot report a version the artifact does not have.

- [ ] **Step 1: Inspect the frontier before committing to it**

Run: `creditboost-search --data data/application_train.csv`
Expected: about four minutes, then a frontier and `selection rule would choose: baseline`.

The expected reading, from the measurements taken while the spec was written:
baseline near AUC 0.753 and min AIR 0.810; `external-scores-only` near AUC 0.725
and min AIR 0.866, outside the 0.01 budget; `no-external-scores` markedly *worse*
on fairness, near 0.720.

**If it chooses anything other than `baseline`, stop and report.** That is a real
finding, not a hitch: it means an alternative cleared the budget, and adopting it
is a code change and a conversation, exactly as Task 6's refusal path describes.

- [ ] **Step 2: Bump `MODEL_VERSION`**

In `src/creditboost/config.py`: `MODEL_VERSION = "0.4.0"`.

- [ ] **Step 3: Train with the search**

Run: `creditboost-train --data data/application_train.csv --provenance production --search`
Expected: exit 0. The ratios should reproduce `model-v0.3.0`'s exactly — sex 0.868,
marital status 0.818, age 0.810 — because ranking reads only the training split and
the final fit is unchanged: same split, same `PARAMS`, same seed.

- [ ] **Step 4: Confirm the artifact carries the frontier**

Run:
```bash
.venv/bin/python -c "
from creditboost.schema import ModelMetadata
m = ModelMetadata.model_validate_json(open('models/model_meta.json').read())
print('version   ', m.version)
print('selected  ', m.selection.selected)
print('candidates', len(m.selection.candidates))
print('ratios    ', {a.attribute: a.adverse_impact_ratio for a in m.fairness.attributes})
"
```
Expected: version `0.4.0`, selected `baseline`, the full candidate count, and the
three ratios above.

- [ ] **Step 5: Release and pin**

Run: `./scripts/release-model.sh 0.4.0`
Expected: the release is published and `models/model.lock.json` is rewritten.

- [ ] **Step 6: Verify, and confirm the window is closed**

Run: `creditboost-artifact verify && docker build -t creditboost:dev . && ./scripts/smoke.sh http://localhost:8000`
Expected: verify passes, the image builds, and the smoke test passes. `docker build`
succeeding is the proof the red window opened in Task 7 is closed.

- [ ] **Step 7: Commit**

```bash
git add src/creditboost/config.py models/model.lock.json
git commit -m "feat: retrain with the alternative search, release model-v0.4.0

The search ranked every candidate at a matched approval rate on a split
nested inside the training data, and the baseline won: no alternative
cleared the 0.01 AUC budget by more than the 0.01 noise guard. That
negative result is recorded in the artifact, which is the point -- an
artifact that discarded it could not distinguish looked-and-stayed from
never-looked.

Ratios reproduce model-v0.3.0 exactly, as expected: ranking reads only the
training split and the final fit is unchanged.

MODEL_VERSION and the lockfile move together, and docker build is green
again.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update the roadmap in `CLAUDE.md`**

Add after the Milestone 4 entry:

```markdown
**Milestone 5 — less discriminatory alternative search.** Done. `creditboost-train --search`
ranks a catalog of model specifications at a matched approval rate on a split nested inside
the training data, applies a 0.01 AUC budget and a 0.01 noise guard, and stamps the whole
frontier into the artifact. `creditboost-artifact verify` refuses a production artifact that
carries no such record.

Measurement shaped every decision. Bootstrapping `model-v0.3.0` put the age ratio at 0.810
with sd 0.0046 and 1% of resamples already below the floor — it passed within noise, not
comfortably. Sixteen small perturbations spanned 0.8058 to 0.8146, entirely inside that
noise. The band threshold, by contrast, moves the ratio further than every model variant
combined, which is exactly why it is excluded as a search axis rather than exploited.

The one real finding: removing the external bureau scores makes fairness markedly *worse*
(0.810 → 0.720), so the disparity lives in the application-form features. `model-v0.4.0`
records a negative result — no alternative cleared the budget — which is the business
necessity evidence, not a failure.

- Design spec: `docs/superpowers/specs/2026-09-02-creditboost-less-discriminatory-alternative-design.md`
- Implementation plan: `docs/superpowers/plans/2026-09-02-creditboost-less-discriminatory-alternative.md`

**Still open:** remediation that actually moves the ratio, intersectional analysis, fairness
of the reason codes across groups, deployment, prediction persistence.
```

Also amend the Milestone 3 "Known open question" note: the `NAME_INCOME_TYPE` proxy levels
are now measured — they move the ratio by 0.0001 — and remain in the catalog as
`maternity-to-working` and `income-type-proxies-dropped` so that finding is re-established on
every search rather than remembered.

- [ ] **Step 2: Add the invariants to `CLAUDE.md`**

```markdown
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
```

- [ ] **Step 3: Update the commands section in `CLAUDE.md` and `README.md`**

```bash
creditboost-search --data data/application_train.csv     # print the frontier, write nothing
creditboost-train --data data/application_train.csv --provenance production --search
```

Note that the search adds roughly four minutes to a training run, at about 3–4 seconds
per candidate.

- [ ] **Step 4: Verify every claim**

Re-read both files against the code. Every number cited must be one actually measured, and
every filename and flag must exist.

Run: `.venv/bin/python -m pytest -q && ruff check . && ruff format --check . && mypy src/ && lint-imports`

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: record Milestone 5 and the search invariants

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```
