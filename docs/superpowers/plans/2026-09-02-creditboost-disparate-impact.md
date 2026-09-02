# Disparate Impact Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the adverse impact ratio of every trained model, record it in the artifact, and refuse to write one that fails the four-fifths rule.

**Architecture:** A new pure module `fairness.py` bands validation probabilities, groups them by protected attribute, and returns a `FairnessReport` of adverse impact ratios. `train.py` calls it after the AUC gate and refuses to write on failure; the report is stamped into `ModelMetadata`, which makes it visible at `/metadata` with no endpoint change. Computation and enforcement are separate functions so the gate can be tested without training anything.

**Tech Stack:** Python 3.12, pandas, pydantic v2, xgboost, pytest, import-linter.

**Spec:** `docs/superpowers/specs/2026-09-02-creditboost-disparate-impact-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Python 3.12 only.** On macOS, `brew install libomp` before `pip install`.
- **No new runtime dependency.** pandas and pydantic are already present.
- **The adverse outcome is `band != "low"`.** Not `band == "high"` — at a 97% approval rate the ratio on approvals cannot discriminate, and a 9.19× age disparity passes at 0.974.
- **The ratio is `min/max` over *favourable* rates.** Inverted, a failing model reads as passing. This is the single most dangerous error in the milestone.
- **`MIN_ADVERSE_IMPACT_RATIO` is 0.80 and is a gate, not a dial.** There is no override flag. If a model cannot pass, that is a conversation about the approach.
- **"Not measured" must never read as "passed."** Exactly one of `adverse_impact_ratio` and `unmeasured_reason` is ever set.
- **No member of `PROTECTED_ATTRIBUTES` becomes a model feature.** This milestone measures outcomes; it adds no input. `FEATURE_ORDER` stays at 20 entries and `REQUEST_FIELDS` at 19.
- **Never lower `config.MIN_VALIDATION_AUC`** (0.70).
- **No applicant financial field is ever logged**, and nothing new is logged here beyond aggregate group rates on a *failure* path.
- **`ruff` line-length 100**, rules `["E", "F", "I", "UP", "B"]`. Run `ruff check . && ruff format --check . && mypy src/ && lint-imports` before every commit.
- CI never downloads from Kaggle. Training stays manual, local and credentialed.

## Build-Red Window

**Task 4 breaks `docker build`, and Task 5 is the only thing that fixes it.**

Task 4 makes `ModelMetadata.fairness` required. The pinned `model-v0.2.0` has no such field, so `artifact.load()` rejects it and `creditboost-artifact verify` fails inside the Docker builder — correctly. Consequences:

- Between Task 4's commit and Task 5's, `docker build .` fails. Expected, and documented in Task 4's commit message.
- Task 5 requires `data/application_train.csv` (present on the development machine) and `gh` credentials.
- Task 6 depends only on Task 5. **The branch must not merge until Task 5 lands.**

Nothing affecting training changes in this milestone — same features, hyperparameters and seeded split — so the retrain is in substance a metadata refresh and the ratios should reproduce the figures already measured: sex 0.868, marital status 0.818, age 0.810.

## File Structure

| File | Responsibility |
|---|---|
| `src/creditboost/schema.py` | **Modify.** Add `GroupRate`, `AttributeFairness`, `FairnessReport`; later a required `fairness` field on `ModelMetadata`. |
| `src/creditboost/fairness.py` | **Create.** `evaluate` and `failing_attributes`. Imports `config`, `banding`, `schema` only. |
| `src/creditboost/config.py` | **Modify.** Add `FAIRNESS_ATTRIBUTES`, `MIN_FAIRNESS_GROUP_SIZE`, `MIN_ADVERSE_IMPACT_RATIO`, `ECOA_PROTECTED_AGE`; bump `MODEL_VERSION` to `0.3.0`. |
| `src/creditboost/data.py` | **Modify.** Require `FAIRNESS_ATTRIBUTES` in the training frame. |
| `src/creditboost/train.py` | **Modify.** Evaluate fairness after the AUC gate, refuse on failure, stamp the report. |
| `tests/fixtures/generate_fixture.py` | **Modify.** Emit `CODE_GENDER`, a third column category: analysis-only, never accepted from callers. |
| `tests/fixtures/sample.csv` | **Regenerate.** One added column; pre-existing columns verified unchanged. |
| `tests/conftest.py` | **Modify.** Add `a_passing_fairness_report()`, used by every test that builds a `ModelMetadata`. |
| `models/model.lock.json` | **Modify.** Repointed at `model-v0.3.0`. |
| `pyproject.toml` | **Modify.** Add `creditboost.fairness` to the import-linter contract. |
| `tests/test_fairness.py` | **Create.** The metric, driven by constructed frames. |
| `tests/test_schema.py`, `tests/test_artifact.py`, `tests/test_api.py`, `tests/test_artifact_cli.py` | **Modify.** Five `ModelMetadata` construction sites need the new field. |
| `tests/test_train.py`, `tests/test_fixture.py` | **Modify.** Gate behaviour and fixture columns. |
| `CLAUDE.md`, `README.md` | **Modify.** Invariant ledger, roadmap, `/metadata` documentation. |

---

### Task 1: The fairness schema

Types only. Nothing computes or consumes them yet, so this task is independently reviewable.

**Files:**
- Modify: `src/creditboost/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `creditboost.schema.GroupRate` (fields `group: str`, `adverse_rate: float`, `n: int`), `AttributeFairness` (`attribute: str`, `adverse_impact_ratio: float | None`, `unmeasured_reason: str | None`, `groups: list[GroupRate]`), `FairnessReport` (`adverse_definition: str`, `band_low_max: float`, `min_group_size: int`, `attributes: list[AttributeFairness]`). Tasks 2, 4 and 5 depend on every field name here.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schema.py`:

```python
def test_a_measured_attribute_is_accepted():
    from creditboost.schema import AttributeFairness, GroupRate

    attribute = AttributeFairness(
        attribute="CODE_GENDER",
        adverse_impact_ratio=0.87,
        groups=[
            GroupRate(group="F", adverse_rate=0.222, n=40561),
            GroupRate(group="M", adverse_rate=0.325, n=20940),
        ],
    )
    assert attribute.unmeasured_reason is None


def test_an_unmeasured_attribute_is_accepted():
    from creditboost.schema import AttributeFairness

    attribute = AttributeFairness(
        attribute="NAME_FAMILY_STATUS",
        unmeasured_reason="fewer than two groups reached the minimum size of 100",
    )
    assert attribute.adverse_impact_ratio is None


def test_an_attribute_cannot_be_both_measured_and_unmeasured():
    """'Not measured' reading as 'passed' is the failure that would make the
    whole report worthless, so the two states are mutually exclusive."""
    import pytest
    from pydantic import ValidationError

    from creditboost.schema import AttributeFairness

    with pytest.raises(ValidationError):
        AttributeFairness(
            attribute="CODE_GENDER",
            adverse_impact_ratio=0.87,
            unmeasured_reason="also unmeasured, somehow",
        )


def test_an_attribute_must_be_one_or_the_other():
    import pytest
    from pydantic import ValidationError

    from creditboost.schema import AttributeFairness

    with pytest.raises(ValidationError):
        AttributeFairness(attribute="CODE_GENDER")


def test_a_ratio_outside_zero_to_one_is_rejected():
    """min/max over favourable rates cannot exceed 1. A value above it means the
    ratio was computed upside down."""
    import pytest
    from pydantic import ValidationError

    from creditboost.schema import AttributeFairness

    with pytest.raises(ValidationError):
        AttributeFairness(attribute="CODE_GENDER", adverse_impact_ratio=1.23)


def test_a_fairness_report_round_trips_through_json():
    from creditboost.schema import AttributeFairness, FairnessReport, GroupRate

    report = FairnessReport(
        adverse_definition="band != low",
        band_low_max=0.10,
        min_group_size=100,
        attributes=[
            AttributeFairness(
                attribute="CODE_GENDER",
                adverse_impact_ratio=0.868,
                groups=[GroupRate(group="F", adverse_rate=0.222, n=40561)],
            )
        ],
    )

    assert FairnessReport.model_validate_json(report.model_dump_json()) == report
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_schema.py -k "fairness or attribute or ratio" -v`
Expected: FAIL with `ImportError: cannot import name 'AttributeFairness' from 'creditboost.schema'`

- [ ] **Step 3: Add the models**

In `src/creditboost/schema.py`, add `model_validator` to the pydantic import so the line reads:

```python
from pydantic import BaseModel, ConfigDict, Field, model_validator
```

Then add these classes immediately above `class ModelMetadata`:

```python
class GroupRate(BaseModel):
    """One group's outcome rate. `n` travels with the rate so a reader can judge
    whether a ratio near the threshold is signal or noise."""

    group: str
    adverse_rate: float = Field(ge=0, le=1)
    n: int = Field(ge=0)


class AttributeFairness(BaseModel):
    """One protected attribute's adverse impact ratio, or the reason there isn't
    one. Exactly one of the two is ever set: an attribute that could not be
    measured has established nothing, and must never read as having passed."""

    attribute: str
    adverse_impact_ratio: float | None = Field(default=None, ge=0, le=1)
    unmeasured_reason: str | None = None
    groups: list[GroupRate] = Field(default_factory=list)

    @model_validator(mode="after")
    def exactly_one_outcome(self) -> AttributeFairness:
        measured = self.adverse_impact_ratio is not None
        unmeasured = self.unmeasured_reason is not None
        if measured == unmeasured:
            raise ValueError(
                f"attribute {self.attribute!r} must be either measured "
                "(adverse_impact_ratio) or unmeasured (unmeasured_reason), never "
                "both and never neither"
            )
        return self


class FairnessReport(BaseModel):
    """Disparate impact across protected attributes, measured on the validation
    split at training time.

    band_low_max records the policy the measurement was taken under. Risk-band
    thresholds are business policy that changes without retraining, so a stored
    ratio can go stale; recording the threshold makes that discoverable rather
    than silent. When band policy moves, fairness must be re-measured.
    """

    adverse_definition: str
    band_low_max: float = Field(ge=0, le=1)
    min_group_size: int = Field(ge=1)
    attributes: list[AttributeFairness]
```

The `-> AttributeFairness` return annotation works unquoted because `schema.py` already has `from __future__ import annotations` at the top; confirm it is there rather than assuming.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_schema.py -v`
Expected: PASS, including the pre-existing tests.

- [ ] **Step 5: Lint and type-check**

Run: `ruff check . && ruff format --check . && mypy src/ && lint-imports`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/creditboost/schema.py tests/test_schema.py
git commit -m "feat: add the fairness report schema

An attribute is either measured or carries a reason it was not, never both and
never neither: 'not measured' reading as 'passed' is the failure that would
make the whole report worthless.

The ratio is bounded to [0, 1] because min/max over favourable rates cannot
exceed 1 -- a value above it means the ratio was computed upside down."
```

---

### Task 2: `fairness.py`, the measurement

**Files:**
- Create: `src/creditboost/fairness.py`
- Modify: `src/creditboost/config.py` (four constants)
- Modify: `pyproject.toml` (contract)
- Test: `tests/test_fairness.py`

**Interfaces:**
- Consumes: `schema.GroupRate`, `schema.AttributeFairness`, `schema.FairnessReport` from Task 1; `banding.risk_band`.
- Produces: `creditboost.fairness.evaluate(frame: pd.DataFrame, probabilities: Sequence[float], min_group_size: int = config.MIN_FAIRNESS_GROUP_SIZE) -> FairnessReport`, `creditboost.fairness.failing_attributes(report: FairnessReport, minimum: float = config.MIN_ADVERSE_IMPACT_RATIO) -> list[AttributeFairness]`, and `fairness.ADVERSE_DEFINITION: str`. Task 4 calls both functions with exactly these signatures.

- [ ] **Step 1: Add the config constants**

Append to `src/creditboost/config.py`:

```python
# --- Disparate impact ------------------------------------------------------

# Columns that must be present in training data so outcomes can be measured
# across protected groups. This overlaps PROTECTED_ATTRIBUTES but is a different
# statement and must not be merged with it: PROTECTED_ATTRIBUTES says what may
# never be a model feature, a prohibition; FAIRNESS_ATTRIBUTES says what must be
# available to measure, a requirement. CODE_GENDER is required here while
# remaining something the service never accepts from a caller.
FAIRNESS_ATTRIBUTES: tuple[str, ...] = (
    "CODE_GENDER",
    "DAYS_BIRTH",
    "NAME_FAMILY_STATUS",
)

# ECOA protects applicants aged 62 and over specifically, so age is bucketed at
# that line rather than by quantile.
ECOA_PROTECTED_AGE = 62

# A rate estimated from a handful of rows is too noisy to gate a release on.
MIN_FAIRNESS_GROUP_SIZE = 100

# The four-fifths rule. A gate, not a dial: see the AUC floor above.
MIN_ADVERSE_IMPACT_RATIO = 0.80
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_fairness.py`:

```python
import pandas as pd
import pytest

from creditboost import config
from creditboost.fairness import evaluate, failing_attributes


def frame_and_probabilities(
    groups: list[str],
    probabilities: list[float],
    attribute: str = "CODE_GENDER",
) -> tuple[pd.DataFrame, list[float]]:
    """Build a frame carrying every fairness attribute, varying only the one
    under test. The others are held constant so they fall to a single group and
    are recorded unmeasured, which keeps each test about one thing."""
    frame = pd.DataFrame(
        {
            "CODE_GENDER": "F",
            "DAYS_BIRTH": -12_000,
            "NAME_FAMILY_STATUS": "Married",
        },
        index=range(len(groups)),
    )
    frame[attribute] = groups
    return frame, probabilities


def rates(attribute_name: str, report) -> dict:
    found = [a for a in report.attributes if a.attribute == attribute_name]
    assert found, f"{attribute_name} missing from the report"
    return found[0]


def test_the_ratio_is_min_over_max_not_max_over_min():
    """The load-bearing test. Inverted, a model at 0.81 reads as 1.23 and every
    model passes the gate -- a silent, total failure of the whole milestone.

    Group A: 2 of 10 adverse -> favourable 0.8.  Group B: 6 of 10 adverse ->
    favourable 0.4.  min/max = 0.4/0.8 = 0.5, which must be < 1.
    """
    groups = ["A"] * 10 + ["B"] * 10
    probs = ([0.5] * 2 + [0.01] * 8) + ([0.5] * 6 + [0.01] * 4)
    frame, probabilities = frame_and_probabilities(groups, probs)

    report = evaluate(frame, probabilities, min_group_size=5)

    assert rates("CODE_GENDER", report).adverse_impact_ratio == pytest.approx(0.5)


def test_adverse_means_not_banded_low_not_merely_high():
    """A medium-banded applicant is adverse. Counting only `high` is the
    definition this design rejects: it cannot discriminate at a high approval
    rate. 0.15 bands medium under the default thresholds, never high."""
    groups = ["A"] * 10 + ["B"] * 10
    probs = ([0.15] * 10) + ([0.01] * 10)
    frame, probabilities = frame_and_probabilities(groups, probs)

    report = evaluate(frame, probabilities, min_group_size=5)

    measured = rates("CODE_GENDER", report)
    by_group = {g.group: g.adverse_rate for g in measured.groups}
    assert by_group["A"] == pytest.approx(1.0), "medium must count as adverse"
    assert by_group["B"] == pytest.approx(0.0)
    assert measured.adverse_impact_ratio == pytest.approx(0.0)


def test_groups_below_the_minimum_size_are_excluded():
    groups = ["A"] * 10 + ["B"] * 3
    probs = [0.01] * 13
    frame, probabilities = frame_and_probabilities(groups, probs)

    report = evaluate(frame, probabilities, min_group_size=5)

    measured = rates("CODE_GENDER", report)
    assert [g.group for g in measured.groups] == ["A"]
    assert measured.adverse_impact_ratio is None
    assert "minimum size" in measured.unmeasured_reason


def test_an_unmeasured_attribute_never_fails_the_gate():
    """Unmeasured establishes nothing either way. It must not be reported as a
    failure, and it must not be reported as a pass."""
    groups = ["A"] * 10 + ["B"] * 3
    probs = [0.01] * 13
    frame, probabilities = frame_and_probabilities(groups, probs)

    report = evaluate(frame, probabilities, min_group_size=5)

    assert failing_attributes(report) == []


def test_a_ratio_below_the_floor_is_reported_failing():
    groups = ["A"] * 10 + ["B"] * 10
    probs = ([0.5] * 5 + [0.01] * 5) + ([0.01] * 10)
    frame, probabilities = frame_and_probabilities(groups, probs)

    report = evaluate(frame, probabilities, min_group_size=5)

    failures = failing_attributes(report)
    assert [a.attribute for a in failures] == ["CODE_GENDER"]


def test_exactly_the_floor_passes():
    """0.80 is the threshold, and the comparison is strictly less-than, so a
    model landing exactly on it is not failed."""
    groups = ["A"] * 10 + ["B"] * 10
    probs = ([0.5] * 2 + [0.01] * 8) + ([0.01] * 10)
    frame, probabilities = frame_and_probabilities(groups, probs)

    report = evaluate(frame, probabilities, min_group_size=5)

    assert rates("CODE_GENDER", report).adverse_impact_ratio == pytest.approx(0.8)
    assert failing_attributes(report) == []


def test_age_is_bucketed_at_the_ecoa_line():
    """15 U.S.C. 1691(a)(1) protects applicants 62 and over, so that is the
    boundary -- not a quantile of whatever population happens to be present."""
    older = -int(70 * 365.25)
    younger = -int(30 * 365.25)
    groups = [older] * 10 + [younger] * 10
    frame, probabilities = frame_and_probabilities(
        groups, [0.01] * 20, attribute="DAYS_BIRTH"
    )

    report = evaluate(frame, probabilities, min_group_size=5)

    labels = {g.group for g in rates("DAYS_BIRTH", report).groups}
    assert labels == {"62 and over", "under 62"}


def test_a_missing_age_is_excluded_rather_than_bucketed_as_young():
    """NaN >= 62 is False, so a careless implementation files unknown ages under
    'under 62' and reports a rate for people it knows nothing about."""
    older = -int(70 * 365.25)
    groups = [older] * 10 + [None] * 10
    frame, probabilities = frame_and_probabilities(
        groups, [0.01] * 20, attribute="DAYS_BIRTH"
    )

    report = evaluate(frame, probabilities, min_group_size=5)

    measured = rates("DAYS_BIRTH", report)
    assert [g.group for g in measured.groups] == ["62 and over"]
    assert measured.groups[0].n == 10


def test_a_report_records_the_policy_it_was_measured_under():
    groups = ["A"] * 10 + ["B"] * 10
    frame, probabilities = frame_and_probabilities(groups, [0.01] * 20)

    report = evaluate(frame, probabilities, min_group_size=5)

    assert report.band_low_max == config.RISK_BAND_LOW_MAX
    assert report.adverse_definition == "band != low"
    assert report.min_group_size == 5


def test_every_fairness_attribute_appears_in_the_report():
    groups = ["A"] * 10 + ["B"] * 10
    frame, probabilities = frame_and_probabilities(groups, [0.01] * 20)

    report = evaluate(frame, probabilities, min_group_size=5)

    assert [a.attribute for a in report.attributes] == list(config.FAIRNESS_ATTRIBUTES)


def test_a_wholly_adverse_population_is_unmeasured_not_zero():
    """Every group entirely adverse makes max favourable 0 and the ratio 0/0.
    A degenerate model the gate cannot speak to, not a failure it can assert."""
    groups = ["A"] * 10 + ["B"] * 10
    frame, probabilities = frame_and_probabilities(groups, [0.9] * 20)

    report = evaluate(frame, probabilities, min_group_size=5)

    measured = rates("CODE_GENDER", report)
    assert measured.adverse_impact_ratio is None
    assert "favourable" in measured.unmeasured_reason
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_fairness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'creditboost.fairness'`

- [ ] **Step 4: Write the module**

Create `src/creditboost/fairness.py`:

```python
"""Disparate impact measurement.

Milestone 3 established that no protected attribute is a model feature. That is
a statement about inputs, and it is not fairness: neutral features act as
proxies, which is why disparate impact is a distinct doctrine from disparate
treatment. This module measures outcomes instead.

The adverse outcome is `band != "low"` -- the applicant is not auto-approved --
rather than `band == "high"`. At the shipped model's 97% approval rate, ratios
of approval rates compress toward 1.0 and cannot discriminate: an applicant
under 62 is denied at 9.19 times the rate of one aged 62 or over, and the
four-fifths test on approvals returns 0.974, a comfortable pass. Regulation B
defines adverse action to include credit granted on substantially different
terms, so a medium band is adverse action.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from . import config
from .banding import risk_band
from .schema import AttributeFairness, FairnessReport, GroupRate

ADVERSE_DEFINITION = "band != low"


def _age_buckets(frame: pd.DataFrame) -> pd.Series:
    """Bucket at the ECOA line. Unknown ages become NA rather than falling into
    the younger bucket: NaN >= 62 is False, which would silently report a rate
    for applicants whose age is not known."""
    years = -pd.to_numeric(frame["DAYS_BIRTH"], errors="coerce") / 365.25
    over = years >= config.ECOA_PROTECTED_AGE

    labels = pd.Series(pd.NA, index=frame.index, dtype="string")
    labels[over] = f"{config.ECOA_PROTECTED_AGE} and over"
    labels[~over & years.notna()] = f"under {config.ECOA_PROTECTED_AGE}"
    return labels


def _groups(frame: pd.DataFrame, attribute: str) -> pd.Series:
    if attribute == "DAYS_BIRTH":
        return _age_buckets(frame)
    return frame[attribute].astype("string")


def evaluate(
    frame: pd.DataFrame,
    probabilities: Sequence[float],
    min_group_size: int = config.MIN_FAIRNESS_GROUP_SIZE,
) -> FairnessReport:
    """Adverse impact ratio per protected attribute, over the rows in `frame`.

    min_group_size is a parameter rather than read from config directly so tests
    can lower it; train.py takes the configured default.
    """
    adverse = pd.Series(
        [risk_band(float(p)) != "low" for p in probabilities], index=frame.index
    )

    attributes: list[AttributeFairness] = []
    for attribute in config.FAIRNESS_ATTRIBUTES:
        table = pd.DataFrame({"group": _groups(frame, attribute), "adverse": adverse})
        table = table.dropna(subset=["group"])
        summary = table.groupby("group")["adverse"].agg(["mean", "size"])
        eligible = summary[summary["size"] >= min_group_size]

        rates = [
            GroupRate(group=str(name), adverse_rate=float(row["mean"]), n=int(row["size"]))
            for name, row in eligible.iterrows()
        ]

        if len(eligible) < 2:
            attributes.append(
                AttributeFairness(
                    attribute=attribute,
                    unmeasured_reason=(
                        "fewer than two groups reached the minimum size of "
                        f"{min_group_size}"
                    ),
                    groups=rates,
                )
            )
            continue

        favourable = 1.0 - eligible["mean"]
        if favourable.max() <= 0.0:
            attributes.append(
                AttributeFairness(
                    attribute=attribute,
                    unmeasured_reason=(
                        "no applicant in any group received the favourable outcome"
                    ),
                    groups=rates,
                )
            )
            continue

        # min over max, never the reverse. Inverted, a failing model reads as
        # passing and the gate silently permits everything.
        attributes.append(
            AttributeFairness(
                attribute=attribute,
                adverse_impact_ratio=float(favourable.min() / favourable.max()),
                groups=rates,
            )
        )

    return FairnessReport(
        adverse_definition=ADVERSE_DEFINITION,
        band_low_max=config.RISK_BAND_LOW_MAX,
        min_group_size=min_group_size,
        attributes=attributes,
    )


def failing_attributes(
    report: FairnessReport,
    minimum: float = config.MIN_ADVERSE_IMPACT_RATIO,
) -> list[AttributeFairness]:
    """Attributes whose measured ratio falls below the floor.

    An unmeasured attribute is never returned: it has established nothing, so it
    is neither a failure to assert nor a pass to claim.
    """
    return [
        attribute
        for attribute in report.attributes
        if attribute.adverse_impact_ratio is not None
        and attribute.adverse_impact_ratio < minimum
    ]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_fairness.py -v`
Expected: PASS, all 11 tests.

- [ ] **Step 6: Register the module in the architecture contract**

In `pyproject.toml`, add `"creditboost.fairness"` to `source_modules` in the `[[tool.importlinter.contracts]]` block, after `"creditboost.reasons"`.

Run: `lint-imports`
Expected: `Contracts: 1 kept, 0 broken.`

- [ ] **Step 7: Lint, type-check, run everything**

Run: `ruff check . && ruff format --check . && mypy src/ && pytest -q`
Expected: all clean, all passing.

- [ ] **Step 8: Commit**

```bash
git add src/creditboost/fairness.py src/creditboost/config.py tests/test_fairness.py pyproject.toml
git commit -m "feat: add disparate impact measurement

Adverse impact ratio per protected attribute, min/max over favourable rates,
with adverse defined as not-banded-low rather than banded-high. On approvals the
four-fifths test cannot discriminate at a 97% approval rate: an applicant under
62 is denied at 9.19x the rate of one aged 62 or over and the test returns
0.974. Reg B treats credit on substantially different terms as adverse action,
so a medium band is adverse.

Age buckets at the ECOA line of 62, and an unknown age is excluded rather than
filed as young -- NaN >= 62 is False, which would report a rate for applicants
whose age is not known.

Measurement and enforcement are separate functions so the gate can be tested
against constructed reports without training anything."
```

---

### Task 3: Make the attributes available

The training frame must carry `CODE_GENDER`, and so must the fixture. Nothing gates yet, so this task is safe to land on its own.

**Files:**
- Modify: `src/creditboost/data.py:26` (required columns)
- Modify: `tests/fixtures/generate_fixture.py`
- Regenerate: `tests/fixtures/sample.csv`
- Test: `tests/test_fixture.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: `config.FAIRNESS_ATTRIBUTES` from Task 2.
- Produces: a fixture carrying every fairness attribute, and a loader that refuses data without them. Task 4 depends on both.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fixture.py`:

```python
def test_fixture_has_every_fairness_attribute(frame):
    """CODE_GENDER is not a request field -- the service never accepts it -- but
    fairness measurement needs it, so the fixture carries it anyway."""
    for column in config.FAIRNESS_ATTRIBUTES:
        assert column in frame.columns


def test_fixture_gender_groups_are_both_large_enough_to_measure(frame):
    """A 200-row fixture split evenly gives 100 per group, which is exactly the
    default minimum. Drawing it randomly could land at 95/105 and make the
    integration test flaky, so the split is deterministic."""
    counts = frame["CODE_GENDER"].value_counts()
    assert set(counts.index) == {"F", "M"}
    assert counts.min() == 100
```

Append to `tests/test_data.py`:

```python
def test_training_data_without_a_fairness_attribute_is_rejected(tmp_path):
    """Fairness cannot be gated on attributes that are not there, so their
    absence is an error rather than a silent skip."""
    import pandas as pd
    import pytest

    from creditboost import config
    from creditboost.data import MissingColumnsError, load_training_frame

    frame = pd.read_csv(config.REPO_ROOT / "tests" / "fixtures" / "sample.csv")
    frame = frame.drop(columns=["CODE_GENDER"])
    path = tmp_path / "no_gender.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(MissingColumnsError, match="CODE_GENDER"):
        load_training_frame(path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_fixture.py tests/test_data.py -v`
Expected: the three new tests FAIL — the fixture has no `CODE_GENDER` column, and the loader does not require it.

- [ ] **Step 3: Require the attributes in the loader**

In `src/creditboost/data.py`, change:

```python
    required = {*config.REQUEST_FIELDS, config.TARGET_COLUMN}
```

to:

```python
    # FAIRNESS_ATTRIBUTES as well as REQUEST_FIELDS: training gates on disparate
    # impact, which cannot be measured without the attributes. CODE_GENDER is
    # required here while remaining something the service never accepts.
    required = {*config.REQUEST_FIELDS, *config.FAIRNESS_ATTRIBUTES, config.TARGET_COLUMN}
```

- [ ] **Step 4: Emit the analysis-only column from the generator**

In `tests/fixtures/generate_fixture.py`, add below `MONITORING_ONLY_LEVELS`:

```python
# Analysis-only columns: present in training data so outcomes can be measured
# across groups, never accepted from a caller, never a model feature. This is a
# third category, distinct from model features and from monitoring-only request
# fields, and CODE_GENDER is its first member.
ANALYSIS_ONLY_FIELDS: tuple[str, ...] = ("CODE_GENDER",)
```

Then replace the final return statement:

```python
    return frame[list(config.REQUEST_FIELDS) + [config.TARGET_COLUMN]]
```

with:

```python
    # Deterministic and exactly balanced rather than drawn: 200 rows split evenly
    # gives 100 per group, exactly the default minimum group size, so a random
    # draw landing at 95/105 would make fairness measurement on the fixture
    # flaky. Alternating by index also consumes no rng, so no existing column
    # shifts position in the stream.
    frame["CODE_GENDER"] = np.where(np.arange(N_ROWS) % 2 == 0, "F", "M")

    analysis_only = [c for c in ANALYSIS_ONLY_FIELDS if c not in config.REQUEST_FIELDS]
    return frame[list(config.REQUEST_FIELDS) + analysis_only + [config.TARGET_COLUMN]]
```

- [ ] **Step 5: Regenerate the fixture**

Run: `python tests/fixtures/generate_fixture.py`

- [ ] **Step 6: Verify the pre-existing columns are unchanged**

The Milestone 3 equivalent of this step asserted the columns were unchanged and was wrong. Verify rather than assume:

```bash
git show HEAD:tests/fixtures/sample.csv > /tmp/old_sample.csv
python - <<'PY'
import pandas as pd
old = pd.read_csv("/tmp/old_sample.csv")
new = pd.read_csv("tests/fixtures/sample.csv")
print("added:  ", sorted(set(new.columns) - set(old.columns)))
print("removed:", sorted(set(old.columns) - set(new.columns)))
shared = [c for c in old.columns if c in new.columns]
pd.testing.assert_frame_equal(old[shared], new[shared])
print(f"all {len(shared)} pre-existing columns identical")
PY
```

Expected: `added: ['CODE_GENDER']`, `removed: []`, and the assertion passes. If it does not, the new column consumed rng — move its assignment later in `build_fixture` until it does.

- [ ] **Step 7: Run the full suite**

Run: `pytest -v`
Expected: PASS. `test_generator_is_deterministic` in particular must still pass.

- [ ] **Step 8: Lint and commit**

```bash
ruff check . && ruff format --check . && mypy src/ && lint-imports
git add src/creditboost/data.py tests/fixtures/generate_fixture.py \
        tests/fixtures/sample.csv tests/test_fixture.py tests/test_data.py
git commit -m "feat: carry protected attributes into training data

CODE_GENDER is required in training data and still never accepted from a
caller. The fixture gains a third column category alongside model features and
monitoring-only request fields: analysis-only columns, present so outcomes can
be measured across groups.

The gender split is deterministic and exactly balanced rather than drawn. 200
rows split evenly gives exactly the default minimum group size, so a random
95/105 would make fairness measurement on the fixture flaky; alternating by
index also consumes no rng, so no existing column moved -- verified against the
previous fixture rather than assumed."
```

---

### Task 4: Require the report and gate the training run

**This task turns `docker build` red.** Task 5 is the only thing that fixes it.

**Files:**
- Modify: `src/creditboost/schema.py` (`ModelMetadata.fairness`)
- Modify: `src/creditboost/train.py:105-127` (the gate and the metadata)
- Modify: `tests/conftest.py` (shared helper)
- Modify: `tests/test_artifact.py:26`, `tests/test_schema.py:75`, `tests/test_schema.py:90`, `tests/test_api.py:25`, `tests/test_artifact_cli.py:47` (five `ModelMetadata` construction sites)
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `fairness.evaluate`, `fairness.failing_attributes` from Task 2; the fixture from Task 3.
- Produces: `ModelMetadata.fairness: FairnessReport`, required. Task 5 depends on it.

- [ ] **Step 1: Add the shared test helper**

Making `fairness` required breaks every test that builds a `ModelMetadata`. Rather than repeat a literal five times, add to `tests/conftest.py`:

```python
def a_passing_fairness_report():
    """A minimal, comfortably-passing report for tests that need a valid
    ModelMetadata but are not about fairness."""
    from creditboost import config
    from creditboost.schema import AttributeFairness, FairnessReport, GroupRate

    return FairnessReport(
        adverse_definition="band != low",
        band_low_max=config.RISK_BAND_LOW_MAX,
        min_group_size=config.MIN_FAIRNESS_GROUP_SIZE,
        attributes=[
            AttributeFairness(
                attribute="CODE_GENDER",
                adverse_impact_ratio=0.95,
                groups=[
                    GroupRate(group="F", adverse_rate=0.20, n=500),
                    GroupRate(group="M", adverse_rate=0.21, n=500),
                ],
            )
        ],
    )
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_train.py`:

```python
def test_training_stamps_a_fairness_report(fixture_path, tmp_path):
    """Every shipped model carries measured disparate impact -- that is what
    making the metadata field required buys."""
    import json

    from creditboost import config
    from creditboost.train import main

    model_path = tmp_path / "model.json"
    meta_path = tmp_path / "meta.json"
    code = main(
        [
            "--data", str(fixture_path),
            "--model-out", str(model_path),
            "--metadata-out", str(meta_path),
            "--provenance", "fixture",
        ]
    )

    assert code == 0
    report = json.loads(meta_path.read_text())["fairness"]
    assert report["band_low_max"] == config.RISK_BAND_LOW_MAX
    assert report["adverse_definition"] == "band != low"
    assert [a["attribute"] for a in report["attributes"]] == list(config.FAIRNESS_ATTRIBUTES)


def test_a_failing_ratio_writes_no_artifact(fixture_path, tmp_path, monkeypatch):
    """The gate, proven by forcing a failure rather than hoping for one. A real
    model that fails is exactly what this must refuse to write."""
    from creditboost import train
    from creditboost.schema import AttributeFairness, FairnessReport, GroupRate

    failing = FairnessReport(
        adverse_definition="band != low",
        band_low_max=0.10,
        min_group_size=100,
        attributes=[
            AttributeFairness(
                attribute="CODE_GENDER",
                adverse_impact_ratio=0.61,
                groups=[
                    GroupRate(group="F", adverse_rate=0.20, n=500),
                    GroupRate(group="M", adverse_rate=0.51, n=500),
                ],
            )
        ],
    )
    monkeypatch.setattr(train, "evaluate", lambda *a, **k: failing)

    model_path = tmp_path / "model.json"
    meta_path = tmp_path / "meta.json"
    code = train.main(
        [
            "--data", str(fixture_path),
            "--model-out", str(model_path),
            "--metadata-out", str(meta_path),
            "--provenance", "fixture",
        ]
    )

    assert code == 1
    assert not model_path.exists()
    assert not meta_path.exists()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_train.py -k "fairness or failing_ratio" -v`
Expected: FAIL — `test_training_stamps_a_fairness_report` with `KeyError: 'fairness'`, and the gate test because `train.evaluate` does not exist.

- [ ] **Step 4: Add the required metadata field**

In `src/creditboost/schema.py`, add to `ModelMetadata`, after `provenance`:

```python
    fairness: FairnessReport
```

Required rather than optional on purpose: an optional field makes "every shipped model has measured disparate impact" a hope rather than a guarantee.

- [ ] **Step 5: Fix the five construction sites**

Each of these builds a `ModelMetadata` and now needs the new field. In every one, add `fairness=a_passing_fairness_report(),` beside `provenance=...`, importing the helper at the top of the module with `from tests.conftest import a_passing_fairness_report`:

- `tests/test_artifact.py:26`
- `tests/test_schema.py:75` and `tests/test_schema.py:90`
- `tests/test_api.py:25`
- `tests/test_artifact_cli.py:47`

Run `pytest -q` after this step; every failure should now be about the gate, not about missing metadata.

- [ ] **Step 6: Assert `/metadata` exposes the report**

Success criterion 8 is that the report reaches `/metadata` with no endpoint change. That
happens for free — the handler returns `metadata.model_dump()` — but "for free" is exactly
what regresses unnoticed, so it gets a test. Append to `tests/test_api.py`:

```python
def test_metadata_exposes_the_fairness_report(client):
    """The endpoint dumps the whole ModelMetadata, so the report arrives with no
    endpoint change. A service that gates on fairness should publish what it
    measured; the block is aggregate group rates only, never applicant data."""
    report = client.get("/metadata").json()["fairness"]

    assert report["adverse_definition"] == "band != low"
    assert report["band_low_max"] == config.RISK_BAND_LOW_MAX
    assert "attributes" in report
```

- [ ] **Step 7: Wire the gate into training**

In `src/creditboost/train.py`, add the imports beside the existing ones:

```python
from .fairness import evaluate, failing_attributes
```

Then, in `main`, immediately after the AUC gate's `return 1` block and before `metadata = ModelMetadata(`:

```python
    # fit() does not return validation predictions, so they are recomputed here.
    # One extra pass over the validation split is cheap next to changing fit()'s
    # signature, which tests and callers depend on.
    fairness = evaluate(valid_frame, booster.predict(_matrix(valid_frame)))
    failures = failing_attributes(fairness)
    if failures:
        for attribute in failures:
            logger.error(
                "adverse impact ratio %.4f for %s is below the floor %.2f; "
                "no artifact written",
                attribute.adverse_impact_ratio,
                attribute.attribute,
                config.MIN_ADVERSE_IMPACT_RATIO,
            )
            for group in attribute.groups:
                logger.error(
                    "    %s: adverse rate %.4f (n=%d)",
                    group.group,
                    group.adverse_rate,
                    group.n,
                )
        return 1

    logger.info(
        "adverse impact ratios: %s",
        {a.attribute: a.adverse_impact_ratio for a in fairness.attributes},
    )
```

and add `fairness=fairness,` to the `ModelMetadata(...)` call beside `provenance=args.provenance`.

The group rates logged on the failure path are aggregate statistics, never applicant data, and they appear only when a run is being refused — the operator needs them to understand why.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `pytest -v`
Expected: PASS.

- [ ] **Step 9: Confirm the build is now red, for the right reason**

Run: `docker build -t creditboost:m4-check . 2>&1 | tail -20`
Expected: **failure**, with a pydantic validation error naming `fairness` as a missing field while loading `model-v0.2.0`. Confirm the message names `fairness` — a failure for any other reason means something else is wrong.

- [ ] **Step 10: Lint and commit**

```bash
ruff check . && ruff format --check . && mypy src/ && lint-imports
git add src/creditboost/schema.py src/creditboost/train.py tests/conftest.py \
        tests/test_train.py tests/test_artifact.py tests/test_schema.py \
        tests/test_api.py tests/test_artifact_cli.py
git commit -m "feat: gate training on disparate impact

Training measures the adverse impact ratio on the validation split after the
AUC gate and refuses to write a model failing any measured attribute, logging
each group's rate and size so the refusal is actionable. There is no override
flag, for the same reason the AUC floor has none.

ModelMetadata.fairness is required rather than optional: optional would make
'every shipped model has measured disparate impact' a hope rather than a
guarantee. Five test construction sites take a shared conftest helper.

NOTE: docker build fails from this commit until model-v0.3.0 is released. That
is artifact.load correctly refusing a model-v0.2.0 artifact that carries no
fairness report."
```

---

### Task 5: Retrain, release `model-v0.3.0`, close the window

Requires `data/application_train.csv` and `gh` credentials. Both are present.

**Files:**
- Modify: `src/creditboost/config.py:8` (`MODEL_VERSION`)
- Modify: `models/model.lock.json`

**Interfaces:**
- Consumes: the gate from Task 4; `scripts/release-model.sh` from Milestone 2.
- Produces: release `model-v0.3.0`, a lockfile pinning it, `MODEL_VERSION == "0.3.0"`.

- [ ] **Step 1: Bump `MODEL_VERSION` before training**

In `src/creditboost/config.py`, change `MODEL_VERSION = "0.2.0"` to `MODEL_VERSION = "0.3.0"`.

The order matters. `train.py` stamps `config.MODEL_VERSION` into the metadata, and `scripts/release-model.sh` refuses to publish when the metadata version disagrees with the tag. Training first produces an artifact stamped `0.2.0` that the release script rejects, costing a full retrain.

- [ ] **Step 2: Train**

Run: `creditboost-train --data data/application_train.csv --provenance production`

Expected: it logs the metrics, then a line of adverse impact ratios, then writes both files. Nothing about training changed this milestone, so expect `roc_auc` at 0.75307 and ratios near sex 0.868, marital 0.818, age 0.810.

**If the fairness gate refuses the model, stop and report.** Do not lower `MIN_ADVERSE_IMPACT_RATIO`, and do not add an override. A refusal here is a finding about the model, and the margin on age is one point, so it is a real possibility.

- [ ] **Step 3: Record the figures**

```bash
python -c "
import json
meta = json.load(open('models/model_meta.json'))
print('version: ', meta['version'])
print('roc_auc: ', round(meta['metrics']['roc_auc'], 5))
print('adverse definition:', meta['fairness']['adverse_definition'])
print('band_low_max:      ', meta['fairness']['band_low_max'])
for a in meta['fairness']['attributes']:
    ratio = a['adverse_impact_ratio']
    shown = f\"{ratio:.4f}\" if ratio is not None else f\"unmeasured ({a['unmeasured_reason']})\"
    print(f\"  {a['attribute']:22} {shown}\")
"
```

Write these figures into Step 6's commit message. Do not write a number you have not read.

- [ ] **Step 4: Publish the release**

Run: `./scripts/release-model.sh 0.3.0`

Expected: the release is created with both assets and the script prints `wrote models/model.lock.json for model-v0.3.0`.

- [ ] **Step 5: Prove the build is green again**

```bash
git diff models/model.lock.json
gh release view model-v0.3.0 --json assets -q '[.assets[] | "\(.name)  \(.digest)"] | .[]'
docker build -t creditboost:m4 .
docker run -d --rm --name cb-m4 -p 8000:8000 creditboost:m4
./scripts/smoke.sh http://localhost:8000
curl -fsS http://localhost:8000/metadata | python -m json.tool | head -40
docker stop cb-m4
```

Expected: the lockfile digests match the published asset digests exactly; the build log shows `fetched model-v0.3.0` and `verified against model-v0.3.0`; the smoke test passes; `/metadata` shows the fairness report with no endpoint change.

- [ ] **Step 6: Commit**

```bash
git add src/creditboost/config.py models/model.lock.json
git commit -m "feat: retrain with disparate impact measured, release model-v0.3.0

Adverse impact ratios <FIGURES>, all clear of the 0.80 floor. roc_auc <FIGURE>.

Nothing affecting training changed this milestone -- same features,
hyperparameters and seeded split -- so this is in substance a metadata refresh,
and the ratios reproduce what was measured against model-v0.2.0.

Closes the build-red window opened by the previous commit. model-v0.2.0 remains
released, though it can no longer be loaded by this code, which is the skew gate
working as intended."
```

Replace `<FIGURES>` and `<FIGURE>` with the values read in Step 3.

---

### Task 6: Documentation

**Files:**
- Modify: `CLAUDE.md` (Repository state, Roadmap, Invariants)
- Modify: `README.md` (`/metadata` section, project layout)

- [ ] **Step 1: Update `CLAUDE.md`'s Invariants**

Add after the `PROTECTED_ATTRIBUTES` bullet:

```markdown
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
```

- [ ] **Step 2: Update `CLAUDE.md`'s Repository state and Roadmap**

Change the Repository state line to cover Milestones 1 through 4, and add to the Roadmap:

```markdown
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

**Still open:** the `NAME_INCOME_TYPE` proxy levels recorded in Milestone 3 are now
measurable but not resolved; remediation of any disparity found, intersectional analysis,
and fairness of the reason codes across groups are all unspecced.
```

- [ ] **Step 3: Update `README.md`**

In the `GET /metadata` section, note that the response includes a `fairness` block, with a real excerpt:

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

Add a sentence: the ratio is the four-fifths adverse impact ratio measured on the validation split at training time, adverse meaning the applicant was not auto-approved; a model below 0.80 on any measured attribute cannot be trained.

In Project layout, add to the `src/creditboost/` listing:

```
  fairness.py            # adverse impact ratios; the training-time gate
```

Replace the figures in the excerpt with the real ones from Task 5 Step 3.

- [ ] **Step 4: Verify the documented example**

Run the container and confirm `/metadata` returns a `fairness` block matching the README's shape and figures. A documented example that does not run is worse than none.

- [ ] **Step 5: Run the milestone's success criteria**

Work through the ten criteria at the end of this plan and confirm each:

```bash
pytest -v
ruff check . && ruff format --check . && mypy src/ && lint-imports
docker build -t creditboost:m4 .
```

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: record Milestone 4 and the disparate impact invariants

Documents why the four-fifths rule is applied to not-banded-low rather than to
approvals: at a 97% approval rate the conventional form passes a 9.19x age
disparity at 0.974.

Records the band-policy rule as an explicit documentation-only invariant rather
than leaving the gap implicit."
```

---

## Success Criteria

Verify each before considering the milestone done. Each maps to the spec's criterion of the same number.

- [ ] 1. `creditboost-train` computes a ratio per `FAIRNESS_ATTRIBUTES` entry and refuses to write a model failing any measured one.
- [ ] 2. The ratio is `min/max` over favourable rates with adverse `band != "low"`, proven by a test that fails when the direction is inverted.
- [ ] 3. An attribute with fewer than two qualifying groups is recorded unmeasured and never satisfies the gate.
- [ ] 4. `ModelMetadata.fairness` is required; an artifact lacking it cannot load.
- [ ] 5. The report records `band_low_max`, `adverse_definition`, `min_group_size`, and each group's rate and size.
- [ ] 6. `model-v0.3.0` is trained, released, pinned, and verified inside `docker build`, with `MODEL_VERSION` at `0.3.0`.
- [ ] 7. The retrained model clears the AUC floor and the 0.80 ratio on every measured attribute, with the figures recorded in Task 5's commit message.
- [ ] 8. `/metadata` exposes the report with no endpoint change.
- [ ] 9. The fixture carries `CODE_GENDER`, and its pre-existing columns are verified unchanged.
- [ ] 10. No new runtime dependency; `ruff`, `mypy`, `lint-imports` and the full suite are clean.
