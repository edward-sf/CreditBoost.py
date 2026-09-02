# Adverse Action Reason Codes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Return up to four plain-language reasons on every `/predict`, and stop scoring on marital status.

**Architecture:** XGBoost's `pred_contribs=True` gives an exact per-feature contribution row. Those contributions are summed into ten concepts, filtered to the adverse direction, ranked, and rendered through a curated catalog in `config.py`. A new pure module `reasons.py` does the ranking with plain dictionaries; `serve/app.py` handles the array at the edge. Separately, `NAME_FAMILY_STATUS` stops being a model feature and becomes an accepted-but-never-modelled field, guarded by a new `PROTECTED_ATTRIBUTES` set.

**Tech Stack:** Python 3.12, xgboost (`pred_contribs`), pydantic v2, pandas, FastAPI, pytest, import-linter.

**Spec:** `docs/superpowers/specs/2026-09-02-creditboost-adverse-action-reason-codes-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Python 3.12 only.** On macOS, `brew install libomp` before `pip install`, or `import xgboost` fails.
- **No new runtime dependency.** `pred_contribs` is built into the xgboost already in the image. Do **not** add the `shap` package.
- **`FEATURE_ORDER` ends this milestone with exactly 20 entries; `REQUEST_FIELDS` with exactly 19.**
- **No protected attribute is ever a model feature.** `PROTECTED_ATTRIBUTES = ("CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS")` and `FEATURE_ORDER` must stay disjoint. The transform still *reads* `DAYS_BIRTH` to derive `employed_to_age`; that is the sanctioned exception and it is about features, not reads.
- **No reason text may name or imply age, sex, or marital status.** Enforced by a word-boundary test in Task 3.
- **Missing values are never imputed.** NaN reaches XGBoost intact.
- **Never lower `config.MIN_VALIDATION_AUC`** (0.70) to force a model through.
- **No applicant financial field may ever be logged,** and this milestone logs nothing new.
- **`models/model.lock.json` and `config.MODEL_VERSION` move in the same commit.**
- **`ruff` line-length 100**, rules `["E", "F", "I", "UP", "B"]`. Run `ruff check . && ruff format --check . && mypy src/ && lint-imports` before every commit.
- CI never downloads from Kaggle. Training stays a manual, local, credentialed step.

## Build-Red Window

**Task 1 breaks `docker build`, and Task 2 is the only thing that fixes it.**

Task 1 takes `FEATURE_ORDER` to 20 features while the pinned release `model-v0.1.0` still contains a 21-feature model. `creditboost-artifact verify` runs inside the Docker builder and compares the artifact's `feature_order` against the code's, so the build fails there — correctly. That is the Milestone 2 skew gate doing exactly its job, not a regression.

Consequences, which every executor must understand before starting:

- Between Task 1's commit and Task 2's, `docker build .` fails. This is expected and documented in Task 1's commit message.
- Task 2 requires the Kaggle dataset, which is **not on the development machine** and must be downloaded manually.
- Tasks 3–5 depend only on Task 1, so they may proceed while Task 2 is pending. **The branch must not merge until Task 2 lands.**

## File Structure

| File | Responsibility |
|---|---|
| `src/creditboost/config.py` | **Modify.** Add `PROTECTED_ATTRIBUTES`, `MONITORING_ONLY_FIELDS`, `ALWAYS_PRESENT_FEATURES`, `ReasonText`, `REASON_CONCEPTS`, `REASON_TEXT`, `MAX_REASONS`. Remove `NAME_FAMILY_STATUS` from `CATEGORICAL_LEVELS`; rewire `REQUEST_FIELDS` so it no longer follows `CATEGORICAL_FEATURES` alone. Bump `MODEL_VERSION` to `0.2.0`. |
| `src/creditboost/reasons.py` | **Create.** `principal_reasons` plus concept summing and wording selection. Pure — imports `config` and `schema` only. |
| `src/creditboost/schema.py` | **Modify.** Add the `Reason` model; add `reasons` to `PredictResponse`. `PredictRequest` is unchanged. |
| `src/creditboost/serve/app.py` | **Modify.** Keep the frame from `transform`, add the `pred_contribs` call, build the mappings, call `principal_reasons`. |
| `models/model.lock.json` | **Modify.** Repointed at `model-v0.2.0` by `scripts/release-model.sh`. |
| `pyproject.toml` | **Modify.** Add `creditboost.reasons` to the import-linter contract. |
| `tests/test_config.py` | **Modify.** Amend the count test; add the protected-attribute, partition, catalog and forbidden-wording tests. |
| `tests/test_features.py` | **Modify.** Assert marital status is absent from the transform output. |
| `tests/test_reasons.py` | **Create.** The pure module, driven by plain dictionaries. |
| `tests/test_api.py` | **Modify.** End-to-end reasons, plus the margin-reconstruction test. |
| `CLAUDE.md`, `README.md` | **Modify.** Invariant ledger, roadmap, API documentation. |

---

### Task 1: Quarantine marital status and drop to 20 features

The compliance fix. `NAME_FAMILY_STATUS` stops being a model feature and becomes an accepted-but-never-modelled field.

**The coupling that makes this non-trivial:** `config.py:96` defines `CATEGORICAL_FEATURES = tuple(CATEGORICAL_LEVELS)`, and **both** `FEATURE_ORDER` and `REQUEST_FIELDS` derive from it. Removing `NAME_FAMILY_STATUS` from `CATEGORICAL_LEVELS` therefore drops it from the request schema too, which is the one outcome this milestone explicitly rejected. The fix is a separate `MONITORING_ONLY_FIELDS` tuple appended to `REQUEST_FIELDS`, mirroring how `DAYS_BIRTH` is already appended there.

**Files:**
- Modify: `src/creditboost/config.py:58-64` (remove the `NAME_FAMILY_STATUS` entry from `CATEGORICAL_LEVELS`), `src/creditboost/config.py:111-113` (rewire `REQUEST_FIELDS`)
- Modify: `tests/test_config.py:7-8` (count 21 → 20)
- Test: `tests/test_config.py`, `tests/test_features.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `config.PROTECTED_ATTRIBUTES: tuple[str, ...]`, `config.MONITORING_ONLY_FIELDS: tuple[str, ...]`, and a 20-entry `config.FEATURE_ORDER`. Tasks 2 and 3 both depend on the final 20-feature list.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_protected_attributes_are_never_model_features():
    """ECOA / Regulation B, 15 U.S.C. 1691(a)(1): race, color, religion, national
    origin, sex or marital status, or age. This is the general form of the rule
    the codebase previously stated only for CODE_GENDER and raw DAYS_BIRTH."""
    overlap = set(config.PROTECTED_ATTRIBUTES) & set(config.FEATURE_ORDER)
    assert not overlap, f"protected attribute(s) used as model features: {sorted(overlap)}"


def test_marital_status_is_accepted_but_never_modelled():
    """Collected so fair-lending monitoring stays possible -- Reg B 1002.13
    requires exactly this for dwelling-secured credit -- and never scored on."""
    assert "NAME_FAMILY_STATUS" in config.REQUEST_FIELDS
    assert "NAME_FAMILY_STATUS" not in config.FEATURE_ORDER
    assert "NAME_FAMILY_STATUS" not in config.CATEGORICAL_LEVELS


def test_monitoring_only_fields_are_all_protected_attributes():
    """A field accepted but not modelled needs a reason to exist. The only
    sanctioned reason is that it is a protected attribute kept for monitoring."""
    for name in config.MONITORING_ONLY_FIELDS:
        assert name in config.PROTECTED_ATTRIBUTES


def test_request_fields_is_the_concatenation_of_its_parts():
    """REQUEST_FIELDS deliberately no longer tracks CATEGORICAL_FEATURES alone:
    a field can be accepted without being modelled."""
    expected = (
        config.NUMERIC_FEATURES
        + config.BINARY_FEATURES
        + config.CATEGORICAL_FEATURES
        + ("DAYS_BIRTH",)
        + config.MONITORING_ONLY_FIELDS
    )
    assert config.REQUEST_FIELDS == expected
```

Append to `tests/test_features.py`, directly below `test_raw_age_is_not_in_the_output`:

```python
def test_marital_status_is_not_in_the_output():
    """base_record() still supplies NAME_FAMILY_STATUS, exactly as a real caller
    may. The transform must ignore it: accepted is not the same as modelled."""
    assert "NAME_FAMILY_STATUS" not in transform([base_record()]).columns
```

Do **not** remove `"NAME_FAMILY_STATUS": "Married"` from `base_record()` in `tests/test_features.py:27`. Its presence is what makes the test above able to fail.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_config.py tests/test_features.py -v`
Expected: `test_protected_attributes_are_never_model_features` FAILS with `AttributeError: module 'creditboost.config' has no attribute 'PROTECTED_ATTRIBUTES'`, and the other new tests fail similarly. Do not proceed until you have seen them fail.

- [ ] **Step 3: Remove the feature from `CATEGORICAL_LEVELS`**

In `src/creditboost/config.py`, delete the entire `"NAME_FAMILY_STATUS": (...)` entry from the `CATEGORICAL_LEVELS` dict — the key and all six levels (`"Single / not married"`, `"Married"`, `"Civil marriage"`, `"Widow"`, `"Separated"`, `"Unknown"`).

`CATEGORICAL_FEATURES` derives from `tuple(CATEGORICAL_LEVELS)`, so it drops from 6 to 5 entries and `FEATURE_ORDER` from 21 to 20 automatically. No other change is needed to make the feature disappear from the transform: `features.py` iterates `config.CATEGORICAL_LEVELS`.

- [ ] **Step 4: Add the protected-attribute declarations**

In `src/creditboost/config.py`, immediately above the `REQUEST_FIELDS` definition, add:

```python
# ECOA, 15 U.S.C. 1691(a)(1), names the prohibited bases in one sentence: race,
# color, religion, national origin, sex or marital status, or age. None of them
# may be a model feature.
#
# The rule is about features, not reads: the transform still reads DAYS_BIRTH to
# derive employed_to_age. Regulation B allows age in an empirically derived,
# demonstrably and statistically sound scoring system, provided an elderly
# applicant is never assigned a negative factor for it. Marital status has no
# equivalent allowance, which is why NAME_FAMILY_STATUS is not a feature.
PROTECTED_ATTRIBUTES: tuple[str, ...] = (
    "CODE_GENDER",
    "DAYS_BIRTH",
    "NAME_FAMILY_STATUS",
)

# Accepted from callers and retained so disparate-impact analysis remains
# possible on live traffic, but never transformed and never scored on. Reg B
# 1002.13 requires collecting certain protected attributes for exactly this
# monitoring purpose. An attribute a service refuses to accept cannot be
# collected retroactively; a feature can always be restored by retraining.
MONITORING_ONLY_FIELDS: tuple[str, ...] = ("NAME_FAMILY_STATUS",)
```

- [ ] **Step 5: Rewire `REQUEST_FIELDS`**

Replace the existing definition at `src/creditboost/config.py:111-113`:

```python
# DAYS_BIRTH is accepted from callers to derive employed_to_age, but raw age is
# deliberately not a model feature: age is a protected basis under ECOA.
REQUEST_FIELDS: tuple[str, ...] = (
    NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + ("DAYS_BIRTH",)
)
```

with:

```python
# Deliberately no longer a function of CATEGORICAL_FEATURES alone. Two fields are
# accepted without being modelled: DAYS_BIRTH, consumed only to derive
# employed_to_age, and everything in MONITORING_ONLY_FIELDS.
REQUEST_FIELDS: tuple[str, ...] = (
    NUMERIC_FEATURES
    + BINARY_FEATURES
    + CATEGORICAL_FEATURES
    + ("DAYS_BIRTH",)
    + MONITORING_ONLY_FIELDS
)
```

Counts after this change: `NUMERIC_FEATURES` 10 + `BINARY_FEATURES` 2 + `CATEGORICAL_FEATURES` 5 + `DERIVED_FEATURES` 3 = **20** features; 10 + 2 + 5 + 1 + 1 = **19** request fields.

- [ ] **Step 6: Amend the count test**

In `tests/test_config.py`, change:

```python
def test_feature_order_has_exactly_21_entries():
    assert len(config.FEATURE_ORDER) == 21
```

to:

```python
def test_feature_order_has_exactly_20_entries():
    assert len(config.FEATURE_ORDER) == 20
```

Leave `test_request_fields_has_exactly_19_entries` exactly as it is. It now guards the quarantine: if someone later drops the field, that test fails.

- [ ] **Step 7: Run the full suite**

Run: `pytest -v`
Expected: PASS. `tests/test_schema.py:25` asserts `set(PredictRequest.model_fields) == set(config.REQUEST_FIELDS)` — it passes only because the pydantic field was kept, so a green run here is itself evidence the quarantine works. `tests/test_artifact.py` builds boosters sized from `len(config.FEATURE_ORDER)`, so it follows the change automatically.

If anything fails, read it before changing it. A failure here means something depended on marital status being a feature, which is worth understanding rather than patching.

- [ ] **Step 8: Lint and type-check**

Run: `ruff check . && ruff format --check . && mypy src/ && lint-imports`
Expected: all clean.

- [ ] **Step 9: Commit**

```bash
git add src/creditboost/config.py tests/test_config.py tests/test_features.py
git commit -m "feat: stop scoring on marital status, drop to 20 features

Marital status is an enumerated ECOA prohibited basis, 15 U.S.C. 1691(a)(1),
named in the same sentence as sex and age. The codebase excluded those two and
used NAME_FAMILY_STATUS as model feature 16, with Widow and Separated among its
scoring levels. Milestone 3's reason codes forced the issue: disclose it and the
service states a prohibited basis drove the decision; suppress it and the notice
is untruthful.

The field is still accepted and now declared MONITORING_ONLY_FIELDS. Reg B
1002.13 requires collecting protected attributes for monitoring, and an
attribute a service refuses to accept cannot be collected retroactively, whereas
a feature can always be restored by retraining. REQUEST_FIELDS therefore stops
deriving from CATEGORICAL_FEATURES alone.

NOTE: docker build fails from this commit until the retrained model-v0.2.0 is
released. That is the M2 skew gate correctly refusing a 21-feature artifact
against 20-feature code."
```

---

### Task 2: Retrain, release `model-v0.2.0`, and close the build-red window

**This task requires the Kaggle dataset and `gh` credentials.** It is the only manual, credentialed task in the milestone, and it is what makes `docker build` work again.

**Files:**
- Modify: `src/creditboost/config.py:8` (`MODEL_VERSION`)
- Modify: `models/model.lock.json` (written by the release script)

**Interfaces:**
- Consumes: the 20-feature `FEATURE_ORDER` from Task 1; `scripts/release-model.sh` and `creditboost-artifact` from Milestone 2.
- Produces: release `model-v0.2.0`, a lockfile pinning it, and `config.MODEL_VERSION == "0.2.0"`.

- [ ] **Step 1: Bump `MODEL_VERSION` before training, not after**

In `src/creditboost/config.py`, change `MODEL_VERSION = "0.1.0"` to `MODEL_VERSION = "0.2.0"`.

**The order matters and is easy to get backwards.** `train.py` stamps `config.MODEL_VERSION` into the metadata sidecar it writes, and `scripts/release-model.sh` refuses to publish when the metadata's version disagrees with the tag you asked for. Training before the bump produces an artifact stamped `0.1.0` that the release script will reject, forcing a full retrain.

- [ ] **Step 2: Obtain the dataset**

Download `application_train.csv` from Kaggle's [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) competition and place it at `data/application_train.csv`. The path is gitignored. CI never sees this file.

Confirm it is the expected file before training:

```bash
python -c "
import pandas as pd
frame = pd.read_csv('data/application_train.csv')
print('rows:', len(frame))
print('has TARGET:', 'TARGET' in frame.columns)
print('has NAME_FAMILY_STATUS:', 'NAME_FAMILY_STATUS' in frame.columns)
"
```

Expected: 307511 rows, both `True`. The second matters: the column must still exist in the source data, because that is what makes this milestone's feature removal reversible.

- [ ] **Step 3: Train**

Run: `creditboost-train --data data/application_train.csv --provenance production`

Expected: it writes `models/model.json` and `models/model_meta.json` and prints the validation metrics.

- [ ] **Step 4: Record the accuracy against the baseline**

Run:

```bash
python -c "
import json
meta = json.load(open('models/model_meta.json'))
print('version: ', meta['version'])
print('features:', len(meta['feature_order']))
print('metrics: ', meta['metrics'])
print('marital status present:', 'NAME_FAMILY_STATUS' in meta['feature_order'])
"
```

Expected: version `0.2.0`, 20 features, `marital status present: False`.

Compare `roc_auc` against the Milestone 1 baseline of **0.7533** and write the figure into the commit message in Step 7.

**If `roc_auc` is below `config.MIN_VALIDATION_AUC` (0.70), the training CLI has already refused to write anything. Do not lower the floor.** Stop and report: a 0.07 AUC drop from removing one weak categorical would indicate something wrong with the data or the transform, not a genuine cost of the compliance fix.

- [ ] **Step 5: Publish the release**

Run: `./scripts/release-model.sh 0.2.0`

Expected: `gh` creates the release with both assets, then the script prints `wrote models/model.lock.json for model-v0.2.0`.

If the script refuses with "artifact version is 0.1.0 but you asked to release 0.2.0", Step 1 was skipped or run after training. Bump `MODEL_VERSION`, retrain, and try again.

- [ ] **Step 6: Prove the build is green again**

```bash
docker build -t creditboost:m3 .
docker run -d --rm --name cb-m3 -p 8000:8000 creditboost:m3
./scripts/smoke.sh http://localhost:8000
curl -fsS http://localhost:8000/metadata
docker stop cb-m3
```

Expected: the build log shows `fetched model-v0.2.0` and `artifact in /build/models verified against model-v0.2.0`; the smoke test passes; `/metadata` reports 20 features and no `NAME_FAMILY_STATUS`.

- [ ] **Step 7: Commit**

```bash
git add src/creditboost/config.py models/model.lock.json
git commit -m "feat: retrain without marital status, release model-v0.2.0

20 features, validation roc_auc <FIGURE> against the 0.7533 baseline of the
21-feature model-v0.1.0.

Closes the build-red window opened by the previous commit. model-v0.1.0 remains
released and downloadable, so reverting is a three-line lockfile diff."
```

Replace `<FIGURE>` with the actual value from Step 4. Do not write a number you have not read.

---

### Task 3: The reason catalog

Pure configuration and its guards. No behaviour changes; nothing calls this yet.

**Files:**
- Modify: `src/creditboost/config.py` (append the catalog)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: the 20-entry `config.FEATURE_ORDER` from Task 1.
- Produces: `config.ReasonText` (a `NamedTuple` with fields `code: str`, `unfavourable: str`, `absent: str | None`), `config.REASON_CONCEPTS: dict[str, tuple[str, ...]]`, `config.REASON_TEXT: dict[str, ReasonText]`, `config.ALWAYS_PRESENT_FEATURES: tuple[str, ...]`, `config.MAX_REASONS: int`. Task 4 depends on every one of these names.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
import re


def test_reason_concepts_partition_feature_order_exactly():
    """The invariant that rots silently when the feature list changes: a feature
    added without a concept would simply never be reportable, and nothing else
    would notice."""
    mapped = [name for features in config.REASON_CONCEPTS.values() for name in features]

    assert len(mapped) == len(set(mapped)), "a feature appears in more than one concept"
    assert set(mapped) == set(config.FEATURE_ORDER), (
        f"unmapped features: {sorted(set(config.FEATURE_ORDER) - set(mapped))}; "
        f"unknown names in the map: {sorted(set(mapped) - set(config.FEATURE_ORDER))}"
    )


def test_every_concept_has_reason_text():
    assert set(config.REASON_TEXT) == set(config.REASON_CONCEPTS)


def test_reason_codes_are_unique():
    codes = [text.code for text in config.REASON_TEXT.values()]
    assert len(codes) == len(set(codes))


def test_absent_text_exists_exactly_where_it_is_reachable():
    """A concept containing an always-present feature can never be fully absent,
    so absent text there would be unreachable and would drift unnoticed."""
    for concept, features in config.REASON_CONCEPTS.items():
        can_be_absent = not (set(features) & set(config.ALWAYS_PRESENT_FEATURES))
        has_absent = config.REASON_TEXT[concept].absent is not None
        assert can_be_absent == has_absent, (
            f"concept {concept!r}: reachable={can_be_absent} but absent text "
            f"{'present' if has_absent else 'missing'}"
        )


def test_always_present_features_really_are_model_features():
    for name in config.ALWAYS_PRESENT_FEATURES:
        assert name in config.FEATURE_ORDER


def test_no_reason_text_names_or_implies_a_protected_attribute():
    """A reason code is a disclosure to the applicant. Saying their age, sex or
    marital status counted against them is the exact harm the feature set was
    cleaned to prevent -- the wording must not reintroduce it."""
    forbidden = (
        "age",
        "aged",
        "elderly",
        "young",
        "old",
        "birth",
        "sex",
        "gender",
        "male",
        "female",
        "marital",
        "married",
        "unmarried",
        "spouse",
        "widow",
        "widowed",
        "divorced",
        "separated",
    )
    pattern = re.compile(r"\b(" + "|".join(forbidden) + r")\b", re.IGNORECASE)

    for concept, text in config.REASON_TEXT.items():
        for field in (text.unfavourable, text.absent):
            if field is None:
                continue
            match = pattern.search(field)
            assert match is None, (
                f"concept {concept!r} text names a protected attribute "
                f"({match.group(0)!r}): {field!r}"
            )


def test_max_reasons_is_four():
    """Reg B's commentary treats more than four principal reasons as unhelpful."""
    assert config.MAX_REASONS == 4
```

Note the `import re` goes at the top of the file with the other imports, not inline.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `AttributeError: module 'creditboost.config' has no attribute 'REASON_CONCEPTS'`.

- [ ] **Step 3: Add the catalog**

Append to the end of `src/creditboost/config.py`:

```python
# --- Adverse action reason codes -------------------------------------------
#
# Business policy, like the risk-band thresholds: this catalog changes without
# retraining, so it lives here rather than in the artifact.


class ReasonText(NamedTuple):
    """One concept's disclosure wording. `absent` is None where the concept can
    never be fully missing, because every such text would be unreachable."""

    code: str
    unfavourable: str
    absent: str | None = None


MAX_REASONS = 4

# Features whose request fields are required and non-derived, so they can never
# be NaN. AMT_INCOME_TOTAL and AMT_CREDIT are required with gt=0, which also
# makes credit_to_income always computable.
ALWAYS_PRESENT_FEATURES: tuple[str, ...] = (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "credit_to_income",
)

# Contributions are summed within a concept before ranking. Ranking per feature
# instead would let one idea -- three external bureau scores, say -- fill all
# four slots by saying the same thing four times.
#
# Grouping also closes an ECOA trap: employed_to_age is the one feature through
# which age enters the model, and it can never surface on its own because it is
# only ever reported as `employment`.
REASON_CONCEPTS: dict[str, tuple[str, ...]] = {
    "external_credit": ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"),
    "loan_size": ("AMT_CREDIT", "AMT_GOODS_PRICE", "credit_to_income"),
    "repayment_burden": ("AMT_ANNUITY", "annuity_to_income"),
    "employment": ("DAYS_EMPLOYED", "employed_to_age"),
    "employment_profile": ("NAME_INCOME_TYPE", "OCCUPATION_TYPE"),
    "assets": ("FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_HOUSING_TYPE"),
    "household": ("CNT_CHILDREN", "CNT_FAM_MEMBERS"),
    "income": ("AMT_INCOME_TOTAL",),
    "education": ("NAME_EDUCATION_TYPE",),
    "product": ("NAME_CONTRACT_TYPE",),
}

REASON_TEXT: dict[str, ReasonText] = {
    "external_credit": ReasonText(
        code="EXTERNAL_CREDIT",
        unfavourable="Credit scores from external bureaus are low",
        absent="No external credit score on file",
    ),
    "loan_size": ReasonText(
        code="LOAN_SIZE",
        unfavourable="Loan amount is high relative to income",
    ),
    "repayment_burden": ReasonText(
        code="REPAYMENT_BURDEN",
        unfavourable="Scheduled repayments are high relative to income",
        absent="Scheduled repayment amount was not provided",
    ),
    "employment": ReasonText(
        code="EMPLOYMENT_TENURE",
        unfavourable="Length of current employment is short",
        absent="No employment history on record",
    ),
    "employment_profile": ReasonText(
        code="EMPLOYMENT_PROFILE",
        unfavourable="Reported employment category is associated with elevated risk",
        absent="Employment details were not provided",
    ),
    "assets": ReasonText(
        code="ASSETS",
        unfavourable="Limited evidence of asset ownership",
        absent="No asset ownership information was provided",
    ),
    "household": ReasonText(
        code="HOUSEHOLD_SIZE",
        unfavourable="Size of household is high relative to income",
        absent="Household size was not provided",
    ),
    "income": ReasonText(
        code="INCOME",
        unfavourable="Stated income is low relative to the amount requested",
    ),
    "education": ReasonText(
        code="EDUCATION",
        unfavourable="Reported education level is associated with elevated risk",
        absent="Education level was not provided",
    ),
    "product": ReasonText(
        code="PRODUCT_TYPE",
        unfavourable="Requested product type is associated with elevated risk",
        absent="Product type was not provided",
    ),
}
```

Add `NamedTuple` to the existing `typing` import at the top of `config.py`, or add `from typing import NamedTuple` if there is no such import yet.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS. If `test_reason_concepts_partition_feature_order_exactly` fails, the assertion message names exactly which features are unmapped or unknown — fix the map, never the test.

- [ ] **Step 5: Lint and type-check**

Run: `ruff check . && ruff format --check . && mypy src/ && lint-imports`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/creditboost/config.py tests/test_config.py
git commit -m "feat: add the adverse action reason catalog

Ten concepts partitioning the 20 features, each with a code, unfavourable
wording, and -- where reachable -- distinct wording for absent data. Grouping is
what stops one idea filling all four reason slots, and it closes an ECOA trap:
employed_to_age can never surface alone because it is only ever reported as
employment.

A word-boundary test asserts no reason text names age, sex or marital status. A
disclosure that reintroduces a protected basis in prose would undo the feature
work it accompanies."
```

---

### Task 4: `reasons.py`, the pure ranking module

**Files:**
- Create: `src/creditboost/reasons.py`
- Modify: `src/creditboost/schema.py` (add the `Reason` model)
- Modify: `pyproject.toml` (add `creditboost.reasons` to the contract)
- Test: `tests/test_reasons.py`

**Interfaces:**
- Consumes: `config.REASON_CONCEPTS`, `config.REASON_TEXT`, `config.MAX_REASONS` from Task 3.
- Produces: `creditboost.schema.Reason` (pydantic model, fields `code: str` and `description: str`) and `creditboost.reasons.principal_reasons(contributions: Mapping[str, float], missing: Mapping[str, bool]) -> list[Reason]`. Task 5 calls exactly this signature.

- [ ] **Step 1: Add the `Reason` model**

In `src/creditboost/schema.py`, add above `PredictResponse`:

```python
class Reason(BaseModel):
    """One principal factor increasing an applicant's risk.

    `code` is stable and machine-readable; `description` is the plain-language
    text a creditor can put in front of an applicant. Contribution magnitudes are
    deliberately not exposed: they are meaningless to an applicant and would make
    the model straightforward to reverse-engineer.
    """

    code: str
    description: str
```

`Reason` lives here rather than in `reasons.py` because it is part of the response contract, and this module is where the contracts live.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_reasons.py`:

```python
from creditboost import config
from creditboost.reasons import principal_reasons


def no_contributions() -> dict[str, float]:
    """Every feature at zero. Tests raise only the features they care about,
    so each asserts one behaviour rather than inheriting a fixture's noise."""
    return dict.fromkeys(config.FEATURE_ORDER, 0.0)


def nothing_missing() -> dict[str, bool]:
    return dict.fromkeys(config.FEATURE_ORDER, False)


def test_a_positive_concept_becomes_a_reason():
    contributions = no_contributions() | {"EXT_SOURCE_2": 0.5}

    reasons = principal_reasons(contributions, nothing_missing())

    assert [reason.code for reason in reasons] == ["EXTERNAL_CREDIT"]


def test_a_helpful_feature_is_never_a_reason():
    """A feature that pushed the applicant toward approval is not a reason for
    denial. This is the difference between an explanation and a reason code."""
    contributions = no_contributions() | {"EXT_SOURCE_2": -0.9}

    assert principal_reasons(contributions, nothing_missing()) == []


def test_contributions_are_summed_within_a_concept():
    """Three weak bureau contributions must beat one stronger single feature --
    that is the whole point of grouping before ranking."""
    contributions = no_contributions() | {
        "EXT_SOURCE_1": 0.2,
        "EXT_SOURCE_2": 0.2,
        "EXT_SOURCE_3": 0.2,
        "NAME_EDUCATION_TYPE": 0.5,
    }

    reasons = principal_reasons(contributions, nothing_missing())

    assert [reason.code for reason in reasons] == ["EXTERNAL_CREDIT", "EDUCATION"]


def test_reasons_are_ordered_by_concept_total_descending():
    contributions = no_contributions() | {
        "NAME_EDUCATION_TYPE": 0.1,
        "EXT_SOURCE_1": 0.9,
        "CNT_CHILDREN": 0.5,
    }

    reasons = principal_reasons(contributions, nothing_missing())

    assert [reason.code for reason in reasons] == [
        "EXTERNAL_CREDIT",
        "HOUSEHOLD_SIZE",
        "EDUCATION",
    ]


def test_at_most_four_reasons_are_returned():
    contributions = no_contributions() | {
        "EXT_SOURCE_1": 0.9,
        "AMT_CREDIT": 0.8,
        "AMT_ANNUITY": 0.7,
        "DAYS_EMPLOYED": 0.6,
        "NAME_INCOME_TYPE": 0.5,
        "FLAG_OWN_CAR": 0.4,
    }

    reasons = principal_reasons(contributions, nothing_missing())

    assert len(reasons) == config.MAX_REASONS


def test_no_positive_concept_yields_no_reasons():
    assert principal_reasons(no_contributions(), nothing_missing()) == []


def test_a_fully_missing_concept_uses_the_absent_wording():
    """The thin-file case: saying a score is low when none was ever observed is
    simply untrue."""
    contributions = no_contributions() | {"EXT_SOURCE_1": 0.3, "EXT_SOURCE_2": 0.3}
    missing = nothing_missing() | {
        "EXT_SOURCE_1": True,
        "EXT_SOURCE_2": True,
        "EXT_SOURCE_3": True,
    }

    reasons = principal_reasons(contributions, missing)

    assert reasons[0].description == "No external credit score on file"


def test_a_partially_missing_concept_uses_the_unfavourable_wording():
    """Two bureau scores on file and one absent is not 'no score on file'."""
    contributions = no_contributions() | {"EXT_SOURCE_1": 0.3}
    missing = nothing_missing() | {"EXT_SOURCE_1": True}

    reasons = principal_reasons(contributions, missing)

    assert reasons[0].description == "Credit scores from external bureaus are low"


def test_a_concept_with_no_absent_text_keeps_the_unfavourable_wording():
    """loan_size contains AMT_CREDIT, a required field, so it can never be fully
    absent -- but a defensive path must not crash if it is ever called that way."""
    contributions = no_contributions() | {"AMT_CREDIT": 0.4}
    missing = nothing_missing() | {
        "AMT_CREDIT": True,
        "AMT_GOODS_PRICE": True,
        "credit_to_income": True,
    }

    reasons = principal_reasons(contributions, missing)

    assert reasons[0].description == "Loan amount is high relative to income"


def test_equal_totals_break_ties_deterministically():
    """The same request must always produce the same reasons, in the same order."""
    contributions = no_contributions() | {
        "NAME_EDUCATION_TYPE": 0.5,
        "NAME_CONTRACT_TYPE": 0.5,
    }

    first = principal_reasons(contributions, nothing_missing())
    second = principal_reasons(contributions, nothing_missing())

    assert [reason.code for reason in first] == [reason.code for reason in second]


def test_every_returned_code_comes_from_the_catalog():
    contributions = dict.fromkeys(config.FEATURE_ORDER, 0.1)

    reasons = principal_reasons(contributions, nothing_missing())

    catalog = {text.code for text in config.REASON_TEXT.values()}
    assert {reason.code for reason in reasons} <= catalog
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_reasons.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'creditboost.reasons'`

- [ ] **Step 4: Write the module**

Create `src/creditboost/reasons.py`:

```python
"""Adverse action reason codes.

Under ECOA / Regulation B 1002.9 a creditor taking adverse action must disclose
the specific principal reasons for it. This module turns a row of per-feature
model contributions into at most four of them.

Deliberately pure: contributions arrive as a plain mapping rather than a numpy
array, so the grouping, ranking and wording logic is testable with dictionaries
and no model at all. The array handling lives at the edge, in serve/app.py.
"""

from __future__ import annotations

from collections.abc import Mapping

from . import config
from .schema import Reason


def _concept_total(features: tuple[str, ...], contributions: Mapping[str, float]) -> float:
    """Sum a concept's contributions. A feature absent from the mapping counts as
    zero rather than raising: the partition test already guarantees the map and
    FEATURE_ORDER agree, so this cannot silently hide a missing feature."""
    return sum(contributions.get(name, 0.0) for name in features)


def _is_fully_absent(features: tuple[str, ...], missing: Mapping[str, bool]) -> bool:
    """True only when every feature in the concept was missing after the
    transform. Following the single largest contributor instead would be more
    precise and more surprising: an applicant with two of three bureau scores on
    file could be told no score is on file, because the absent one dominated."""
    return all(missing.get(name, False) for name in features)


def principal_reasons(
    contributions: Mapping[str, float],
    missing: Mapping[str, bool],
) -> list[Reason]:
    """The principal factors increasing this applicant's risk, most significant
    first, at most config.MAX_REASONS of them.

    Only positive contributions are eligible: a feature that pushed the applicant
    toward approval is not a reason for denial.
    """
    totals = [
        (concept, _concept_total(features, contributions))
        for concept, features in config.REASON_CONCEPTS.items()
    ]
    adverse = [(concept, total) for concept, total in totals if total > 0.0]

    # Stable sort, so equal totals fall back to REASON_CONCEPTS' declaration
    # order: the same request always yields the same reasons in the same order.
    adverse.sort(key=lambda pair: pair[1], reverse=True)

    reasons: list[Reason] = []
    for concept, _total in adverse[: config.MAX_REASONS]:
        text = config.REASON_TEXT[concept]
        absent = text.absent is not None and _is_fully_absent(
            config.REASON_CONCEPTS[concept], missing
        )
        reasons.append(
            Reason(code=text.code, description=text.absent if absent else text.unfavourable)
        )
    return reasons
```

Note `text.absent if absent else text.unfavourable` is reached only when `text.absent is not None`, which is what the `absent` expression guarantees — mypy needs the `is not None` check on that side of the `and` to narrow the type.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_reasons.py -v`
Expected: PASS, all 11 tests.

- [ ] **Step 6: Register the module in the architecture contract**

In `pyproject.toml`, add `"creditboost.reasons"` to `source_modules` in the `[[tool.importlinter.contracts]]` block, after `"creditboost.lockfile"`.

Run: `lint-imports`
Expected: `Contracts: 1 kept, 0 broken.`

- [ ] **Step 7: Lint and type-check**

Run: `ruff check . && ruff format --check . && mypy src/ && pytest -q`
Expected: all clean, all passing.

- [ ] **Step 8: Commit**

```bash
git add src/creditboost/reasons.py src/creditboost/schema.py tests/test_reasons.py pyproject.toml
git commit -m "feat: add principal_reasons, the reason ranking module

Sums contributions within a concept, discards concepts that helped the
applicant, ranks by total and takes the top four. Absent wording is used only
when every feature in the concept was missing -- following the single largest
contributor instead would let an applicant with two of three bureau scores on
file be told no score is on file.

Pure by construction: contributions arrive as a mapping, not a numpy array, so
the logic is tested with dictionaries and no model."
```

---

### Task 5: Return reasons from `/predict`

**Files:**
- Modify: `src/creditboost/schema.py` (`PredictResponse`)
- Modify: `src/creditboost/serve/app.py:71-95` (the `predict` handler)
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `reasons.principal_reasons` and `schema.Reason` from Task 4.
- Produces: a `/predict` response carrying `reasons`. Nothing depends on this downstream.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api.py`. The module already provides everything needed: a module-scoped `client` fixture at `tests/test_api.py:41-44`, and a `minimal_payload()` helper at `tests/test_api.py:47-48` returning `{"AMT_INCOME_TOTAL": 100_000.0, "AMT_CREDIT": 400_000.0, "DAYS_BIRTH": -12000}` — the three required fields and nothing else, which is already the thin-file case these tests want. Use both. Do **not** add a new payload builder.

```python
def test_predict_returns_at_most_four_reasons(client):
    response = client.post("/predict", json=minimal_payload())

    assert response.status_code == 200
    reasons = response.json()["reasons"]
    assert 0 <= len(reasons) <= 4


def test_every_reason_has_a_code_and_a_description(client):
    reasons = client.post("/predict", json=minimal_payload()).json()["reasons"]

    catalog = {text.code for text in config.REASON_TEXT.values()}
    for reason in reasons:
        assert reason["code"] in catalog
        assert reason["description"]


def test_the_same_request_yields_the_same_reasons(client):
    payload = minimal_payload()

    first = client.post("/predict", json=payload).json()["reasons"]
    second = client.post("/predict", json=payload).json()["reasons"]

    assert first == second


def test_an_applicant_with_no_external_scores_is_told_exactly_that(client):
    """The thin-file case the service exists for. If external_credit ranks at
    all, it must say no score is on file -- never that the score is low."""
    # minimal_payload() carries the three required fields only, so all three
    # external scores are already absent -- which is the thin-file case exactly.
    reasons = client.post("/predict", json=minimal_payload()).json()["reasons"]

    external = [r for r in reasons if r["code"] == "EXTERNAL_CREDIT"]
    for reason in external:
        assert reason["description"] == "No external credit score on file"


def test_the_not_employed_sentinel_reads_as_no_employment_history(client):
    """365243 is scrubbed to NaN by the transform, so an unemployed applicant
    must never be told their employment is merely short."""
    payload = minimal_payload() | {"DAYS_EMPLOYED": config.DAYS_EMPLOYED_SENTINEL}

    reasons = client.post("/predict", json=payload).json()["reasons"]

    employment = [r for r in reasons if r["code"] == "EMPLOYMENT_TENURE"]
    for reason in employment:
        assert reason["description"] == "No employment history on record"


def test_marital_status_is_still_accepted_and_never_reported(client):
    """Quarantine, end to end: the field is accepted without error and cannot
    appear in any disclosure, because it is not a feature at all."""
    payload = minimal_payload() | {"NAME_FAMILY_STATUS": "Widow"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    text = " ".join(r["description"] for r in response.json()["reasons"]).lower()
    for term in ("marital", "widow", "married", "spouse"):
        assert term not in text
```

- [ ] **Step 2: Write the margin-reconstruction test**

Also append to `tests/test_api.py`:

```python
def test_contributions_plus_bias_reconstruct_the_predicted_probability(client):
    """pred_contribs returns n_features + 1 values, bias last. An off-by-one in
    that row produces reasons that are entirely plausible and entirely wrong --
    a failure with no natural symptom, so it gets an explicit test.

    Summing the whole row (contributions plus bias) gives the margin; the
    logistic of that margin must equal the probability the service reports.
    """
    import math

    import xgboost as xgb

    from creditboost.features import transform
    from creditboost.serve import deps

    payload = minimal_payload()
    probability = client.post("/predict", json=payload).json()["probability"]

    frame = transform([payload])
    matrix = xgb.DMatrix(frame, enable_categorical=True)
    row = deps.get_model().booster.predict(matrix, pred_contribs=True)[0]

    assert len(row) == len(config.FEATURE_ORDER) + 1
    reconstructed = 1.0 / (1.0 + math.exp(-float(row.sum())))
    assert reconstructed == pytest.approx(probability, abs=1e-6)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_api.py -k "reason or contributions or employment or marital" -v`
Expected: FAIL with `KeyError: 'reasons'`, because `PredictResponse` has no such field yet.

- [ ] **Step 4: Add the response field**

In `src/creditboost/schema.py`, add to `PredictResponse`:

```python
    reasons: list[Reason] = Field(
        default_factory=list,
        description=(
            "Principal factors increasing this applicant's risk, most significant "
            "first, at most four. Where a caller takes adverse action on this "
            "score, these are the specific principal reasons ECOA / Regulation B "
            "1002.9 requires. Empty when no factor pushed the score upward."
        ),
    )
```

- [ ] **Step 5: Wire the handler**

In `src/creditboost/serve/app.py`, replace the body of `predict` between `model = deps.get_model()` and the `logger.info(...)` call:

```python
        # The frame is kept rather than inlined into DMatrix: its NaN mask is how
        # a reason knows whether to say a value was unfavourable or absent, and
        # post-transform is the right notion -- the DAYS_EMPLOYED sentinel has
        # already been scrubbed to NaN by this point.
        frame = transform([request.model_dump()])
        matrix = xgb.DMatrix(frame, enable_categorical=True)

        probability = float(model.booster.predict(matrix)[0])

        # A second call rather than deriving the probability from the contribution
        # sum. One call would be cheaper, but it makes the service's primary
        # output a byproduct of the explanation path, where a numeric drift would
        # land on the number that matters most.
        contribution_row = model.booster.predict(matrix, pred_contribs=True)[0]
        if len(contribution_row) != len(config.FEATURE_ORDER) + 1:
            raise RuntimeError(
                f"pred_contribs returned {len(contribution_row)} values for "
                f"{len(config.FEATURE_ORDER)} features; expected one per feature "
                "plus a bias term. Refusing to derive reasons from a misread row."
            )

        contributions = {
            name: float(value)
            for name, value in zip(config.FEATURE_ORDER, contribution_row[:-1], strict=True)
        }
        missing = {name: bool(value) for name, value in frame.isna().iloc[0].items()}

        band = risk_band(probability)
        reasons = principal_reasons(contributions, missing)
```

and change the returned response to include them:

```python
        return PredictResponse(
            probability=probability,
            risk_band=band,
            model_version=model.metadata.version,
            reasons=reasons,
        )
```

Add the import at the top of the module, beside the existing ones:

```python
from ..reasons import principal_reasons
```

`config` and `transform` are already imported in this module; confirm rather than duplicating. **Logging is unchanged** — do not add reasons to the log record. They are applicant disclosures, and the existing invariant's spirit is that nothing about the applicant accumulates in log aggregation.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_api.py -v`
Expected: PASS.

If `test_contributions_plus_bias_reconstruct_the_predicted_probability` fails, do **not** loosen the tolerance. A mismatch means the row is being read wrongly — most likely the bias column is being included in `contributions` or a feature is being dropped — which is exactly the defect the test exists to catch.

- [ ] **Step 7: Measure the latency cost**

The second `pred_contribs` call is not free, and the spec commits to measuring rather than assuming.

```bash
python - <<'PY'
import time
import statistics
from fastapi.testclient import TestClient
from creditboost.serve.app import app

payload = {"AMT_INCOME_TOTAL": 100000.0, "AMT_CREDIT": 400000.0, "DAYS_BIRTH": -12000}
with TestClient(app) as client:
    for _ in range(20):
        client.post("/predict", json=payload)
    samples = []
    for _ in range(200):
        started = time.perf_counter()
        client.post("/predict", json=payload)
        samples.append((time.perf_counter() - started) * 1000)
print(f"median {statistics.median(samples):.2f} ms, p95 {sorted(samples)[189]:.2f} ms")
PY
```

Record the figures in the commit message. This needs a loadable artifact in `models/`, so run it after Task 2. There is no pass/fail threshold — the point is a recorded number, not a gate.

- [ ] **Step 8: Lint, type-check, and run everything**

Run: `ruff check . && ruff format --check . && mypy src/ && lint-imports && pytest -v`
Expected: all clean, all passing.

- [ ] **Step 9: Commit**

```bash
git add src/creditboost/schema.py src/creditboost/serve/app.py tests/test_api.py
git commit -m "feat: return adverse action reason codes from /predict

Up to four plain-language reasons on every response, ranked by concept
contribution. Two booster calls rather than one: deriving the probability from
the contribution sum would make the primary output a byproduct of the
explanation path.

Median latency <FIGURE> ms, p95 <FIGURE> ms.

The margin-reconstruction test is the load-bearing one -- an off-by-one in the
pred_contribs row yields reasons that are entirely plausible and entirely wrong,
a failure with no natural symptom."
```

Replace both `<FIGURE>` placeholders with the values measured in Step 7.

---

### Task 6: Documentation and milestone verification

**Files:**
- Modify: `CLAUDE.md` (Repository state, Roadmap, Invariants, Commands)
- Modify: `README.md` (API section, project layout)

**Interfaces:**
- Consumes: everything above.
- Produces: documentation matching the code.

- [ ] **Step 1: Update `CLAUDE.md`'s Invariants**

Replace this bullet:

```markdown
- **`CODE_GENDER` must never appear** in any feature list, request schema, or transform, and
  **raw `DAYS_BIRTH` must never be a model feature.** Under ECOA / Regulation B, sex and age
  are prohibited bases for credit decisions in the US. Age enters only through the
  `employed_to_age` ratio.
```

with:

```markdown
- **No member of `config.PROTECTED_ATTRIBUTES` is ever a model feature.** ECOA,
  15 U.S.C. §1691(a)(1), names the prohibited bases in one sentence: race, color, religion,
  national origin, sex or marital status, or age. `CODE_GENDER` is never accepted at all;
  `DAYS_BIRTH` and `NAME_FAMILY_STATUS` are accepted and never modelled. The rule is about
  features, not reads — the transform still reads `DAYS_BIRTH` to derive `employed_to_age`,
  which Regulation B allows in an empirically derived, statistically sound scoring system.
  Marital status has no equivalent allowance, which is why it is not a feature.
- **`MONITORING_ONLY_FIELDS` are accepted but never transformed.** Reg B §1002.13 requires
  collecting certain protected attributes precisely so fair-lending monitoring is possible.
  An attribute a service refuses to accept cannot be collected retroactively; a feature can
  always be restored by retraining.
- **`REASON_CONCEPTS` partitions `FEATURE_ORDER` exactly**, and **no reason text names or
  implies age, sex, or marital status.** Both have tests. A disclosure that reintroduces a
  protected basis in prose would undo the feature work it accompanies.
- **At most four reasons, and only from positive contributions.** A feature that helped the
  applicant is not a reason for denial.
```

Also amend the feature-count bullet: `FEATURE_ORDER` has exactly **20** entries; `REQUEST_FIELDS` still has 19.

- [ ] **Step 2: Update `CLAUDE.md`'s Repository state and Roadmap**

Change "Milestones 1 and 2 are implemented" to include Milestone 3, and add to the Roadmap:

```markdown
**Milestone 3 — adverse action reason codes.** Done. `/predict` returns up to four
plain-language reasons drawn from a curated catalog in `config.py`, ranked by summed
XGBoost `pred_contribs` contributions grouped into ten concepts. No new runtime dependency.

Writing the spec surfaced a defect in the Milestone 1 feature set: `NAME_FAMILY_STATUS` was
a model feature, and marital status is an enumerated ECOA prohibited basis. It was removed
and the model retrained as `model-v0.2.0`; the field is still accepted, under
`MONITORING_ONLY_FIELDS`, so later disparate-impact work can measure it.

- Design spec: `docs/superpowers/specs/2026-09-02-creditboost-adverse-action-reason-codes-design.md`
- Implementation plan: `docs/superpowers/plans/2026-09-02-creditboost-adverse-action-reason-codes.md`

**Known open question, deliberately deferred:** `NAME_INCOME_TYPE` carries `Maternity leave`
(a sex proxy) and `Pensioner` (an age proxy) as levels, so an `employment_profile` reason can
implicate a protected characteristic indirectly. Unlike marital status these are levels
rather than a whole prohibited-basis feature, and dropping employment type would cost real
signal. It belongs to disparate-impact measurement, which is unspecced.
```

- [ ] **Step 3: Update `README.md`**

In the `POST /predict` section, add `reasons` to the documented response, with a real example:

```json
{
  "probability": 0.34,
  "risk_band": "high",
  "model_version": "0.2.0",
  "reasons": [
    {"code": "EXTERNAL_CREDIT", "description": "No external credit score on file"},
    {"code": "LOAN_SIZE", "description": "Loan amount is high relative to income"}
  ]
}
```

Add a sentence explaining what they are: the principal factors increasing risk, at most four, ordered by contribution, and the specific principal reasons Reg B §1002.9 requires where a caller takes adverse action on the score.

In Project layout, add `reasons.py` to the `src/creditboost/` listing:

```
  reasons.py             # contributions -> at most four adverse action reasons
```

- [ ] **Step 4: Verify every documented command and example still works**

Run the README's `POST /predict` example against a running container and confirm the response shape matches what the README claims, including `reasons`. A documented example that does not run is worse than none.

- [ ] **Step 5: Run the milestone's success criteria**

Work through the spec's ten criteria and confirm each:

```bash
python -c "
from creditboost import config
assert len(config.FEATURE_ORDER) == 20, len(config.FEATURE_ORDER)
assert len(config.REQUEST_FIELDS) == 19, len(config.REQUEST_FIELDS)
assert not set(config.PROTECTED_ATTRIBUTES) & set(config.FEATURE_ORDER)
assert config.MODEL_VERSION == '0.2.0'
print('criteria 1, 3 (config half): ok')
"
pytest -v
ruff check . && ruff format --check . && mypy src/ && lint-imports
docker build -t creditboost:m3 .
```

Criteria 5 through 9 are covered by the tests written in Tasks 3–5; criterion 10 by the lint block and the absence of any new entry in `pyproject.toml`'s runtime `dependencies`.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: record Milestone 3 and the generalised ECOA invariant

The prohibited-basis rule is now stated once, over PROTECTED_ATTRIBUTES, rather
than as prose naming two fields and silently omitting the third.

Records the NAME_INCOME_TYPE proxy-level question as a named open item for
disparate-impact work rather than leaving it undocumented."
```

---

## Success Criteria

Verify each before considering the milestone done. Each maps to the spec's criterion of the same number.

- [ ] 1. `FEATURE_ORDER` has 20 entries, disjoint from `PROTECTED_ATTRIBUTES`; `REQUEST_FIELDS` has 19.
- [ ] 2. `REASON_CONCEPTS` partitions `FEATURE_ORDER` exactly, proven by a test that failed first.
- [ ] 3. `model-v0.2.0` is released and pinned, `MODEL_VERSION` is `0.2.0`, and `creditboost-artifact verify` passes inside `docker build`.
- [ ] 4. The retrain cleared the AUC floor, with the achieved figure recorded against the 0.7533 baseline in Task 2's commit message.
- [ ] 5. `POST /predict` returns at most four reasons, each a catalog code plus plain-language text.
- [ ] 6. A concept whose total is not positive never appears in a response.
- [ ] 7. An applicant with no external score on file is told exactly that, never that their score is low.
- [ ] 8. An applicant with the `DAYS_EMPLOYED` sentinel is told no employment history is on record.
- [ ] 9. No reason text names or implies age, sex, or marital status — enforced by a word-boundary test.
- [ ] 10. No new runtime dependency; `ruff`, `mypy`, `lint-imports` and the full suite are clean.
