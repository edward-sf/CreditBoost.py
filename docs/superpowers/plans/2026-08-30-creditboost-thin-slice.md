# CreditBoost Thin End-to-End Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete minimal path from Home Credit training data through an XGBoost model to a containerized FastAPI scoring service published to GHCR by CI.

**Architecture:** A single installable `creditboost` package whose `features.transform()` is imported by both the training CLI and the serving app, making train/serve skew structurally impossible. The trained model is committed as a versioned JSON artifact and baked into a multi-stage Docker image. GitHub Actions lints, tests, builds, smoke-tests the running container, and publishes to GHCR on merges to main.

**Tech Stack:** Python 3.11+, XGBoost, scikit-learn, pandas, FastAPI, Pydantic v2, pytest, ruff, mypy, Docker, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-30-creditboost-thin-slice-design.md`

## Global Constraints

- `requires-python = ">=3.11"`. CI matrix tests 3.11 and 3.12; the Docker image pins 3.12.
- `serve/` may import `artifact`, `features`, `schema`, `banding`, `config` — **never** `data` or `train`. Enforced by a test in Task 8.
- Training dependencies (pandas, scikit-learn) belong to the `[train]` extra only. The runtime image installs the base package plus FastAPI/uvicorn, never `[train]`.
- `CODE_GENDER` must never appear in any feature list, request schema, or transform. Raw `DAYS_BIRTH` must never appear in `FEATURE_ORDER`.
- `FEATURE_ORDER` has exactly 21 entries. `REQUEST_FIELDS` has exactly 19.
- `DAYS_EMPLOYED == 365243` maps to NaN **before** derived ratios are computed.
- Missing values are never imputed. NaN reaches XGBoost intact.
- Risk bands: `low` for probability < 0.10, `medium` for 0.10 ≤ p < 0.30, `high` for p ≥ 0.30.
- No applicant financial field may ever be written to a log.
- Run `ruff format .` before each task's commit step. The code blocks in this plan
  are written for readability and are not all ruff-format clean; CI checks
  formatting, so normalise it as you go rather than at the end.
- Model version is the hand-bumped `MODEL_VERSION` constant in `config.py`, initially `"0.1.0"`.

## Deviation From Spec (requires reviewer awareness)

The spec assumes one committed artifact. But Tasks 9 and 10 (Docker, CI) need an artifact to exist, while a *production* artifact needs the manual Kaggle download. Building those tasks against nothing would leave them unverifiable until the very end.

**Resolution:** `model_meta.json` gains a `provenance` field, `"fixture"` or `"production"`. Task 7 commits a fixture-trained artifact (`provenance: "fixture"`, trained with the AUC floor overridden to 0) so the full pipeline is verifiable without credentials. Task 11 replaces it with the production artifact. The `/metadata` endpoint exposes `provenance`, so a fixture model can never be mistaken for a real one in a deployed container.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Packaging, dependencies, extras, console script, tool config |
| `src/creditboost/config.py` | All constants: feature lists, category levels, thresholds, paths |
| `src/creditboost/features.py` | The shared transform and `FEATURE_ORDER` |
| `src/creditboost/schema.py` | Pydantic request, response, and metadata models |
| `src/creditboost/banding.py` | Probability to risk band |
| `src/creditboost/artifact.py` | Artifact save/load and the feature-order gate |
| `src/creditboost/data.py` | CSV load, column validation, SHA-256, stratified split |
| `src/creditboost/train.py` | Training CLI |
| `src/creditboost/serve/deps.py` | Process-wide loaded-model state |
| `src/creditboost/serve/app.py` | FastAPI app factory and routes |
| `tests/fixtures/generate_fixture.py` | Deterministic synthetic fixture generator |
| `tests/fixtures/sample.csv` | Committed synthetic fixture |
| `Dockerfile`, `.dockerignore` | Runtime image |
| `.github/workflows/ci.yml` | Lint, test, build, push |

---

### Task 1: Project scaffolding and configuration

**Files:**
- Create: `pyproject.toml`, `src/creditboost/__init__.py`, `src/creditboost/config.py`, `src/creditboost/serve/__init__.py`, `tests/__init__.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `config.MODEL_VERSION: str`, `config.FEATURE_ORDER: tuple[str, ...]`, `config.REQUEST_FIELDS: tuple[str, ...]`, `config.NUMERIC_FEATURES`, `config.BINARY_FEATURES`, `config.CATEGORICAL_LEVELS: dict[str, tuple[str, ...]]`, `config.CATEGORICAL_FEATURES`, `config.DERIVED_FEATURES`, `config.DAYS_EMPLOYED_SENTINEL: int`, `config.RISK_BAND_LOW_MAX: float`, `config.RISK_BAND_MEDIUM_MAX: float`, `config.MIN_VALIDATION_AUC: float`, `config.RANDOM_SEED: int`, `config.VALIDATION_SIZE: float`, `config.TARGET_COLUMN: str`, `config.MODEL_PATH`, `config.METADATA_PATH`, `config.DEFAULT_DATA_PATH`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path

from creditboost import config


def test_feature_order_has_exactly_21_entries():
    assert len(config.FEATURE_ORDER) == 21


def test_request_fields_has_exactly_19_entries():
    assert len(config.REQUEST_FIELDS) == 19


def test_feature_order_has_no_duplicates():
    assert len(set(config.FEATURE_ORDER)) == len(config.FEATURE_ORDER)


def test_feature_order_is_the_concatenation_of_its_parts():
    expected = (
        config.NUMERIC_FEATURES
        + config.BINARY_FEATURES
        + config.CATEGORICAL_FEATURES
        + config.DERIVED_FEATURES
    )
    assert config.FEATURE_ORDER == expected


def test_gender_is_excluded_everywhere():
    """ECOA / Regulation B: sex is a prohibited basis for credit decisions."""
    assert "CODE_GENDER" not in config.FEATURE_ORDER
    assert "CODE_GENDER" not in config.REQUEST_FIELDS


def test_raw_age_is_not_a_model_feature_but_is_a_request_field():
    """DAYS_BIRTH is consumed only to derive employed_to_age."""
    assert "DAYS_BIRTH" not in config.FEATURE_ORDER
    assert "DAYS_BIRTH" in config.REQUEST_FIELDS


def test_every_categorical_feature_has_declared_levels():
    for name in config.CATEGORICAL_FEATURES:
        assert name in config.CATEGORICAL_LEVELS
        assert len(config.CATEGORICAL_LEVELS[name]) > 0


def test_risk_band_thresholds_are_ordered_within_zero_to_one():
    assert 0 < config.RISK_BAND_LOW_MAX < config.RISK_BAND_MEDIUM_MAX < 1


def test_model_dir_honours_the_environment_override(monkeypatch):
    """The container installs the package into /opt/venv, where the repo-relative
    default would resolve into site-packages instead of /app/models."""
    import importlib

    monkeypatch.setenv("CREDITBOOST_MODEL_DIR", "/app/models")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.MODEL_PATH == Path("/app/models/model.json")
        assert reloaded.METADATA_PATH == Path("/app/models/model_meta.json")
    finally:
        monkeypatch.delenv("CREDITBOOST_MODEL_DIR")
        importlib.reload(config)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'creditboost'`

- [ ] **Step 3: Create the packaging file**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "creditboost"
version = "0.1.0"
description = "Loan default risk scoring for thin-file borrowers"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic>=2.9",
    "xgboost>=2.1",
    "pandas>=2.2",
    "numpy>=1.26",
]

[project.optional-dependencies]
train = ["scikit-learn>=1.5"]
dev = [
    "pytest>=8.3",
    "pytest-cov>=5.0",
    "httpx>=0.27",
    "ruff>=0.7",
    "mypy>=1.13",
    "pandas-stubs",
]

[project.scripts]
creditboost-train = "creditboost.train:main"

[tool.hatch.build.targets.wheel]
packages = ["src/creditboost"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
# No `packages` key: CI invokes `mypy src/`, and mypy rejects a path argument
# when packages is also configured.
```

Note: `pandas` is a base dependency, not `[train]`-only, because `features.transform()` uses it and serving imports it. Only scikit-learn is training-exclusive.

- [ ] **Step 4: Create the package skeleton**

```bash
mkdir -p src/creditboost/serve tests/fixtures
touch src/creditboost/__init__.py src/creditboost/serve/__init__.py tests/__init__.py
```

- [ ] **Step 5: Write the config module**

```python
# src/creditboost/config.py
"""Project-wide constants. No logic lives here."""

from __future__ import annotations

import os
from pathlib import Path

MODEL_VERSION = "0.1.0"

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent

# The repo-relative default works for an editable install. Inside the container
# the package lives in /opt/venv/lib/..., where that default would resolve into
# site-packages, so the image sets CREDITBOOST_MODEL_DIR=/app/models instead.
MODEL_DIR = Path(os.environ.get("CREDITBOOST_MODEL_DIR", REPO_ROOT / "models"))
MODEL_PATH = MODEL_DIR / "model.json"
METADATA_PATH = MODEL_DIR / "model_meta.json"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "application_train.csv"

TARGET_COLUMN = "TARGET"

NUMERIC_FEATURES: tuple[str, ...] = (
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
)

BINARY_FEATURES: tuple[str, ...] = ("FLAG_OWN_CAR", "FLAG_OWN_REALTY")

CATEGORICAL_LEVELS: dict[str, tuple[str, ...]] = {
    "NAME_CONTRACT_TYPE": ("Cash loans", "Revolving loans"),
    "NAME_INCOME_TYPE": (
        "Working",
        "State servant",
        "Commercial associate",
        "Pensioner",
        "Unemployed",
        "Student",
        "Businessman",
        "Maternity leave",
    ),
    "NAME_EDUCATION_TYPE": (
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ),
    "NAME_FAMILY_STATUS": (
        "Single / not married",
        "Married",
        "Civil marriage",
        "Widow",
        "Separated",
        "Unknown",
    ),
    "NAME_HOUSING_TYPE": (
        "House / apartment",
        "Rented apartment",
        "With parents",
        "Municipal apartment",
        "Office apartment",
        "Co-op apartment",
    ),
    "OCCUPATION_TYPE": (
        "Laborers",
        "Sales staff",
        "Core staff",
        "Managers",
        "Drivers",
        "High skill tech staff",
        "Accountants",
        "Medicine staff",
        "Security staff",
        "Cooking staff",
        "Cleaning staff",
        "Private service staff",
        "Low-skill Laborers",
        "Waiters/barmen staff",
        "Secretaries",
        "Realty agents",
        "HR staff",
        "IT staff",
    ),
}

CATEGORICAL_FEATURES: tuple[str, ...] = tuple(CATEGORICAL_LEVELS)

DERIVED_FEATURES: tuple[str, ...] = (
    "credit_to_income",
    "annuity_to_income",
    "employed_to_age",
)

# The exact column order the model is trained on and scored with.
FEATURE_ORDER: tuple[str, ...] = (
    NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + DERIVED_FEATURES
)

# DAYS_BIRTH is accepted from callers to derive employed_to_age, but raw age is
# deliberately not a model feature: age is a protected basis under ECOA.
REQUEST_FIELDS: tuple[str, ...] = (
    NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + ("DAYS_BIRTH",)
)

# Home Credit encodes "not employed" as this positive sentinel in DAYS_EMPLOYED.
DAYS_EMPLOYED_SENTINEL = 365243

# Business policy, not a model property: changes without retraining.
RISK_BAND_LOW_MAX = 0.10
RISK_BAND_MEDIUM_MAX = 0.30

MIN_VALIDATION_AUC = 0.70
RANDOM_SEED = 42
VALIDATION_SIZE = 0.2
```

- [ ] **Step 6: Install and run the tests**

Run: `pip install -e ".[train,dev]" && pytest tests/test_config.py -v`
Expected: PASS — 9 tests

- [ ] **Step 7: Add data/ to gitignore and commit**

```bash
printf '\n# Local dataset (obtained manually from Kaggle)\ndata/\n' >> .gitignore
git add pyproject.toml .gitignore src/creditboost tests
git commit -m "feat: add package scaffolding and configuration constants"
```

---

### Task 2: Synthetic test fixture

**Files:**
- Create: `tests/fixtures/generate_fixture.py`, `tests/fixtures/sample.csv`
- Test: `tests/test_fixture.py`

**Interfaces:**
- Consumes: `config.NUMERIC_FEATURES`, `config.BINARY_FEATURES`, `config.CATEGORICAL_LEVELS`, `config.TARGET_COLUMN`, `config.DAYS_EMPLOYED_SENTINEL`, `config.RANDOM_SEED`
- Produces: `tests/fixtures/sample.csv` — 200 rows with all `REQUEST_FIELDS` plus `TARGET`. Later tasks read it via the `fixture_path` pytest fixture defined in Task 6.

The fixture is synthetic, not sampled: Kaggle's terms restrict redistributing the real data. Generating it also lets us guarantee the edge cases exist.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fixture.py
import pandas as pd
import pytest

from creditboost import config

FIXTURE = "tests/fixtures/sample.csv"


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


def test_fixture_has_every_request_field_and_the_target(frame):
    for column in config.REQUEST_FIELDS:
        assert column in frame.columns
    assert config.TARGET_COLUMN in frame.columns


def test_fixture_has_200_rows(frame):
    assert len(frame) == 200


def test_fixture_contains_both_target_classes(frame):
    assert set(frame[config.TARGET_COLUMN].unique()) == {0, 1}


def test_fixture_contains_the_not_employed_sentinel(frame):
    """The 365243 sentinel must be present so the scrub is exercised."""
    assert (frame["DAYS_EMPLOYED"] == config.DAYS_EMPLOYED_SENTINEL).sum() >= 5


def test_fixture_contains_rows_with_all_external_scores_missing(frame):
    """The thin-file borrower: no external credit score at all."""
    all_missing = frame[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isna().all(axis=1)
    assert all_missing.sum() >= 5


def test_fixture_categorical_values_are_declared_levels(frame):
    for column, levels in config.CATEGORICAL_LEVELS.items():
        present = set(frame[column].dropna().unique())
        assert present <= set(levels), f"{column} has undeclared levels: {present - set(levels)}"


def test_generator_is_deterministic():
    """Regenerating must not produce a spurious diff."""
    from tests.fixtures.generate_fixture import build_fixture

    first = build_fixture()
    second = build_fixture()
    pd.testing.assert_frame_equal(first, second)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fixture.py -v`
Expected: FAIL — `FileNotFoundError: tests/fixtures/sample.csv`

- [ ] **Step 3: Write the generator**

```python
# tests/fixtures/generate_fixture.py
"""Generates a synthetic Home Credit-shaped fixture.

Synthetic by design: Kaggle's competition terms restrict redistributing the real
dataset, and generating it lets us guarantee every edge case is represented.

Regenerate with:  python tests/fixtures/generate_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from creditboost import config

N_ROWS = 200
OUTPUT = Path(__file__).parent / "sample.csv"


def build_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)

    income = rng.lognormal(mean=11.8, sigma=0.5, size=N_ROWS).round(2)
    credit = (income * rng.uniform(1.5, 6.0, size=N_ROWS)).round(2)

    frame = pd.DataFrame(
        {
            "EXT_SOURCE_1": rng.uniform(0, 1, size=N_ROWS).round(4),
            "EXT_SOURCE_2": rng.uniform(0, 1, size=N_ROWS).round(4),
            "EXT_SOURCE_3": rng.uniform(0, 1, size=N_ROWS).round(4),
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": (credit / rng.uniform(10, 30, size=N_ROWS)).round(2),
            "AMT_GOODS_PRICE": (credit * rng.uniform(0.8, 1.0, size=N_ROWS)).round(2),
            "DAYS_EMPLOYED": -rng.integers(30, 12000, size=N_ROWS),
            "DAYS_BIRTH": -rng.integers(7700, 25000, size=N_ROWS),
            "CNT_CHILDREN": rng.integers(0, 4, size=N_ROWS),
            "CNT_FAM_MEMBERS": rng.integers(1, 6, size=N_ROWS).astype(float),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], size=N_ROWS),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], size=N_ROWS),
        }
    )

    for column, levels in config.CATEGORICAL_LEVELS.items():
        frame[column] = rng.choice(list(levels), size=N_ROWS)

    # Rows 0-9: pensioners, flagged with the not-employed sentinel.
    frame.loc[0:9, "DAYS_EMPLOYED"] = config.DAYS_EMPLOYED_SENTINEL
    frame.loc[0:9, "NAME_INCOME_TYPE"] = "Pensioner"

    # Rows 10-19: thin-file borrowers with no external score of any kind.
    frame.loc[10:19, ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]] = np.nan

    # EXT_SOURCE_1 is missing in most real rows; mirror that sparsity.
    sparse = rng.choice(N_ROWS, size=int(N_ROWS * 0.55), replace=False)
    frame.loc[sparse, "EXT_SOURCE_1"] = np.nan

    # A weak but genuine signal, so a model trained on this is not pure noise.
    risk = (
        frame["EXT_SOURCE_2"].fillna(0.5) * -1.5
        + (frame["AMT_CREDIT"] / frame["AMT_INCOME_TOTAL"]) * 0.25
        + rng.normal(0, 0.35, size=N_ROWS)
    )
    frame[config.TARGET_COLUMN] = (risk > np.quantile(risk, 0.92)).astype(int)

    return frame[list(config.REQUEST_FIELDS) + [config.TARGET_COLUMN]]


if __name__ == "__main__":
    build_fixture().to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT}")
```

- [ ] **Step 4: Generate the fixture and run the tests**

Run: `python tests/fixtures/generate_fixture.py && pytest tests/test_fixture.py -v`
Expected: PASS — 7 tests

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/generate_fixture.py tests/fixtures/sample.csv tests/test_fixture.py
git commit -m "test: add deterministic synthetic Home Credit fixture"
```

---

### Task 3: The shared feature transform

This is the linchpin of the whole design. Every other component depends on it being right, and the parity test at the end is what makes train/serve skew impossible rather than merely unlikely.

**Files:**
- Create: `src/creditboost/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: everything `config` produces in Task 1
- Produces: `features.transform(records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame` — returns a frame whose columns are exactly `config.FEATURE_ORDER` in that order, numeric columns as `float64`, categorical columns as pandas `Categorical` with declared levels. Also re-exports `features.FEATURE_ORDER`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
import numpy as np
import pandas as pd
import pytest

from creditboost import config
from creditboost.features import FEATURE_ORDER, transform


def base_record() -> dict:
    return {
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.6,
        "EXT_SOURCE_3": 0.7,
        "AMT_INCOME_TOTAL": 100_000.0,
        "AMT_CREDIT": 400_000.0,
        "AMT_ANNUITY": 20_000.0,
        "AMT_GOODS_PRICE": 380_000.0,
        "DAYS_EMPLOYED": -2000,
        "DAYS_BIRTH": -12000,
        "CNT_CHILDREN": 1,
        "CNT_FAM_MEMBERS": 3.0,
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "N",
        "NAME_CONTRACT_TYPE": "Cash loans",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "OCCUPATION_TYPE": "Managers",
    }


def test_output_columns_are_exactly_feature_order():
    result = transform([base_record()])
    assert list(result.columns) == list(FEATURE_ORDER)


def test_raw_age_is_not_in_the_output():
    assert "DAYS_BIRTH" not in transform([base_record()]).columns


def test_not_employed_sentinel_becomes_nan():
    record = base_record() | {"DAYS_EMPLOYED": config.DAYS_EMPLOYED_SENTINEL}
    assert np.isnan(transform([record])["DAYS_EMPLOYED"].iloc[0])


def test_sentinel_is_scrubbed_before_the_ratio_is_derived():
    """A 365243 tenure must not leak into employed_to_age as a plausible number."""
    record = base_record() | {"DAYS_EMPLOYED": config.DAYS_EMPLOYED_SENTINEL}
    assert np.isnan(transform([record])["employed_to_age"].iloc[0])


def test_missing_external_scores_stay_nan_and_are_not_imputed():
    record = base_record() | {"EXT_SOURCE_1": None, "EXT_SOURCE_2": None, "EXT_SOURCE_3": None}
    row = transform([record]).iloc[0]
    for column in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        assert np.isnan(row[column]), f"{column} was imputed"


def test_derived_ratios_are_computed_correctly():
    row = transform([base_record()]).iloc[0]
    assert row["credit_to_income"] == pytest.approx(4.0)
    assert row["annuity_to_income"] == pytest.approx(0.2)
    assert row["employed_to_age"] == pytest.approx(2000 / 12000)


def test_zero_denominator_yields_nan_rather_than_raising():
    record = base_record() | {"AMT_INCOME_TOTAL": 0.0}
    row = transform([record]).iloc[0]
    assert np.isnan(row["credit_to_income"])
    assert np.isnan(row["annuity_to_income"])


def test_binary_flags_map_to_one_and_zero():
    row = transform([base_record()]).iloc[0]
    assert row["FLAG_OWN_CAR"] == 1.0
    assert row["FLAG_OWN_REALTY"] == 0.0


def test_unrecognised_binary_flag_becomes_nan():
    record = base_record() | {"FLAG_OWN_CAR": "maybe"}
    assert np.isnan(transform([record])["FLAG_OWN_CAR"].iloc[0])


def test_unknown_categorical_level_becomes_nan_without_raising():
    record = base_record() | {"OCCUPATION_TYPE": "Astronaut"}
    result = transform([record])
    assert pd.isna(result["OCCUPATION_TYPE"].iloc[0])


def test_categorical_columns_carry_the_declared_levels():
    result = transform([base_record()])
    for column, levels in config.CATEGORICAL_LEVELS.items():
        assert list(result[column].cat.categories) == list(levels)


def test_missing_input_column_becomes_nan_rather_than_raising():
    record = base_record()
    del record["AMT_GOODS_PRICE"]
    assert np.isnan(transform([record])["AMT_GOODS_PRICE"].iloc[0])


def test_transform_is_deterministic():
    first = transform([base_record()])
    second = transform([base_record()])
    pd.testing.assert_frame_equal(first, second)


def test_empty_input_returns_empty_frame_with_correct_columns():
    result = transform([])
    assert len(result) == 0
    assert list(result.columns) == list(FEATURE_ORDER)


def test_dict_and_dataframe_paths_produce_identical_matrices():
    """The parity guarantee: serving and training cannot diverge."""
    record = base_record()
    from_api = transform([record])
    from_training = transform(pd.DataFrame([record]))
    pd.testing.assert_frame_equal(from_api, from_training)


def test_parity_holds_across_the_whole_fixture():
    frame = pd.read_csv("tests/fixtures/sample.csv")
    records = frame.to_dict(orient="records")
    pd.testing.assert_frame_equal(
        transform(frame).reset_index(drop=True),
        transform(records).reset_index(drop=True),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'creditboost.features'`

- [ ] **Step 3: Write the transform**

```python
# src/creditboost/features.py
"""The single feature transform, imported by both training and serving.

There is exactly one implementation of this logic on purpose. Training records
the resulting FEATURE_ORDER into the artifact metadata, and serving refuses to
start if the two disagree, which is what makes train/serve skew impossible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from . import config

FEATURE_ORDER = config.FEATURE_ORDER

Records = pd.DataFrame | Sequence[Mapping[str, Any]]


def _as_frame(records: Records) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records
    return pd.DataFrame(list(records))


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Always return a Series, so a missing input column degrades to NaN."""
    if name in frame.columns:
        return frame[name]
    return pd.Series([None] * len(frame), index=frame.index, dtype="object")


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(_column(frame, name), errors="coerce").astype("float64")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, yielding NaN where the denominator is zero or missing."""
    return numerator / denominator.mask(denominator == 0)


def transform(records: Records) -> pd.DataFrame:
    """Turn raw applicant records into the model's feature matrix.

    Accepts either a DataFrame (training) or a sequence of mappings (serving);
    both paths run identical logic. Missing values are never imputed — XGBoost
    routes NaN natively, and for a thin-file borrower a missing external score
    is itself signal.
    """
    frame = _as_frame(records)
    if frame.empty:
        frame = pd.DataFrame(index=pd.RangeIndex(0))

    out = pd.DataFrame(index=frame.index)

    for name in config.NUMERIC_FEATURES:
        out[name] = _numeric(frame, name)

    # Scrub the not-employed sentinel BEFORE deriving ratios from it. Left raw,
    # 365243 becomes a ~1000-year tenure that reads as a plausible value.
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].mask(
        out["DAYS_EMPLOYED"] == config.DAYS_EMPLOYED_SENTINEL
    )

    for name in config.BINARY_FEATURES:
        mapped = _column(frame, name).map({"Y": 1.0, "N": 0.0})
        out[name] = pd.to_numeric(mapped, errors="coerce").astype("float64")

    for name, levels in config.CATEGORICAL_LEVELS.items():
        out[name] = pd.Categorical(_column(frame, name), categories=list(levels))

    days_birth = _numeric(frame, "DAYS_BIRTH")
    income = out["AMT_INCOME_TOTAL"]
    out["credit_to_income"] = _safe_divide(out["AMT_CREDIT"], income)
    out["annuity_to_income"] = _safe_divide(out["AMT_ANNUITY"], income)
    out["employed_to_age"] = _safe_divide(out["DAYS_EMPLOYED"], days_birth)

    return out[list(FEATURE_ORDER)]
```

Note on `employed_to_age`: both inputs are negative day counts, so the ratio is positive and is unitless employment tenure relative to age. Raw age never reaches the model.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features.py -v`
Expected: PASS — 16 tests. If `test_empty_input_returns_empty_frame_with_correct_columns` fails on dtype, confirm `_as_frame([])` produced a zero-row frame before column assignment.

- [ ] **Step 5: Commit**

```bash
git add src/creditboost/features.py tests/test_features.py
git commit -m "feat: add shared feature transform with train/serve parity tests"
```

---

### Task 4: Request/response schemas and risk banding

**Files:**
- Create: `src/creditboost/schema.py`, `src/creditboost/banding.py`
- Test: `tests/test_schema.py`, `tests/test_banding.py`

**Interfaces:**
- Consumes: `config.RISK_BAND_LOW_MAX`, `config.RISK_BAND_MEDIUM_MAX`, `config.FEATURE_ORDER`
- Produces:
  - `schema.PredictRequest` — Pydantic model over the 19 request fields; only `AMT_INCOME_TOTAL`, `AMT_CREDIT`, and `DAYS_BIRTH` are required
  - `schema.PredictResponse(probability: float, risk_band: str, model_version: str)`
  - `schema.ModelMetadata(version, trained_at, dataset_sha256, n_train_rows, feature_order, metrics, xgboost_version, provenance)`
  - `banding.risk_band(probability: float) -> Literal["low", "medium", "high"]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_banding.py
import pytest

from creditboost.banding import risk_band


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "low"),
        (0.05, "low"),
        (0.0999, "low"),
        (0.10, "medium"),
        (0.20, "medium"),
        (0.2999, "medium"),
        (0.30, "high"),
        (0.95, "high"),
        (1.0, "high"),
    ],
)
def test_bands_are_assigned_at_the_configured_boundaries(probability, expected):
    assert risk_band(probability) == expected


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_probability_outside_zero_to_one_raises(probability):
    with pytest.raises(ValueError):
        risk_band(probability)
```

```python
# tests/test_schema.py
import pytest
from pydantic import ValidationError

from creditboost import config
from creditboost.schema import ModelMetadata, PredictRequest, PredictResponse


def minimal_payload() -> dict:
    return {"AMT_INCOME_TOTAL": 100_000.0, "AMT_CREDIT": 400_000.0, "DAYS_BIRTH": -12000}


def test_minimal_payload_is_accepted():
    """A thin-file borrower supplies very little; only three fields are required."""
    request = PredictRequest(**minimal_payload())
    assert request.AMT_INCOME_TOTAL == 100_000.0
    assert request.EXT_SOURCE_1 is None


def test_request_exposes_exactly_the_configured_fields():
    assert set(PredictRequest.model_fields) == set(config.REQUEST_FIELDS)


def test_gender_is_not_an_accepted_field():
    assert "CODE_GENDER" not in PredictRequest.model_fields


@pytest.mark.parametrize("field", ["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH"])
def test_omitting_a_required_field_is_rejected(field):
    payload = minimal_payload()
    del payload[field]
    with pytest.raises(ValidationError):
        PredictRequest(**payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("AMT_INCOME_TOTAL", -1.0),
        ("AMT_INCOME_TOTAL", 0.0),
        ("AMT_CREDIT", -5.0),
        ("AMT_ANNUITY", 0.0),
        ("EXT_SOURCE_1", 1.5),
        ("EXT_SOURCE_2", -0.1),
        ("CNT_CHILDREN", -1),
        ("DAYS_BIRTH", 100),
    ],
)
def test_out_of_range_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        PredictRequest(**(minimal_payload() | {field: value}))


def test_days_employed_accepts_the_positive_sentinel():
    """365243 is a valid input meaning 'not employed'; the transform scrubs it."""
    request = PredictRequest(**(minimal_payload() | {"DAYS_EMPLOYED": 365243}))
    assert request.DAYS_EMPLOYED == 365243


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(**(minimal_payload() | {"FAVOURITE_COLOUR": "blue"}))


def test_response_carries_the_model_version():
    response = PredictResponse(probability=0.2, risk_band="medium", model_version="0.1.0")
    assert response.model_dump()["model_version"] == "0.1.0"


def test_metadata_round_trips_through_json():
    metadata = ModelMetadata(
        version="0.1.0",
        trained_at="2026-08-30T12:00:00Z",
        dataset_sha256="abc123",
        n_train_rows=160,
        feature_order=list(config.FEATURE_ORDER),
        metrics={"roc_auc": 0.75, "pr_auc": 0.24, "brier": 0.068},
        xgboost_version="2.1.0",
        provenance="fixture",
    )
    assert ModelMetadata.model_validate_json(metadata.model_dump_json()) == metadata


def test_metadata_rejects_an_unknown_provenance():
    with pytest.raises(ValidationError):
        ModelMetadata(
            version="0.1.0",
            trained_at="2026-08-30T12:00:00Z",
            dataset_sha256="abc123",
            n_train_rows=160,
            feature_order=list(config.FEATURE_ORDER),
            metrics={},
            xgboost_version="2.1.0",
            provenance="guesswork",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schema.py tests/test_banding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'creditboost.schema'`

- [ ] **Step 3: Write the banding module**

```python
# src/creditboost/banding.py
"""Probability to risk band.

Thresholds are business policy, not model properties: they live in config and
change without retraining, which is why they are not stored in the artifact.
"""

from __future__ import annotations

from typing import Literal

from . import config

RiskBand = Literal["low", "medium", "high"]


def risk_band(probability: float) -> RiskBand:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if probability < config.RISK_BAND_LOW_MAX:
        return "low"
    if probability < config.RISK_BAND_MEDIUM_MAX:
        return "medium"
    return "high"
```

- [ ] **Step 4: Write the schema module**

```python
# src/creditboost/schema.py
"""API and artifact contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .banding import RiskBand


class PredictRequest(BaseModel):
    """One loan applicant.

    Only income, credit amount, and date of birth are required. Everything else
    is optional because thin-file borrowers, by definition, have sparse records —
    and a missing external score is signal the model uses rather than an error.

    CODE_GENDER is deliberately absent: sex is a prohibited basis for credit
    decisions under ECOA / Regulation B.
    """

    model_config = ConfigDict(extra="forbid")

    AMT_INCOME_TOTAL: float = Field(gt=0)
    AMT_CREDIT: float = Field(gt=0)
    DAYS_BIRTH: float = Field(lt=0, description="Days before application; negative")

    EXT_SOURCE_1: float | None = Field(default=None, ge=0, le=1)
    EXT_SOURCE_2: float | None = Field(default=None, ge=0, le=1)
    EXT_SOURCE_3: float | None = Field(default=None, ge=0, le=1)
    AMT_ANNUITY: float | None = Field(default=None, gt=0)
    AMT_GOODS_PRICE: float | None = Field(default=None, gt=0)
    DAYS_EMPLOYED: float | None = Field(
        default=None, description="Negative day count, or 365243 meaning not employed"
    )
    CNT_CHILDREN: int = Field(default=0, ge=0)
    CNT_FAM_MEMBERS: float | None = Field(default=None, ge=1)

    FLAG_OWN_CAR: str | None = None
    FLAG_OWN_REALTY: str | None = None
    NAME_CONTRACT_TYPE: str | None = None
    NAME_INCOME_TYPE: str | None = None
    NAME_EDUCATION_TYPE: str | None = None
    NAME_FAMILY_STATUS: str | None = None
    NAME_HOUSING_TYPE: str | None = None
    OCCUPATION_TYPE: str | None = None


class PredictResponse(BaseModel):
    # 'model_' is a protected Pydantic namespace; disabling it lets us name the
    # field model_version, which is what the field actually is.
    model_config = ConfigDict(protected_namespaces=())

    probability: float = Field(ge=0, le=1)
    risk_band: RiskBand
    model_version: str


class ModelMetadata(BaseModel):
    """Sidecar written next to model.json. The feature_order field is the gate
    that prevents a model being served against a mismatched transform."""

    model_config = ConfigDict(protected_namespaces=())

    version: str
    trained_at: str
    dataset_sha256: str
    n_train_rows: int
    feature_order: list[str]
    metrics: dict[str, float]
    xgboost_version: str
    provenance: Literal["fixture", "production"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_schema.py tests/test_banding.py -v`
Expected: PASS — 11 banding cases and 19 schema cases (several are parameterised)

- [ ] **Step 6: Commit**

```bash
git add src/creditboost/schema.py src/creditboost/banding.py tests/test_schema.py tests/test_banding.py
git commit -m "feat: add request/response schemas and risk banding"
```

---

### Task 5: Artifact persistence and the feature-order gate

**Files:**
- Create: `src/creditboost/artifact.py`
- Test: `tests/test_artifact.py`

**Interfaces:**
- Consumes: `schema.ModelMetadata`, `config.FEATURE_ORDER`, `config.MODEL_PATH`, `config.METADATA_PATH`
- Produces:
  - `artifact.LoadedModel` — frozen dataclass with `.booster: xgboost.Booster` and `.metadata: ModelMetadata`
  - `artifact.save(booster, metadata, model_path=..., metadata_path=...) -> None`
  - `artifact.load(model_path=..., metadata_path=...) -> LoadedModel`
  - `artifact.FeatureOrderMismatchError`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_artifact.py
import numpy as np
import pytest
import xgboost as xgb

from creditboost import config
from creditboost.artifact import FeatureOrderMismatchError, load, save
from creditboost.schema import ModelMetadata


@pytest.fixture
def booster() -> xgb.Booster:
    rng = np.random.default_rng(0)
    matrix = xgb.DMatrix(rng.normal(size=(40, len(config.FEATURE_ORDER))), label=rng.integers(0, 2, 40))
    return xgb.train({"objective": "binary:logistic"}, matrix, num_boost_round=2)


def metadata(feature_order: list[str] | None = None) -> ModelMetadata:
    return ModelMetadata(
        version=config.MODEL_VERSION,
        trained_at="2026-08-30T12:00:00Z",
        dataset_sha256="abc123",
        n_train_rows=40,
        feature_order=feature_order if feature_order is not None else list(config.FEATURE_ORDER),
        metrics={"roc_auc": 0.75},
        xgboost_version=xgb.__version__,
        provenance="fixture",
    )


def test_save_then_load_round_trips(tmp_path, booster):
    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    save(booster, metadata(), model_path, meta_path)
    loaded = load(model_path, meta_path)
    assert loaded.metadata == metadata()
    assert loaded.booster.num_boosted_rounds() == booster.num_boosted_rounds()


def test_load_rejects_a_mismatched_feature_order(tmp_path, booster):
    """The gate: a model trained on different features must never serve."""
    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    stale = [*config.FEATURE_ORDER[:-1], "some_removed_feature"]
    save(booster, metadata(stale), model_path, meta_path)
    with pytest.raises(FeatureOrderMismatchError):
        load(model_path, meta_path)


def test_load_rejects_a_merely_reordered_feature_list(tmp_path, booster):
    """Order matters, not just membership: XGBoost scores by position."""
    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    swapped = list(config.FEATURE_ORDER)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    save(booster, metadata(swapped), model_path, meta_path)
    with pytest.raises(FeatureOrderMismatchError):
        load(model_path, meta_path)


def test_load_raises_when_the_model_file_is_absent(tmp_path, booster):
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(metadata().model_dump_json())
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing.json", meta_path)


def test_save_creates_the_parent_directory(tmp_path, booster):
    nested = tmp_path / "models" / "v1"
    save(booster, metadata(), nested / "model.json", nested / "meta.json")
    assert (nested / "model.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_artifact.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'creditboost.artifact'`

- [ ] **Step 3: Write the artifact module**

```python
# src/creditboost/artifact.py
"""Artifact persistence and the startup gate against train/serve skew."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import xgboost as xgb

from . import config
from .schema import ModelMetadata


class FeatureOrderMismatchError(RuntimeError):
    """The artifact was trained on a different feature layout than this code emits."""


@dataclass(frozen=True)
class LoadedModel:
    booster: xgb.Booster
    metadata: ModelMetadata


def save(
    booster: xgb.Booster,
    metadata: ModelMetadata,
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.METADATA_PATH,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path))
    metadata_path.write_text(metadata.model_dump_json(indent=2) + "\n")


def load(
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.METADATA_PATH,
) -> LoadedModel:
    """Load the artifact, refusing any model whose features disagree with ours."""
    if not model_path.exists():
        raise FileNotFoundError(f"model artifact not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"model metadata not found: {metadata_path}")

    metadata = ModelMetadata.model_validate_json(metadata_path.read_text())

    expected = list(config.FEATURE_ORDER)
    if metadata.feature_order != expected:
        raise FeatureOrderMismatchError(
            "artifact feature order does not match this build's FEATURE_ORDER; "
            f"artifact has {len(metadata.feature_order)} features, code expects "
            f"{len(expected)}. Retrain the model against this code."
        )

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return LoadedModel(booster=booster, metadata=metadata)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_artifact.py -v`
Expected: PASS — 5 tests

- [ ] **Step 5: Commit**

```bash
git add src/creditboost/artifact.py tests/test_artifact.py
git commit -m "feat: add artifact persistence with feature-order gate"
```

---

### Task 6: Dataset loading and splitting

**Files:**
- Create: `src/creditboost/data.py`, `tests/conftest.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `config.REQUEST_FIELDS`, `config.TARGET_COLUMN`, `config.RANDOM_SEED`, `config.VALIDATION_SIZE`
- Produces:
  - `data.MissingColumnsError`
  - `data.file_sha256(path: Path) -> str`
  - `data.load_training_frame(path: Path) -> pd.DataFrame`
  - `data.split(frame: pd.DataFrame, seed: int = ..., validation_size: float = ...) -> tuple[pd.DataFrame, pd.DataFrame]`
  - `tests/conftest.py` defines the `fixture_path` pytest fixture used by Tasks 6, 7, and 8

- [ ] **Step 1: Write the failing test**

```python
# tests/conftest.py
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "sample.csv"
```

```python
# tests/test_data.py
import pytest

from creditboost import config
from creditboost.data import MissingColumnsError, file_sha256, load_training_frame, split


def test_loads_the_fixture(fixture_path):
    frame = load_training_frame(fixture_path)
    assert len(frame) == 200
    assert config.TARGET_COLUMN in frame.columns


def test_missing_required_column_raises_and_names_it(fixture_path, tmp_path):
    frame = load_training_frame(fixture_path).drop(columns=["AMT_CREDIT"])
    truncated = tmp_path / "truncated.csv"
    frame.to_csv(truncated, index=False)
    with pytest.raises(MissingColumnsError, match="AMT_CREDIT"):
        load_training_frame(truncated)


def test_absent_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_training_frame(tmp_path / "nope.csv")


def test_sha256_is_stable_and_content_sensitive(tmp_path):
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text("hello")
    second.write_text("hello")
    assert file_sha256(first) == file_sha256(second)
    second.write_text("goodbye")
    assert file_sha256(first) != file_sha256(second)


def test_split_partitions_every_row_exactly_once(fixture_path):
    frame = load_training_frame(fixture_path)
    train, valid = split(frame)
    assert len(train) + len(valid) == len(frame)
    assert set(train.index).isdisjoint(valid.index)


def test_split_is_stratified_on_the_target(fixture_path):
    """At an 8% base rate an unstratified split can miss positives entirely."""
    frame = load_training_frame(fixture_path)
    train, valid = split(frame)
    overall = frame[config.TARGET_COLUMN].mean()
    assert train[config.TARGET_COLUMN].mean() == pytest.approx(overall, abs=0.03)
    assert valid[config.TARGET_COLUMN].mean() == pytest.approx(overall, abs=0.03)


def test_split_is_reproducible_for_a_fixed_seed(fixture_path):
    frame = load_training_frame(fixture_path)
    first, _ = split(frame)
    second, _ = split(frame)
    assert list(first.index) == list(second.index)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'creditboost.data'`

- [ ] **Step 3: Write the data module**

```python
# src/creditboost/data.py
"""Training-time dataset access. Never imported by the serving package."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


class MissingColumnsError(ValueError):
    """The CSV lacks columns the transform requires."""


def file_sha256(path: Path) -> str:
    """Content hash, recorded in metadata so a model traces to its training data."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_frame(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"training data not found: {path}")

    frame = pd.read_csv(path)

    required = {*config.REQUEST_FIELDS, config.TARGET_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise MissingColumnsError(f"training data is missing columns: {', '.join(missing)}")

    return frame


def split(
    frame: pd.DataFrame,
    seed: int = config.RANDOM_SEED,
    validation_size: float = config.VALIDATION_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split. Stratification is required, not optional: positives are
    roughly 8% of rows, and an unstratified split can skew the validation base
    rate enough to make AUC unstable."""
    train, valid = train_test_split(
        frame,
        test_size=validation_size,
        random_state=seed,
        stratify=frame[config.TARGET_COLUMN],
    )
    return train, valid
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data.py -v`
Expected: PASS — 7 tests

- [ ] **Step 5: Commit**

```bash
git add src/creditboost/data.py tests/conftest.py tests/test_data.py
git commit -m "feat: add dataset loading, validation, and stratified split"
```

---

### Task 7: Training CLI and the fixture artifact

**Files:**
- Create: `src/creditboost/train.py`, `models/model.json`, `models/model_meta.json`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `data.load_training_frame`, `data.split`, `data.file_sha256`, `features.transform`, `artifact.save`, `schema.ModelMetadata`
- Produces:
  - `train.fit(train_frame, valid_frame) -> tuple[xgb.Booster, dict[str, float]]`
  - `train.main(argv: list[str] | None = None) -> int` — exit code 0 on success, 1 when the AUC gate rejects the model
  - `models/model.json` and `models/model_meta.json` with `provenance="fixture"`, consumed by Tasks 9 and 10

**Second deviation from spec — no `scale_pos_weight`.** The spec calls for `scale_pos_weight` to counter the class imbalance *and* for a Brier score. These are in tension: `scale_pos_weight` reweights the positive class, which inflates predicted probabilities away from the true base rate. That destroys calibration, makes Brier meaningless, and invalidates risk-band thresholds chosen for an 8% prevalence. Since we serve a probability and band it, calibration is a correctness requirement and ranking alone is not enough. AUC is largely insensitive to the reweighting anyway. **Resolution:** train without `scale_pos_weight`, keep Brier honest, and keep the configured bands meaningful. Handling imbalance through calibrated reweighting is a reasonable cycle-2 topic.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
import json

import pytest

from creditboost import config
from creditboost.artifact import load
from creditboost.data import load_training_frame, split
from creditboost.train import fit, main


@pytest.fixture(scope="module")
def trained(fixture_path):
    frame = load_training_frame(fixture_path)
    train_frame, valid_frame = split(frame)
    return fit(train_frame, valid_frame)


def test_fit_returns_a_booster_and_the_three_metrics(trained):
    _, metrics = trained
    assert set(metrics) == {"roc_auc", "pr_auc", "brier"}


def test_accuracy_is_not_reported(trained):
    """At an 8% base rate accuracy carries no information; reporting it misleads."""
    _, metrics = trained
    assert "accuracy" not in metrics


def test_metrics_are_in_valid_ranges(trained):
    _, metrics = trained
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["brier"] <= 1.0


@pytest.mark.slow
def test_cli_writes_a_loadable_artifact(fixture_path, tmp_path):
    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    code = main(
        [
            "--data", str(fixture_path),
            "--model-out", str(model_path),
            "--metadata-out", str(meta_path),
            "--min-auc", "0.0",
            "--provenance", "fixture",
        ]
    )
    assert code == 0

    loaded = load(model_path, meta_path)
    assert loaded.metadata.feature_order == list(config.FEATURE_ORDER)
    assert loaded.metadata.provenance == "fixture"
    assert loaded.metadata.version == config.MODEL_VERSION
    assert loaded.metadata.n_train_rows == 160
    assert loaded.metadata.dataset_sha256


@pytest.mark.slow
def test_cli_refuses_to_write_a_model_below_the_auc_floor(fixture_path, tmp_path):
    """A bad retrain must produce no artifact at all, so nothing downstream
    needs to detect one."""
    model_path, meta_path = tmp_path / "model.json", tmp_path / "meta.json"
    code = main(
        [
            "--data", str(fixture_path),
            "--model-out", str(model_path),
            "--metadata-out", str(meta_path),
            "--min-auc", "1.01",
        ]
    )
    assert code == 1
    assert not model_path.exists()
    assert not meta_path.exists()


def test_committed_fixture_artifact_is_present_and_declares_its_provenance():
    """Tasks 9 and 10 depend on this artifact existing in the repo."""
    metadata = json.loads(config.METADATA_PATH.read_text())
    assert metadata["provenance"] in {"fixture", "production"}
    assert metadata["feature_order"] == list(config.FEATURE_ORDER)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'creditboost.train'`

- [ ] **Step 3: Write the training module**

```python
# src/creditboost/train.py
"""Training CLI.

Run against the real dataset:
    creditboost-train --data data/application_train.csv

The dataset is obtained manually from Kaggle and is gitignored; this command is
never run in CI.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config
from .artifact import save
from .data import file_sha256, load_training_frame, split
from .features import transform
from .schema import ModelMetadata

logger = logging.getLogger(__name__)

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": config.RANDOM_SEED,
}
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30


def _matrix(frame: pd.DataFrame) -> xgb.DMatrix:
    return xgb.DMatrix(
        transform(frame),
        label=frame[config.TARGET_COLUMN],
        enable_categorical=True,
    )


def fit(
    train_frame: pd.DataFrame, valid_frame: pd.DataFrame
) -> tuple[xgb.Booster, dict[str, float]]:
    """Train and evaluate. No scale_pos_weight: it inflates probabilities away
    from the true base rate, which would break calibration, make the Brier score
    meaningless, and invalidate the configured risk-band thresholds."""
    dtrain, dvalid = _matrix(train_frame), _matrix(valid_frame)

    booster = xgb.train(
        PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    # Trim to the best iteration so serving needs no iteration_range and cannot
    # accidentally score with the overfit tail.
    booster = booster[: booster.best_iteration + 1]

    y_valid = valid_frame[config.TARGET_COLUMN]
    probabilities = booster.predict(dvalid)

    metrics = {
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "pr_auc": float(average_precision_score(y_valid, probabilities)),
        "brier": float(brier_score_loss(y_valid, probabilities)),
    }
    return booster, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="creditboost-train")
    parser.add_argument("--data", type=Path, default=config.DEFAULT_DATA_PATH)
    parser.add_argument("--model-out", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--metadata-out", type=Path, default=config.METADATA_PATH)
    parser.add_argument(
        "--min-auc",
        type=float,
        default=config.MIN_VALIDATION_AUC,
        help="Refuse to write a model below this validation ROC-AUC.",
    )
    parser.add_argument(
        "--provenance",
        choices=("fixture", "production"),
        default="production",
        help="Records whether this model was trained on real data or the test fixture.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    frame = load_training_frame(args.data)
    train_frame, valid_frame = split(frame)
    logger.info("training on %d rows, validating on %d", len(train_frame), len(valid_frame))

    booster, metrics = fit(train_frame, valid_frame)
    logger.info("metrics: %s", metrics)

    if metrics["roc_auc"] < args.min_auc:
        logger.error(
            "validation ROC-AUC %.4f is below the floor %.4f; no artifact written",
            metrics["roc_auc"],
            args.min_auc,
        )
        return 1

    metadata = ModelMetadata(
        version=config.MODEL_VERSION,
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        dataset_sha256=file_sha256(args.data),
        n_train_rows=len(train_frame),
        feature_order=list(config.FEATURE_ORDER),
        metrics=metrics,
        xgboost_version=xgb.__version__,
        provenance=args.provenance,
    )
    save(booster, metadata, args.model_out, args.metadata_out)
    logger.info("wrote %s and %s", args.model_out, args.metadata_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Generate the fixture artifact**

The AUC floor is overridden to 0 here: a model trained on 200 synthetic rows will not clear 0.70, and this artifact exists only so Tasks 9 and 10 have something to build and boot against. Task 11 replaces it.

Run:
```bash
creditboost-train \
  --data tests/fixtures/sample.csv \
  --min-auc 0.0 \
  --provenance fixture
```
Expected: writes `models/model.json` and `models/model_meta.json`, exit code 0.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_train.py -v`
Expected: PASS — 6 tests

- [ ] **Step 6: Commit**

```bash
git add src/creditboost/train.py tests/test_train.py models/model.json models/model_meta.json
git commit -m "feat: add training CLI with AUC gate, plus fixture-trained artifact"
```

---

### Task 8: The serving application

**Files:**
- Create: `src/creditboost/serve/deps.py`, `src/creditboost/serve/app.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `artifact.load`, `artifact.LoadedModel`, `features.transform`, `banding.risk_band`, `schema.PredictRequest`, `schema.PredictResponse`
- Produces:
  - `deps.load_model(model_path, metadata_path) -> LoadedModel`, `deps.get_model() -> LoadedModel`, `deps.reset() -> None`
  - `app.create_app(model_path=..., metadata_path=...) -> FastAPI`
  - `app.app` — the module-level instance uvicorn serves

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from creditboost import config
from creditboost.artifact import FeatureOrderMismatchError
from creditboost.data import load_training_frame, split
from creditboost.schema import ModelMetadata
from creditboost.serve.app import create_app
from creditboost.train import fit


@pytest.fixture(scope="module")
def artifact_paths(fixture_path, tmp_path_factory):
    from creditboost.artifact import save

    directory = tmp_path_factory.mktemp("artifact")
    model_path, meta_path = directory / "model.json", directory / "meta.json"

    frame = load_training_frame(fixture_path)
    train_frame, valid_frame = split(frame)
    booster, metrics = fit(train_frame, valid_frame)

    save(
        booster,
        ModelMetadata(
            version=config.MODEL_VERSION,
            trained_at="2026-08-30T12:00:00Z",
            dataset_sha256="fixture",
            n_train_rows=len(train_frame),
            feature_order=list(config.FEATURE_ORDER),
            metrics=metrics,
            xgboost_version="test",
            provenance="fixture",
        ),
        model_path,
        meta_path,
    )
    return model_path, meta_path


@pytest.fixture(scope="module")
def client(artifact_paths):
    with TestClient(create_app(*artifact_paths)) as test_client:
        yield test_client


def minimal_payload() -> dict:
    return {"AMT_INCOME_TOTAL": 100_000.0, "AMT_CREDIT": 400_000.0, "DAYS_BIRTH": -12000}


def test_health_reports_the_model_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_version"] == config.MODEL_VERSION


def test_metadata_exposes_provenance_and_feature_order(client):
    body = client.get("/metadata").json()
    assert body["provenance"] == "fixture"
    assert body["feature_order"] == list(config.FEATURE_ORDER)


def test_predict_returns_a_probability_band_and_version(client):
    body = client.post("/predict", json=minimal_payload()).json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}
    assert body["model_version"] == config.MODEL_VERSION


def test_predict_band_agrees_with_the_configured_thresholds(client):
    from creditboost.banding import risk_band

    body = client.post("/predict", json=minimal_payload()).json()
    assert body["risk_band"] == risk_band(body["probability"])


def test_thin_file_borrower_with_no_external_scores_is_scored(client):
    """The product's whole purpose: score someone with no credit score."""
    payload = minimal_payload() | {
        "EXT_SOURCE_1": None,
        "EXT_SOURCE_2": None,
        "EXT_SOURCE_3": None,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_unemployed_applicant_sentinel_is_accepted(client):
    payload = minimal_payload() | {"DAYS_EMPLOYED": config.DAYS_EMPLOYED_SENTINEL}
    assert client.post("/predict", json=payload).status_code == 200


def test_unknown_occupation_degrades_instead_of_failing(client):
    payload = minimal_payload() | {"OCCUPATION_TYPE": "Astronaut"}
    assert client.post("/predict", json=payload).status_code == 200


def test_missing_required_field_is_rejected(client):
    payload = minimal_payload()
    del payload["AMT_CREDIT"]
    assert client.post("/predict", json=payload).status_code == 422


def test_negative_income_is_rejected(client):
    payload = minimal_payload() | {"AMT_INCOME_TOTAL": -1.0}
    assert client.post("/predict", json=payload).status_code == 422


def test_startup_fails_when_the_artifact_is_missing(tmp_path):
    """A container that cannot score correctly must never accept traffic."""
    application = create_app(tmp_path / "absent.json", tmp_path / "absent_meta.json")
    with pytest.raises(FileNotFoundError), TestClient(application):
        pass


def test_startup_fails_on_a_feature_order_mismatch(artifact_paths, tmp_path):
    import json
    import shutil

    model_path, meta_path = artifact_paths
    stale_model, stale_meta = tmp_path / "model.json", tmp_path / "meta.json"
    shutil.copy(model_path, stale_model)

    metadata = json.loads(meta_path.read_text())
    metadata["feature_order"] = metadata["feature_order"][:-1]
    stale_meta.write_text(json.dumps(metadata))

    with pytest.raises(FeatureOrderMismatchError), TestClient(create_app(stale_model, stale_meta)):
        pass


def test_prediction_logs_carry_no_applicant_financial_data(client, caplog):
    """Financial fields are exactly the PII that must not accumulate in log
    aggregation. Only band, version, latency, and request id may be logged."""
    import logging

    with caplog.at_level(logging.INFO, logger="creditboost.serve"):
        client.post("/predict", json=minimal_payload() | {"AMT_INCOME_TOTAL": 123456.0})

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "123456" not in logged
    for field in ("AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "EXT_SOURCE_1"):
        assert field not in logged


def test_serving_does_not_import_the_training_stack():
    """Enforces the one-way dependency rule: serve/ never reaches into training,
    which is what keeps scikit-learn out of the runtime image."""
    code = "import creditboost.serve.app, sys; assert 'sklearn' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'creditboost.serve.app'`

- [ ] **Step 3: Write the model-state module**

```python
# src/creditboost/serve/deps.py
"""Process-wide loaded-model state.

The artifact is read once at startup rather than per request: loading is slow
and the booster is immutable once loaded.
"""

from __future__ import annotations

from pathlib import Path

from .. import config
from ..artifact import LoadedModel, load

_model: LoadedModel | None = None


def load_model(
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.METADATA_PATH,
) -> LoadedModel:
    global _model
    _model = load(model_path, metadata_path)
    return _model


def get_model() -> LoadedModel:
    if _model is None:
        raise RuntimeError("model is not loaded; startup did not complete")
    return _model


def reset() -> None:
    global _model
    _model = None
```

- [ ] **Step 4: Write the application**

```python
# src/creditboost/serve/app.py
"""FastAPI scoring service.

Imports only artifact, features, banding, schema, and config — never data or
train. That keeps the training stack out of the runtime image, and the rule is
enforced by a test.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import xgboost as xgb
from fastapi import FastAPI

from .. import config
from ..banding import risk_band
from ..features import transform
from ..schema import PredictRequest, PredictResponse
from . import deps

logger = logging.getLogger("creditboost.serve")


def create_app(
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.METADATA_PATH,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Any failure here propagates and the process exits non-zero. That is
        # deliberate: a container that cannot score correctly must not serve.
        model = deps.load_model(model_path, metadata_path)
        logger.info(
            "model loaded",
            extra={"model_version": model.metadata.version, "provenance": model.metadata.provenance},
        )
        yield
        deps.reset()

    application = FastAPI(
        title="CreditBoost",
        description="Default risk scoring for thin-file borrowers",
        version=config.MODEL_VERSION,
        lifespan=lifespan,
    )

    @application.get("/health")
    def health() -> dict[str, str]:
        model = deps.get_model()
        return {
            "status": "ok",
            "model_version": model.metadata.version,
            "provenance": model.metadata.provenance,
        }

    @application.get("/metadata")
    def metadata() -> dict:
        return deps.get_model().metadata.model_dump()

    @application.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        started = time.perf_counter()
        model = deps.get_model()

        matrix = xgb.DMatrix(transform([request.model_dump()]), enable_categorical=True)
        probability = float(model.booster.predict(matrix)[0])
        band = risk_band(probability)

        # Deliberately logs no applicant financial fields: that is exactly the
        # PII that should not accumulate in log aggregation.
        logger.info(
            "prediction served",
            extra={
                "request_id": str(uuid.uuid4()),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "model_version": model.metadata.version,
                "risk_band": band,
            },
        )
        return PredictResponse(
            probability=probability,
            risk_band=band,
            model_version=model.metadata.version,
        )

    return application


app = create_app()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_api.py -v`
Expected: PASS — 13 tests

- [ ] **Step 6: Run the whole suite**

Run: `pytest -v`
Expected: PASS — all tests across every module

- [ ] **Step 7: Commit**

```bash
git add src/creditboost/serve tests/test_api.py
git commit -m "feat: add FastAPI scoring service with fail-fast startup checks"
```

---

### Task 9: Container image

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `scripts/smoke.sh`

**Interfaces:**
- Consumes: the installed package, `models/model.json`, `models/model_meta.json`
- Produces: an image serving on port 8000; `scripts/smoke.sh <base-url>` exercising `/health` and `/predict`, reused verbatim by CI in Task 10

- [ ] **Step 1: Write the smoke test script**

```bash
#!/usr/bin/env bash
# scripts/smoke.sh — verify a running container actually serves predictions.
# A plain `docker build` succeeds even if the model never made it into the
# image; this is what catches that.
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

echo "==> waiting for ${BASE_URL}/health"
for _ in $(seq 1 30); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then break; fi
  sleep 1
done

echo "==> GET /health"
curl -fsS "${BASE_URL}/health" | tee /tmp/health.json
grep -q '"status":"ok"' /tmp/health.json

echo "==> POST /predict (thin-file borrower, no external scores)"
curl -fsS -X POST "${BASE_URL}/predict" \
  -H 'Content-Type: application/json' \
  -d '{"AMT_INCOME_TOTAL": 100000, "AMT_CREDIT": 400000, "DAYS_BIRTH": -12000}' \
  | tee /tmp/predict.json
grep -q '"risk_band"' /tmp/predict.json
grep -q '"model_version"' /tmp/predict.json

echo "==> smoke test passed"
```

- [ ] **Step 2: Write the .dockerignore**

```
.git
.github
data
tests
docs
scripts
.venv
__pycache__
*.pyc
.pytest_cache
.mypy_cache
.ruff_cache
```

- [ ] **Step 3: Write the Dockerfile**

```dockerfile
# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src ./src
# Base package only — never [train]. scikit-learn stays out of the runtime.
RUN pip install --no-cache-dir .

FROM python:3.12-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    CREDITBOOST_MODEL_DIR=/app/models
RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=appuser:appuser models /app/models
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"
CMD ["uvicorn", "creditboost.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

`CREDITBOOST_MODEL_DIR` is required because the package is installed into `/opt/venv`, so `config.py`'s repo-relative default would resolve inside site-packages rather than to `/app/models`.

- [ ] **Step 4: Build the image**

Run: `chmod +x scripts/smoke.sh && docker build -t creditboost:dev .`
Expected: build succeeds

- [ ] **Step 5: Verify the container serves predictions**

Run:
```bash
docker run -d --rm --name creditboost-smoke -p 8000:8000 creditboost:dev
./scripts/smoke.sh http://localhost:8000
docker stop creditboost-smoke
```
Expected: `smoke test passed`, with `/health` reporting `"provenance":"fixture"`

- [ ] **Step 6: Verify the training stack is absent from the image**

Run: `docker run --rm creditboost:dev python -c "import sklearn" 2>&1 | grep -q "No module named 'sklearn'" && echo "confirmed: training stack excluded"`
Expected: `confirmed: training stack excluded`

- [ ] **Step 7: Commit**

```bash
git add Dockerfile .dockerignore scripts/smoke.sh
git commit -m "feat: add multi-stage container image and smoke test"
```

---

### Task 10: CI/CD pipeline

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `scripts/smoke.sh`, `pyproject.toml` extras, the committed artifact
- Produces: images at `ghcr.io/<owner>/creditboost` tagged `latest`, `sha-<short>`, and the model version

CI never downloads from Kaggle. It touches only the synthetic fixture and the committed artifact, which is what keeps it fast and credential-free.

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
  push:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

env:
  IMAGE_NAME: ghcr.io/${{ github.repository }}

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -e ".[train,dev]"
      - run: ruff check .
      - run: ruff format --check .
      - run: mypy src/

  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: pip install -e ".[train,dev]"
      - run: pytest -v --cov=creditboost --cov-report=term-missing

  build:
    needs: [lint, test]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - name: Build image
        uses: docker/build-push-action@v6
        with:
          context: .
          load: true
          tags: creditboost:ci
          cache-from: type=gha
          cache-to: type=gha,mode=max
      - name: Run the container and smoke-test it
        run: |
          docker run -d --rm --name creditboost-ci -p 8000:8000 creditboost:ci
          ./scripts/smoke.sh http://localhost:8000
          docker stop creditboost-ci

  push:
    needs: build
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    # Scoped to this job alone, so pull requests from forks never receive
    # registry credentials.
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - name: Read the model version
        id: version
        run: echo "value=$(python -c "import json;print(json.load(open('models/model_meta.json'))['version'])")" >> "$GITHUB_OUTPUT"
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: |
            ${{ env.IMAGE_NAME }}:latest
            ${{ env.IMAGE_NAME }}:sha-${{ github.sha }}
            ${{ env.IMAGE_NAME }}:${{ steps.version.outputs.value }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

- [ ] **Step 2: Verify the workflow parses**

Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('workflow YAML is valid')"`
Expected: `workflow YAML is valid`

- [ ] **Step 3: Reproduce the CI gates locally before pushing**

Run: `ruff check . && ruff format --check . && mypy src/ && pytest -v`
Expected: all four pass. Fix any formatting or typing failures now rather than discovering them in CI.

- [ ] **Step 4: Commit and push the branch**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add lint, test, build, and GHCR publish pipeline"
git push -u origin planning
```

- [ ] **Step 5: Confirm CI is green**

Run: `gh run watch`
Expected: `lint`, `test` (both Python versions), and `build` all pass. `push` is correctly skipped — this is not a push to main.

---

### Task 11: Production training run

The one manual, credentialed step. Deliberately last, so everything before it was verified without Kaggle access.

**Files:**
- Modify: `models/model.json`, `models/model_meta.json`

**Interfaces:**
- Consumes: `train.main`, `data/application_train.csv` (obtained manually)
- Produces: the production artifact, `provenance="production"`, replacing the fixture artifact from Task 7

- [ ] **Step 1: Obtain the dataset**

Download `application_train.csv` from the Home Credit Default Risk competition on Kaggle and place it at `data/application_train.csv`. The `data/` directory is gitignored; the file is never committed and never enters CI.

Run: `ls -lh data/application_train.csv`
Expected: roughly 158 MB present

- [ ] **Step 2: Train the production model**

Run:
```bash
creditboost-train --data data/application_train.csv --provenance production
```
Expected: logs roughly 246,008 training rows, then the three metrics. ROC-AUC should land near 0.74–0.76 on this feature subset. Exit code 0.

If it exits 1 with an AUC below the 0.70 floor, that is the gate doing its job — do not lower the floor to force it through. Investigate first: the most likely causes are a corrupt or truncated download, or a mismatch between the declared category levels in `config.py` and the values actually present in the CSV.

- [ ] **Step 3: Verify the artifact loads and reports production provenance**

Run:
```bash
python -c "
from creditboost.artifact import load
m = load()
print('version   ', m.metadata.version)
print('provenance', m.metadata.provenance)
print('rows      ', m.metadata.n_train_rows)
print('metrics   ', m.metadata.metrics)
assert m.metadata.provenance == 'production'
assert m.metadata.metrics['roc_auc'] >= 0.70
print('OK')
"
```
Expected: `OK`, with provenance `production`

- [ ] **Step 4: Verify the container serves the production model**

Run:
```bash
docker build -t creditboost:prod .
docker run -d --rm --name creditboost-prod -p 8000:8000 creditboost:prod
./scripts/smoke.sh http://localhost:8000
curl -fsS http://localhost:8000/health | grep -q '"provenance":"production"'
docker stop creditboost-prod
```
Expected: smoke test passes and `/health` reports `"provenance":"production"`

- [ ] **Step 5: Run the full suite once more**

Run: `pytest -v`
Expected: PASS. `test_committed_fixture_artifact_is_present_and_declares_its_provenance` still passes — it accepts either provenance and checks the feature order, which is what actually matters.

- [ ] **Step 6: Commit**

```bash
git add models/model.json models/model_meta.json
git commit -m "feat: replace fixture artifact with production-trained model"
```

- [ ] **Step 7: Merge to main and confirm publication**

```bash
git push
gh pr create --fill --base main
```

After merge, run: `gh run watch`
Expected: all four jobs run; `push` publishes to GHCR. Verify with
`docker pull ghcr.io/<owner>/creditboost:latest`.

---

## Verification Checklist

Against the spec's success criteria:

- [ ] `pip install -e ".[train,dev]" && pytest` passes on a fresh clone with no Kaggle credentials (Tasks 1–8)
- [ ] `creditboost-train` produces both artifact files with validation ROC-AUC ≥ 0.70 (Task 11)
- [ ] `docker build` then `docker run` yields a container whose `/health` reports the model version and whose `/predict` scores a thin-file record with all external scores null (Tasks 9, 11)
- [ ] A pull request runs lint, test, and build without touching registry credentials (Task 10)
- [ ] A merge to main publishes a tagged image to GHCR (Task 11)
- [ ] The startup feature-order check demonstrably fails the container when artifact and code disagree (Task 8, `test_startup_fails_on_a_feature_order_mismatch`)
