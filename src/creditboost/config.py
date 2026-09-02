"""Project-wide constants. No logic lives here."""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

MODEL_VERSION = "0.2.0"

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent

# The repo-relative default works for an editable install. Inside the container
# the package lives in /opt/venv/lib/..., where that default would resolve into
# site-packages, so the image sets CREDITBOOST_MODEL_DIR=/app/models instead.
MODEL_DIR = Path(os.environ.get("CREDITBOOST_MODEL_DIR", REPO_ROOT / "models"))
MODEL_PATH = MODEL_DIR / "model.json"
METADATA_PATH = MODEL_DIR / "model_meta.json"
LOCKFILE_PATH = MODEL_DIR / "model.lock.json"
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

# Home Credit encodes "not employed" as this positive sentinel in DAYS_EMPLOYED.
DAYS_EMPLOYED_SENTINEL = 365243

# Business policy, not a model property: changes without retraining.
RISK_BAND_LOW_MAX = 0.10
RISK_BAND_MEDIUM_MAX = 0.30

MIN_VALIDATION_AUC = 0.70
RANDOM_SEED = 42
VALIDATION_SIZE = 0.2

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
