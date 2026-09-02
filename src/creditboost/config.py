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
