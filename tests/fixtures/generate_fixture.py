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

# Fields accepted from callers but never modelled -- config.MONITORING_ONLY_FIELDS.
# Their levels live here rather than in config.CATEGORICAL_LEVELS, because that dict
# drives the transform: anything in it becomes a model feature. A real application
# row carries marital status, so the fixture carries it too, and later fair-lending
# analysis has an attribute to slice by.
MONITORING_ONLY_LEVELS: dict[str, tuple[str, ...]] = {
    "NAME_FAMILY_STATUS": (
        "Single / not married",
        "Married",
        "Civil marriage",
        "Widow",
        "Separated",
        "Unknown",
    ),
}


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

    # Generated separately from CATEGORICAL_LEVELS because that dict drives the
    # transform. Note the fixture's values shifted when NAME_FAMILY_STATUS left
    # that dict: one fewer rng.choice runs before the draws below it, moving the
    # whole stream. Only the values moved -- the column set is unchanged, and
    # every property the fixture is asserted on still holds.
    for column, levels in MONITORING_ONLY_LEVELS.items():
        frame[column] = rng.choice(list(levels), size=N_ROWS)

    return frame[list(config.REQUEST_FIELDS) + [config.TARGET_COLUMN]]


if __name__ == "__main__":
    build_fixture().to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT}")
