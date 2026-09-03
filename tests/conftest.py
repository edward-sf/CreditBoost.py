from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "sample.csv"


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
