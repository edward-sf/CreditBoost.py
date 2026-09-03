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
