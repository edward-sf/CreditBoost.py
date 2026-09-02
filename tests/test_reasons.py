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
