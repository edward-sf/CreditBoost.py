# tests/test_config.py
import re
from pathlib import Path

from creditboost import config


def test_feature_order_has_exactly_20_entries():
    assert len(config.FEATURE_ORDER) == 20


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
