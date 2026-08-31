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
