import io
import json
import logging

from creditboost.serve.logging_config import JsonFormatter, configure_logging


def _reset_creditboost_logger() -> logging.Logger:
    logger = logging.getLogger("creditboost")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    return logger


def test_configure_logging_is_idempotent():
    """create_app() calls configure_logging() on every invocation, including
    across the many times the test suite constructs an app. A second or
    third call must not stack a second or third handler."""
    logger = _reset_creditboost_logger()
    try:
        configure_logging()
        configure_logging()
        configure_logging()
        assert len(logger.handlers) == 1
    finally:
        _reset_creditboost_logger()


def test_configure_logging_does_not_touch_the_root_logger():
    """Only the creditboost logger tree is configured -- basicConfig() or a
    root-level handler would also reformat uvicorn's own log lines."""
    root = logging.getLogger()
    before = list(root.handlers)
    _reset_creditboost_logger()
    try:
        configure_logging()
        assert root.handlers == before
    finally:
        _reset_creditboost_logger()


def test_configure_logging_disables_propagation():
    """propagate=False keeps records from also reaching the root logger's
    handlers in uvicorn's plain-text format (double logging)."""
    logger = _reset_creditboost_logger()
    try:
        configure_logging()
        assert logger.propagate is False
    finally:
        _reset_creditboost_logger()


def test_json_formatter_emits_one_json_object_with_the_standard_fields():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("creditboost.test_logging_config")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        logger.info(
            "prediction served",
            extra={
                "request_id": "abc-123",
                "latency_ms": 4.2,
                "model_version": "0.1.0",
                "risk_band": "low",
                "provenance": "production",
            },
        )
    finally:
        logger.removeHandler(handler)

    line = json.loads(stream.getvalue().strip())
    assert line["message"] == "prediction served"
    assert line["request_id"] == "abc-123"
    assert line["latency_ms"] == 4.2
    assert line["model_version"] == "0.1.0"
    assert line["risk_band"] == "low"
    assert line["provenance"] == "production"
    assert "level" in line
    assert "timestamp" in line


def test_json_formatter_surfaces_arbitrary_extra_fields():
    """Not a fixed whitelist: whatever travels via extra= is what shows up.
    This is what makes test_prediction_logs_carry_no_applicant_financial_data
    able to actually fail if the app ever logs a disallowed field."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("creditboost.test_logging_config_extra")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        logger.info("something happened", extra={"some_new_field": "some_value"})
    finally:
        logger.removeHandler(handler)

    line = json.loads(stream.getvalue().strip())
    assert line["some_new_field"] == "some_value"
