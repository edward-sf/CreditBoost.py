"""Structured JSON logging for the serving app.

Configures only the `creditboost` logger tree -- never the root logger and
never uvicorn's own loggers -- so a running container emits one JSON object
per log record to stdout, carrying the audit trail the spec requires: request
id, latency, model version, and predicted band. Without this, records emitted
through `creditboost.serve`'s logger have no handler under uvicorn and are
silently dropped; a credit decision would leave no trace at all.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# The attribute names a bare LogRecord carries with no `extra=` supplied.
# Anything beyond this set on a given record came in via `extra=` and is
# exactly what gets surfaced below. This is deliberately not a fixed
# whitelist of "safe" field names: a field that should never be logged (an
# applicant's income, say) has to actually show up in this output for
# test_prediction_logs_carry_no_applicant_financial_data to have anything to
# catch. Filtering it out here would just hide the leak from that guard.
_DEFAULT_RECORD_ATTRS = frozenset(vars(logging.makeLogRecord({})))

_LOGGER_NAME = "creditboost"


class JsonFormatter(logging.Formatter):
    """Renders one LogRecord as one JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _DEFAULT_RECORD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the `creditboost` logger tree to emit structured JSON to stdout.

    Idempotent: `create_app()` calls this on every invocation, including
    across the many times the test suite constructs an app, so a second or
    third call must not stack a second or third handler onto the logger.
    Deliberately does not call `logging.basicConfig()` or touch the root
    logger -- doing so would also reformat uvicorn's own log lines and cause
    creditboost's records to be emitted twice (once here, once via root).
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
