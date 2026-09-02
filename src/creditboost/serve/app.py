"""FastAPI scoring service.

Imports only artifact, features, banding, schema, and config — never data or
train. That keeps the training stack out of the runtime image, and the rule is
enforced by a test.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import xgboost as xgb
from fastapi import FastAPI

from .. import config
from ..banding import risk_band
from ..features import transform
from ..reasons import principal_reasons
from ..schema import PredictRequest, PredictResponse
from . import deps
from .logging_config import configure_logging

logger = logging.getLogger("creditboost.serve")


def create_app(
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.METADATA_PATH,
) -> FastAPI:
    configure_logging()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Any failure here propagates and the process exits non-zero. That is
        # deliberate: a container that cannot score correctly must not serve.
        model = deps.load_model(model_path, metadata_path)
        logger.info(
            "model loaded",
            extra={
                "model_version": model.metadata.version,
                "provenance": model.metadata.provenance,
            },
        )
        yield
        deps.reset()

    application = FastAPI(
        title="CreditBoost",
        description="Default risk scoring for thin-file borrowers",
        version=config.MODEL_VERSION,
        lifespan=lifespan,
    )

    @application.get("/health")
    def health() -> dict[str, str]:
        model = deps.get_model()
        return {
            "status": "ok",
            "model_version": model.metadata.version,
            "provenance": model.metadata.provenance,
        }

    @application.get("/metadata")
    def metadata() -> dict:
        return deps.get_model().metadata.model_dump()

    @application.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        started = time.perf_counter()
        model = deps.get_model()

        # The frame is kept rather than inlined into DMatrix: its NaN mask is how
        # a reason knows whether to say a value was unfavourable or absent, and
        # post-transform is the right notion -- the DAYS_EMPLOYED sentinel has
        # already been scrubbed to NaN by this point.
        frame = transform([request.model_dump()])
        matrix = xgb.DMatrix(frame, enable_categorical=True)

        probability = float(model.booster.predict(matrix)[0])

        # A second call rather than deriving the probability from the contribution
        # sum. One call would be cheaper, but it makes the service's primary
        # output a byproduct of the explanation path, where a numeric drift would
        # land on the number that matters most.
        contribution_row = model.booster.predict(matrix, pred_contribs=True)[0]
        if len(contribution_row) != len(config.FEATURE_ORDER) + 1:
            raise RuntimeError(
                f"pred_contribs returned {len(contribution_row)} values for "
                f"{len(config.FEATURE_ORDER)} features; expected one per feature "
                "plus a bias term. Refusing to derive reasons from a misread row."
            )

        contributions = {
            name: float(value)
            for name, value in zip(config.FEATURE_ORDER, contribution_row[:-1], strict=True)
        }
        # str(name): pandas types index labels as Hashable, not str.
        missing = {str(name): bool(value) for name, value in frame.isna().iloc[0].items()}

        band = risk_band(probability)
        reasons = principal_reasons(contributions, missing)

        # Deliberately logs no applicant financial fields: that is exactly the
        # PII that should not accumulate in log aggregation.
        logger.info(
            "prediction served",
            extra={
                "request_id": str(uuid.uuid4()),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "model_version": model.metadata.version,
                "risk_band": band,
            },
        )
        return PredictResponse(
            probability=probability,
            risk_band=band,
            model_version=model.metadata.version,
            reasons=reasons,
        )

    return application


app = create_app()
