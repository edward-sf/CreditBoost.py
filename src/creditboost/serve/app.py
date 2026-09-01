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

        matrix = xgb.DMatrix(transform([request.model_dump()]), enable_categorical=True)
        probability = float(model.booster.predict(matrix)[0])
        band = risk_band(probability)

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
        )

    return application


app = create_app()
