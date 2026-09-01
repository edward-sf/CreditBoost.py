"""Artifact persistence and the startup gate against train/serve skew."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import xgboost as xgb

from . import config
from .schema import ModelMetadata

logger = logging.getLogger(__name__)


class ArtifactError(RuntimeError):
    """Base class for all artifact-loading errors.

    `load()` is called from Task 8's FastAPI lifespan handler with no `except`
    clause of its own: any exception here propagates, the process exits
    non-zero, and the container never accepts traffic. Every failure path in
    this module must therefore raise rather than return a questionable
    LoadedModel.
    """


class FeatureOrderMismatchError(ArtifactError):
    """The artifact was trained on a different feature layout than this code emits."""


class XGBoostVersionMismatchError(ArtifactError):
    """The artifact was trained with an xgboost major version this build cannot trust."""


class CorruptModelError(ArtifactError):
    """The model file exists but xgboost could not parse it."""


@dataclass(frozen=True)
class LoadedModel:
    booster: xgb.Booster
    metadata: ModelMetadata


def save(
    booster: xgb.Booster,
    metadata: ModelMetadata,
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.METADATA_PATH,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path))
    metadata_path.write_text(metadata.model_dump_json(indent=2) + "\n")


def _first_disagreement(actual: list[str], expected: list[str]) -> str | None:
    """Describe the first index at which two feature-order lists diverge.

    Returns None when every position common to both lists agrees (i.e. one is
    a strict prefix of the other and only their lengths differ).
    """
    for i, (a, e) in enumerate(zip(actual, expected, strict=False)):
        if a != e:
            return f"first disagreement at index {i}: artifact has {a!r}, code expects {e!r}"
    return None


def _major_version(version_string: str) -> int | None:
    """Best-effort leading-component parse. Returns None for anything uninterpretable."""
    try:
        return int(version_string.split(".")[0])
    except (ValueError, IndexError):
        return None


def load(
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.METADATA_PATH,
) -> LoadedModel:
    """Load the artifact, refusing any model whose features disagree with ours."""
    if not model_path.exists():
        raise FileNotFoundError(f"model artifact not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"model metadata not found: {metadata_path}")

    metadata = ModelMetadata.model_validate_json(metadata_path.read_text())

    expected = list(config.FEATURE_ORDER)
    if metadata.feature_order != expected:
        disagreement = _first_disagreement(metadata.feature_order, expected)
        length_note = (
            f"artifact has {len(metadata.feature_order)} features, code expects {len(expected)}"
            if len(metadata.feature_order) != len(expected)
            else None
        )
        detail = "; ".join(part for part in (disagreement, length_note) if part)
        raise FeatureOrderMismatchError(
            "artifact feature order does not match this build's FEATURE_ORDER; "
            f"{detail}. Retrain the model against this code."
        )

    artifact_major = _major_version(metadata.xgboost_version)
    running_major = _major_version(xgb.__version__)
    if artifact_major is None or running_major is None:
        logger.warning(
            "could not parse xgboost version for comparison (artifact=%r, running=%r); "
            "continuing without a version check",
            metadata.xgboost_version,
            xgb.__version__,
        )
    elif artifact_major != running_major:
        raise XGBoostVersionMismatchError(
            f"artifact was trained with xgboost {metadata.xgboost_version} (major "
            f"{artifact_major}), but this build runs xgboost {xgb.__version__} (major "
            f"{running_major}). xgboost can generally load older models but not newer "
            "ones, so this artifact is not trustworthy here. Retrain the model against "
            "this build's xgboost."
        )
    elif metadata.xgboost_version != xgb.__version__:
        logger.warning(
            "artifact xgboost version %r differs from running xgboost %r; same major "
            "version, continuing",
            metadata.xgboost_version,
            xgb.__version__,
        )

    booster = xgb.Booster()
    try:
        booster.load_model(str(model_path))
    except xgb.core.XGBoostError as err:
        raise CorruptModelError(
            f"model artifact at {model_path} could not be loaded by xgboost: {err}"
        ) from err
    return LoadedModel(booster=booster, metadata=metadata)
