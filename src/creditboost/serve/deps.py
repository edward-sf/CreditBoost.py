"""Process-wide loaded-model state.

The artifact is read once at startup rather than per request: loading is slow
and the booster is immutable once loaded.
"""

from __future__ import annotations

from pathlib import Path

from .. import config
from ..artifact import LoadedModel, load

_model: LoadedModel | None = None


def load_model(
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.METADATA_PATH,
) -> LoadedModel:
    global _model
    _model = load(model_path, metadata_path)
    return _model


def get_model() -> LoadedModel:
    if _model is None:
        raise RuntimeError("model is not loaded; startup did not complete")
    return _model


def reset() -> None:
    global _model
    _model = None
