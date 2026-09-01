"""Training-time dataset access. Never imported by the serving package."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


class MissingColumnsError(ValueError):
    """The CSV lacks columns the transform requires."""


def file_sha256(path: Path) -> str:
    """Content hash, recorded in metadata so a model traces to its training data."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_frame(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"training data not found: {path}")

    frame = pd.read_csv(path)

    required = {*config.REQUEST_FIELDS, config.TARGET_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise MissingColumnsError(f"training data is missing columns: {', '.join(missing)}")

    return frame


def split(
    frame: pd.DataFrame,
    seed: int = config.RANDOM_SEED,
    validation_size: float = config.VALIDATION_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split. Stratification is required, not optional: positives are
    roughly 8% of rows, and an unstratified split can skew the validation base
    rate enough to make AUC unstable."""
    train, valid = train_test_split(
        frame,
        test_size=validation_size,
        random_state=seed,
        stratify=frame[config.TARGET_COLUMN],
    )
    return train, valid
