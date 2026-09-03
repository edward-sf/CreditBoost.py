"""Training CLI.

Run against the real dataset:
    creditboost-train --data data/application_train.csv

The dataset is obtained manually from Kaggle and is gitignored; this command is
never run in CI.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config
from .artifact import save
from .data import file_sha256, load_training_frame, split
from .fairness import evaluate, failing_attributes
from .features import transform
from .schema import ModelMetadata

logger = logging.getLogger(__name__)

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": config.RANDOM_SEED,
}
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30


def _matrix(frame: pd.DataFrame) -> xgb.DMatrix:
    return xgb.DMatrix(
        transform(frame),
        label=frame[config.TARGET_COLUMN],
        enable_categorical=True,
    )


def fit(
    train_frame: pd.DataFrame, valid_frame: pd.DataFrame
) -> tuple[xgb.Booster, dict[str, float]]:
    """Train and evaluate. No scale_pos_weight: it inflates probabilities away
    from the true base rate, which would break calibration, make the Brier score
    meaningless, and invalidate the configured risk-band thresholds."""
    dtrain, dvalid = _matrix(train_frame), _matrix(valid_frame)

    booster = xgb.train(
        PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    # Trim to the best iteration so serving needs no iteration_range and cannot
    # accidentally score with the overfit tail.
    booster = booster[: booster.best_iteration + 1]

    y_valid = valid_frame[config.TARGET_COLUMN]
    probabilities = booster.predict(dvalid)

    metrics = {
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "pr_auc": float(average_precision_score(y_valid, probabilities)),
        "brier": float(brier_score_loss(y_valid, probabilities)),
    }
    return booster, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="creditboost-train")
    parser.add_argument("--data", type=Path, default=config.DEFAULT_DATA_PATH)
    parser.add_argument("--model-out", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--metadata-out", type=Path, default=config.METADATA_PATH)
    parser.add_argument(
        "--min-auc",
        type=float,
        default=config.MIN_VALIDATION_AUC,
        help="Refuse to write a model below this validation ROC-AUC.",
    )
    parser.add_argument(
        "--provenance",
        choices=("fixture", "production"),
        default="production",
        help="Records whether this model was trained on real data or the test fixture.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    frame = load_training_frame(args.data)
    train_frame, valid_frame = split(frame)
    logger.info("training on %d rows, validating on %d", len(train_frame), len(valid_frame))

    booster, metrics = fit(train_frame, valid_frame)
    logger.info("metrics: %s", metrics)

    if metrics["roc_auc"] < args.min_auc:
        logger.error(
            "validation ROC-AUC %.4f is below the floor %.4f; no artifact written",
            metrics["roc_auc"],
            args.min_auc,
        )
        return 1

    # fit() does not return validation predictions, so they are recomputed here.
    # One extra pass over the validation split is cheap next to changing fit()'s
    # signature, which tests and callers depend on.
    fairness = evaluate(valid_frame, booster.predict(_matrix(valid_frame)).tolist())
    failures = failing_attributes(fairness)
    if failures:
        for attribute in failures:
            logger.error(
                "adverse impact ratio %.4f for %s is below the floor %.2f; no artifact written",
                attribute.adverse_impact_ratio,
                attribute.attribute,
                config.MIN_ADVERSE_IMPACT_RATIO,
            )
            logger.error(
                "    %d group(s) eligible (n >= %d); groups below the minimum size "
                "are excluded and not shown",
                len(attribute.groups),
                fairness.min_group_size,
            )
            for group in attribute.groups:
                logger.error(
                    "    %s: adverse rate %.4f (n=%d)",
                    group.group,
                    group.adverse_rate,
                    group.n,
                )
        return 1

    logger.info(
        "adverse impact ratios: %s",
        {a.attribute: a.adverse_impact_ratio for a in fairness.attributes},
    )
    unmeasured = [a for a in fairness.attributes if a.adverse_impact_ratio is None]
    if unmeasured:
        for attribute in unmeasured:
            logger.warning(
                "%s was not measured: %s", attribute.attribute, attribute.unmeasured_reason
            )

    metadata = ModelMetadata(
        version=config.MODEL_VERSION,
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        dataset_sha256=file_sha256(args.data),
        n_train_rows=len(train_frame),
        feature_order=list(config.FEATURE_ORDER),
        metrics=metrics,
        xgboost_version=xgb.__version__,
        provenance=args.provenance,
        fairness=fairness,
    )
    save(booster, metadata, args.model_out, args.metadata_out)
    logger.info("wrote %s and %s", args.model_out, args.metadata_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
