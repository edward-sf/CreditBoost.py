"""The creditboost-search command.

Read-only by design: it prints the frontier and what the selection rule would
choose, and writes nothing. Search and training deliberately do not communicate
through a file on disk -- a stored frontier could be stamped onto a model it did
not select -- so this command exists purely for inspection, and
`creditboost-train --search` is the path that produces an artifact.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import config, search
from .data import load_training_frame, split

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="creditboost-search")
    parser.add_argument("--data", type=Path, default=config.DEFAULT_DATA_PATH)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    frame = load_training_frame(args.data)
    train_frame, _ = split(frame)
    ranking = search.rank(train_frame)

    print(f"ranked at a matched approval rate of {ranking.target_approval_rate:.4f}")
    print(f"{'candidate':38s} {'AUC':>8s} {'min AIR':>9s}  features")
    for candidate in ranking.candidates:
        if candidate.failed_reason is None:
            print(
                f"{candidate.name:38s} {candidate.roc_auc:8.4f} "
                f"{candidate.min_adverse_impact_ratio:9.4f}  {candidate.n_features}"
            )
        else:
            print(f"{candidate.name:38s} {'--':>8s} {'--':>9s}  {candidate.failed_reason}")

    try:
        chosen = search.select(ranking.candidates, baseline=search.BASELINE.name)
    except search.BaselineMissingError:
        # No candidate could be scored -- on a small dataset no group reaches
        # MIN_FAIRNESS_GROUP_SIZE. The frontier above shows each candidate's
        # failed_reason, so nothing more can be said.
        print("\nno candidate could be scored; nothing could be selected")
        return 0

    print(f"\nselection rule would choose: {chosen}")
    if chosen == search.BASELINE.name:
        print("no less discriminatory alternative was found within the AUC budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
