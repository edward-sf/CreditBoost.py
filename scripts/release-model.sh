#!/usr/bin/env bash
# scripts/release-model.sh — publish a locally trained model as a GitHub Release
# and rewrite the lockfile that pins it.
#
# Usage:
#   creditboost-train --data data/application_train.csv --provenance production
#   ./scripts/release-model.sh 0.2.0
#
# Training stays a manual, local, credentialed step: the Kaggle dataset is
# gitignored and CI never sees it. This script only publishes what training
# produced.
set -euo pipefail

VERSION="${1:-}"
if [ -z "${VERSION}" ]; then
  echo "usage: $0 <version>   e.g. $0 0.2.0" >&2
  exit 1
fi

TAG="model-v${VERSION}"
MODEL="models/model.json"
METADATA="models/model_meta.json"

for asset in "${MODEL}" "${METADATA}"; do
  if [ ! -f "${asset}" ]; then
    echo "missing ${asset} — run creditboost-train first" >&2
    exit 1
  fi
done

PROVENANCE=$(python -c "import json,sys; print(json.load(open('${METADATA}'))['provenance'])")
if [ "${PROVENANCE}" != "production" ]; then
  echo "refusing to release a '${PROVENANCE}' artifact; retrain with --provenance production" >&2
  exit 1
fi

META_VERSION=$(python -c "import json,sys; print(json.load(open('${METADATA}'))['version'])")
if [ "${META_VERSION}" != "${VERSION}" ]; then
  echo "artifact version is ${META_VERSION} but you asked to release ${VERSION}." >&2
  echo "Bump config.MODEL_VERSION and retrain, or release ${META_VERSION} instead." >&2
  exit 1
fi

echo "==> creating release ${TAG}"
gh release create "${TAG}" \
  --title "Model ${VERSION}" \
  --notes "XGBoost default-risk model ${VERSION}, trained on Home Credit application_train.csv." \
  "${MODEL}" "${METADATA}"

echo "==> rewriting the lockfile"
creditboost-artifact lock --tag "${TAG}"

echo
echo "==> done. Review and commit the lockfile:"
echo "    git diff models/model.lock.json"
echo "    git add models/model.lock.json && git commit -m 'chore: pin model ${TAG}'"
