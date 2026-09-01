#!/usr/bin/env bash
# scripts/smoke.sh — verify a running container actually serves predictions.
# A plain `docker build` succeeds even if the model never made it into the
# image; this is what catches that.
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

echo "==> waiting for ${BASE_URL}/health"
for _ in $(seq 1 30); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then break; fi
  sleep 1
done

echo "==> GET /health"
curl -fsS "${BASE_URL}/health" | tee /tmp/health.json
grep -q '"status":"ok"' /tmp/health.json

echo "==> POST /predict (thin-file borrower, no external scores)"
curl -fsS -X POST "${BASE_URL}/predict" \
  -H 'Content-Type: application/json' \
  -d '{"AMT_INCOME_TOTAL": 100000, "AMT_CREDIT": 400000, "DAYS_BIRTH": -12000}' \
  | tee /tmp/predict.json
grep -q '"risk_band"' /tmp/predict.json
grep -q '"model_version"' /tmp/predict.json

PROBABILITY=$(grep -o '"probability":[^,}]*' /tmp/predict.json | head -1 | cut -d: -f2)
if [ -z "${PROBABILITY}" ]; then
  echo "expected a \"probability\" field in the /predict response, but found none" >&2
  exit 1
fi
echo "${PROBABILITY}" | awk '{ exit ($1 ~ /^-?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$/ && $1 >= 0 && $1 <= 1) ? 0 : 1 }' \
  || { echo "expected \"probability\" to be a number in [0, 1], but got \"${PROBABILITY}\"" >&2; exit 1; }

echo "==> smoke test passed"
