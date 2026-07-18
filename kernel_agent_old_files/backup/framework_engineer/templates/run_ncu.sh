#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CASE_ID="${1:-}"
if [ -z "$CASE_ID" ]; then
  echo "usage: bash scripts/run_ncu.sh <case_id>" >&2
  exit 2
fi

ncu --set full --target-processes all "${PYTHON:-python3}" benchmark.py --case-id "$CASE_ID" --device "${DEVICE:-cuda}" --target "${TARGET:-candidate}" --warmup "${WARMUP:-5}" --repeat "${REPEAT:-20}"
