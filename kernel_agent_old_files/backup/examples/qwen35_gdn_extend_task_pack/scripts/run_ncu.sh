#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GROUP_ID="${1:-}"
SAMPLE_ID="${2:-}"
if [ -z "$GROUP_ID" ]; then
  echo "usage: bash scripts/run_ncu.sh <group_id> [sample_id]" >&2
  exit 2
fi

args=(benchmark.py --group-id "$GROUP_ID" --device "${DEVICE:-cuda}" --target "${TARGET:-candidate}" --warmup "${WARMUP:-5}" --repeat "${REPEAT:-20}")
if [ -n "$SAMPLE_ID" ]; then
  args+=(--sample-id "$SAMPLE_ID")
fi

ncu --set full --target-processes all "${PYTHON:-python3}" "${args[@]}"
