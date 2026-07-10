#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

"${PYTHON:-python3}" benchmark.py --device "${DEVICE:-cuda}" --target "${TARGET:-both}" --warmup "${WARMUP:-20}" --repeat "${REPEAT:-100}"
