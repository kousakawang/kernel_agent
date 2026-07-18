#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

"${PYTHON:-python3}" correctness_test.py --device "${DEVICE:-cuda}" --mode "${CORRECTNESS_MODE:-snapshot-golden}"
