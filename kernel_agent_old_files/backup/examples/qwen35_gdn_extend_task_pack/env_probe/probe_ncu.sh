#!/usr/bin/env bash
set -euo pipefail

which ncu
ncu --version

if [ "${1:-}" != "" ]; then
  ncu --set full --target-processes all "$@"
fi
