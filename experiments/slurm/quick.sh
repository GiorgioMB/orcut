#!/bin/bash
# Smoke test of the whole pipeline at toy size (a few minutes, no downloads, no GRC needed).
set -euo pipefail
cd "$(dirname "$0")/../.."
python experiments/main.py --quick --threads "${1:-4}" --out results-quick
