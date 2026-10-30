#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONTRACT_PATH="rules/bootstrap_invariant_curriculum_widening.json"
OUTPUT_PATH="rules/reports/agi_bootstrap_curriculum_widening_v1.json"

"$PYTHON_BIN" rules/run_bootstrap_invariant_curriculum.py \
  --contract "$CONTRACT_PATH" \
  --check-only \
  --print-commands

"$PYTHON_BIN" rules/run_bootstrap_invariant_curriculum.py \
  --contract "$CONTRACT_PATH" \
  --output "$OUTPUT_PATH"

echo "Saved widening report to $OUTPUT_PATH"
