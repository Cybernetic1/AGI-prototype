#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/yky/misc-programs/.venv/bin/python"

# Control: transfer metarules OFF, SAR OFF, teacher ingestion OFF
AGI_ENABLE_TRANSFER_METARULES=0 "$PYTHON_BIN" rules/run_bootstrap_baseline.py \
  --suite combined \
  --profiles lt-heavy hybrid ilp-heavy \
  --max-cycles 2 \
  --families \
    buy_more_final_gap \
    donation_after_buy_more \
    buy_more_then_donation_and_spoilage \
    buy_more_then_donation_after_spoilage_long_tail \
    unit_conversion_dozen \
    percent_sequence_growth \
    percent_sequence_discount \
    distance_over_speed \
  --output rules/reports/agi_bootstrap_ab_efficiency_control.json \
  --quiet

# Treatment: transfer metarules ON + SAR ON + teacher-ingestion treatment arm ON
AGI_ENABLE_TRANSFER_METARULES=1 "$PYTHON_BIN" rules/run_bootstrap_baseline.py \
  --suite combined \
  --profiles lt-heavy hybrid ilp-heavy \
  --max-cycles 2 \
  --families \
    buy_more_final_gap \
    donation_after_buy_more \
    buy_more_then_donation_and_spoilage \
    buy_more_then_donation_after_spoilage_long_tail \
    unit_conversion_dozen \
    percent_sequence_growth \
    percent_sequence_discount \
    distance_over_speed \
  --output rules/reports/agi_bootstrap_ab_efficiency_treatment.json \
  --quiet \
  --sar-enabled \
  --sar-weight 0.25 \
  --sar-lexical-weight 0.55 \
  --sar-role-weight 0.30 \
  --sar-prior-weight 0.15 \
  --sar-gate-fail-penalty 0.05 \
  --teacher-proposals rules/reports/teacher_proposals_ab_smoke.json \
  --teacher-enable-ingestion \
  --teacher-ab-arm treatment \
  --teacher-id ab-efficiency-teacher \
  --teacher-model smoke-v1

# Optional post-processing: derive compact comparison artifacts
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

control = json.loads(Path("rules/reports/agi_bootstrap_ab_efficiency_control.json").read_text(encoding="utf-8"))
treatment = json.loads(Path("rules/reports/agi_bootstrap_ab_efficiency_treatment.json").read_text(encoding="utf-8"))
print(
    "profiles:",
    [p["profile"] for p in control["profiles"]],
    "| case_count:",
    control["case_count"],
    "| treatment_teacher_counts:",
    treatment.get("teacher_proposal_ingestion", {}).get("counts", {}),
)
PY
