#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-/home/yky/misc-programs/.venv}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
MAX_CYCLES="${MAX_CYCLES:-2}"
SUITE="${SUITE:-combined_widening}"
PROFILE="${PROFILE:-hybrid}"

REPORT_DIR="$ROOT_DIR/rules/reports/background_loop"
RUN_LOG="$REPORT_DIR/runner.log"
METRICS_JSONL="$REPORT_DIR/metrics.jsonl"

mkdir -p "$REPORT_DIR"

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] guarded background loop start suite=$SUITE profile=$PROFILE interval=${INTERVAL_SECONDS}s" | tee -a "$RUN_LOG"

while true; do
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  report="$REPORT_DIR/run_${stamp}.json"

  if python "$ROOT_DIR/rules/run_bootstrap_baseline.py" \
      --suite "$SUITE" \
      --profiles "$PROFILE" \
      --max-cycles "$MAX_CYCLES" \
      --quiet \
      --output "$report" >>"$RUN_LOG" 2>&1; then
    python - "$report" "$stamp" "$METRICS_JSONL" <<'PY'
import json
import pathlib
import sys

report_path = pathlib.Path(sys.argv[1])
stamp = sys.argv[2]
metrics_path = pathlib.Path(sys.argv[3])

payload = json.loads(report_path.read_text(encoding="utf-8"))
profile = (payload.get("profiles") or [{}])[0]
metrics = {
    "timestamp": stamp,
    "suite": payload.get("suite"),
    "profile": profile.get("profile"),
    "total_cases": profile.get("total_cases"),
    "solved_cases": profile.get("solved_cases"),
    "symbolic_valid_cases": profile.get("symbolic_valid_cases"),
    "solve_rate": (profile.get("capability") or {}).get("solve_rate"),
    "symbolic_valid_rate": (profile.get("symbolic_validity") or {}).get("symbolic_valid_rate"),
    "wall_time_sec_total": (profile.get("compute_cost") or {}).get("wall_time_sec_total"),
    "wall_time_sec_per_case": (profile.get("compute_cost") or {}).get("wall_time_sec_per_case"),
    "wall_time_sec_per_valid_solve": (profile.get("compute_cost") or {}).get("wall_time_sec_per_valid_solve"),
}
metrics_path.parent.mkdir(parents=True, exist_ok=True)
with metrics_path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(metrics, ensure_ascii=True) + "\n")
print(json.dumps({"status": "ok", "metrics": metrics}, ensure_ascii=True))
PY
  else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] baseline run failed for report=$report" | tee -a "$RUN_LOG"
  fi

  sleep "$INTERVAL_SECONDS"
done
