#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-/home/yky/misc-programs/.venv}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
MAX_CYCLES="${MAX_CYCLES:-2}"
SUITE="${SUITE:-combined_widening}"
PROFILE="${PROFILE:-hybrid}"
SAR_ENABLED="${SAR_ENABLED:-1}"
SAR_CONFIG_FILE="${SAR_CONFIG_FILE:-}"
SAR_WEIGHT="${SAR_WEIGHT:-0.2}"
TEACHER_PROPOSALS="${TEACHER_PROPOSALS:-}"
TEACHER_ENABLE_INGESTION="${TEACHER_ENABLE_INGESTION:-0}"
TEACHER_AB_ARM="${TEACHER_AB_ARM:-treatment}"
TEACHER_GREEDY_BY_SCORE="${TEACHER_GREEDY_BY_SCORE:-1}"
TEACHER_MIN_SCORE="${TEACHER_MIN_SCORE:-0.0}"
TEACHER_OLLAMA_URL="${TEACHER_OLLAMA_URL:-}"
TEACHER_OLLAMA_MODEL="${TEACHER_OLLAMA_MODEL:-}"
TEACHER_OLLAMA_TIMEOUT_SEC="${TEACHER_OLLAMA_TIMEOUT_SEC:-180}"
ILP_ENABLE_LEARNING_STAGE="${ILP_ENABLE_LEARNING_STAGE:-0}"
ILP_PRESCORE_MIN="${ILP_PRESCORE_MIN:-0.45}"
ILP_CANDIDATE_TOP_N="${ILP_CANDIDATE_TOP_N:-6}"
ILP_MAX_CANDIDATES_PER_FAMILY="${ILP_MAX_CANDIDATES_PER_FAMILY:-3}"
ILP_INCLUDE_AGI_TRANSFER_FAMILIES="${ILP_INCLUDE_AGI_TRANSFER_FAMILIES:-1}"
ILP_LLM_ENABLE_VALIDATOR="${ILP_LLM_ENABLE_VALIDATOR:-0}"
ILP_LLM_VALIDATE_MAX="${ILP_LLM_VALIDATE_MAX:-2}"
ILP_LLM_LATENCY_BUDGET_SEC="${ILP_LLM_LATENCY_BUDGET_SEC:-12.0}"
ILP_LLM_TIMEOUT_SEC="${ILP_LLM_TIMEOUT_SEC:-20}"
ILP_LLM_OLLAMA_URL="${ILP_LLM_OLLAMA_URL:-}"
ILP_LLM_OLLAMA_MODEL="${ILP_LLM_OLLAMA_MODEL:-}"
ILP_REGISTER_PROPOSALS="${ILP_REGISTER_PROPOSALS:-0}"
ILP_MAX_REGISTERED_PER_CASE="${ILP_MAX_REGISTERED_PER_CASE:-2}"
ILP_UTILITY_PATH="${ILP_UTILITY_PATH:-rules/reports/ilp_rule_utility_v1.json}"

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
  run_args=(
    --suite "$SUITE"
    --profiles "$PROFILE"
    --max-cycles "$MAX_CYCLES"
    --quiet
    --output "$report"
  )
  if [[ "$SAR_ENABLED" == "1" ]]; then
    run_args+=(--sar-enabled --sar-weight "$SAR_WEIGHT")
  fi
  if [[ -n "$SAR_CONFIG_FILE" ]]; then
    run_args+=(--sar-config-file "$SAR_CONFIG_FILE")
  fi
  if [[ -n "$TEACHER_PROPOSALS" ]]; then
    run_args+=(--teacher-proposals "$TEACHER_PROPOSALS")
  fi
  if [[ "$TEACHER_ENABLE_INGESTION" == "1" ]]; then
    run_args+=(--teacher-enable-ingestion --teacher-ab-arm "$TEACHER_AB_ARM" --teacher-min-score "$TEACHER_MIN_SCORE")
    if [[ "$TEACHER_GREEDY_BY_SCORE" == "1" ]]; then
      run_args+=(--teacher-greedy-by-score)
    else
      run_args+=(--teacher-no-greedy-by-score)
    fi
  fi
  if [[ -n "$TEACHER_OLLAMA_URL" ]]; then
    run_args+=(--teacher-ollama-url "$TEACHER_OLLAMA_URL")
  fi
  if [[ -n "$TEACHER_OLLAMA_MODEL" ]]; then
    run_args+=(--teacher-ollama-model "$TEACHER_OLLAMA_MODEL")
  fi
  run_args+=(--teacher-ollama-timeout-sec "$TEACHER_OLLAMA_TIMEOUT_SEC")
  if [[ "$ILP_ENABLE_LEARNING_STAGE" == "1" ]]; then
    run_args+=(
      --ilp-enable-learning-stage
      --ilp-prescore-min "$ILP_PRESCORE_MIN"
      --ilp-candidate-top-n "$ILP_CANDIDATE_TOP_N"
      --ilp-max-candidates-per-family "$ILP_MAX_CANDIDATES_PER_FAMILY"
      --ilp-utility-path "$ILP_UTILITY_PATH"
      --ilp-llm-validate-max "$ILP_LLM_VALIDATE_MAX"
      --ilp-llm-latency-budget-sec "$ILP_LLM_LATENCY_BUDGET_SEC"
      --ilp-llm-timeout-sec "$ILP_LLM_TIMEOUT_SEC"
      --ilp-max-registered-per-case "$ILP_MAX_REGISTERED_PER_CASE"
    )
    if [[ "$ILP_INCLUDE_AGI_TRANSFER_FAMILIES" == "1" ]]; then
      run_args+=(--ilp-include-agi-transfer-families)
    fi
    if [[ "$ILP_LLM_ENABLE_VALIDATOR" == "1" ]]; then
      run_args+=(--ilp-llm-enable-validator)
    fi
    if [[ -n "$ILP_LLM_OLLAMA_URL" ]]; then
      run_args+=(--ilp-llm-ollama-url "$ILP_LLM_OLLAMA_URL")
    fi
    if [[ -n "$ILP_LLM_OLLAMA_MODEL" ]]; then
      run_args+=(--ilp-llm-ollama-model "$ILP_LLM_OLLAMA_MODEL")
    fi
    if [[ "$ILP_REGISTER_PROPOSALS" == "1" ]]; then
      run_args+=(--ilp-register-proposals)
    fi
  fi

  if python "$ROOT_DIR/rules/run_bootstrap_baseline.py" "${run_args[@]}" >>"$RUN_LOG" 2>&1; then
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
