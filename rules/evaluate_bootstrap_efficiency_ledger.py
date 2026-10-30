#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate AGI bootstrap reports against the efficiency ledger.")
    parser.add_argument("--ledger", default="rules/bootstrap_efficiency_ledger.json")
    parser.add_argument("--milestone-review", default="rules/reports/agi_bootstrap_milestone_review_v1.json")
    parser.add_argument("--ab-repeated", default="rules/reports/agi_bootstrap_ab_efficiency_repeated_v1.json")
    parser.add_argument("--widening-report", default="rules/reports/agi_bootstrap_curriculum_widening_v1.json")
    parser.add_argument("--output", default="rules/reports/agi_bootstrap_efficiency_ledger_eval_v1.json")
    args = parser.parse_args()

    ledger = _load(Path(args.ledger))
    review = _load(Path(args.milestone_review))
    ab_repeated = _load(Path(args.ab_repeated))
    widening = _load(Path(args.widening_report))

    t = ledger["thresholds"]
    ab_t = t["ab_repeated"]
    f_t = t["efficiency_factors"]

    review_gap = review["gap_to_target"]
    achieved_factor = float(review_gap["hybrid_achieved_factor_vs_baseline"])

    ab_macro = ab_repeated["macro_delta_treatment_minus_control"]
    solve_delta_mean = float(ab_macro["solve_rate_mean"]["mean"])
    symbolic_delta_mean = float(ab_macro["symbolic_valid_rate_mean"]["mean"])
    wall_delta_mean = float(ab_macro["wall_time_sec_total_sum"]["mean"])

    checks = {
      "invariant_all_gates_green": bool(review["aggregated_findings"]["invariant"]["hybrid"]["symbolic_valid_rate"] >= 1.0 and review["aggregated_findings"]["invariant"]["hybrid"]["solve_rate"] >= 1.0)
      if t["required_all_gates_green"] else True,
      "widening_all_gates_green": bool(widening.get("all_gates_green")) if t["required_widening_all_gates_green"] else True,
      "ab_solve_non_degrading": solve_delta_mean >= -float(ab_t["max_solve_rate_drop_mean"]),
      "ab_symbolic_non_degrading": symbolic_delta_mean >= -float(ab_t["max_symbolic_valid_rate_drop_mean"]),
      "ab_wall_time_delta_bound": wall_delta_mean <= float(ab_t["max_wall_time_delta_mean_sec"]),
      "phase_min_factor_met": achieved_factor >= float(f_t["phase_min_factor_vs_hybrid_baseline"]),
      "stretch_factor_met": achieved_factor >= float(f_t["stretch_factor_vs_hybrid_baseline"]),
    }

    overall_pass = all(checks.values())
    payload = {
      "eval_id": "agi-bootstrap-efficiency-ledger-eval-v1",
      "generated_at": datetime.now(timezone.utc).isoformat(),
      "sources": {
        "ledger": args.ledger,
        "milestone_review": args.milestone_review,
        "ab_repeated": args.ab_repeated,
        "widening_report": args.widening_report
      },
      "checks": checks,
      "metrics": {
        "hybrid_achieved_factor_vs_baseline": achieved_factor,
        "long_horizon_target_factor": float(f_t["long_horizon_target_factor"]),
        "ab_solve_delta_mean": solve_delta_mean,
        "ab_symbolic_delta_mean": symbolic_delta_mean,
        "ab_wall_time_delta_mean_sec": wall_delta_mean
      },
      "overall_pass": overall_pass
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved ledger evaluation to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
