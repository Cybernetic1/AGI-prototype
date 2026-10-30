#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _hybrid_metrics(ab_report: dict[str, Any]) -> dict[str, float]:
    rows = ab_report.get("profiles")
    if not isinstance(rows, list):
        raise ValueError("missing profiles in AB report")
    hybrid = next((row for row in rows if isinstance(row, dict) and row.get("profile") == "hybrid"), None)
    if not isinstance(hybrid, dict):
        raise ValueError("hybrid profile not found in AB report")
    metrics = hybrid.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("hybrid metrics missing")
    return {
        "solve_delta_mean": float(metrics["solve_rate"]["delta_treatment_minus_control"]["mean"]),
        "symbolic_delta_mean": float(metrics["symbolic_valid_rate"]["delta_treatment_minus_control"]["mean"]),
        "wall_per_valid_delta_mean": float(
            metrics["wall_time_sec_per_valid_solve"]["delta_treatment_minus_control"]["mean"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate cross-domain transfer from GSM and TinyStories AB studies.")
    parser.add_argument(
        "--gsm-ab",
        default="rules/reports/efficiency_phase/baseline-default-budget.ab_repeated.json",
    )
    parser.add_argument(
        "--tinystories-ab",
        default="rules/reports/agi_bootstrap_ab_tinystories_symbolic_v1.json",
    )
    parser.add_argument(
        "--tinystories-gates",
        default="rules/reports/agi_bootstrap_tinystories_widening_gates_v1.json",
    )
    parser.add_argument("--output", default="rules/reports/agi_bootstrap_cross_domain_transfer_v1.json")
    args = parser.parse_args()

    gsm_path = Path(args.gsm_ab)
    ts_path = Path(args.tinystories_ab)
    gates_path = Path(args.tinystories_gates)

    gsm = _load_json(gsm_path)
    tinystories = _load_json(ts_path)
    gates = _load_json(gates_path)

    gsm_hybrid = _hybrid_metrics(gsm)
    ts_hybrid = _hybrid_metrics(tinystories)

    checks = {
        "gsm_quality_non_degrading": gsm_hybrid["solve_delta_mean"] >= 0.0 and gsm_hybrid["symbolic_delta_mean"] >= 0.0,
        "tinystories_quality_non_degrading": ts_hybrid["solve_delta_mean"] >= 0.0 and ts_hybrid["symbolic_delta_mean"] >= 0.0,
        "tinystories_widening_gates_green": bool(gates.get("all_gates_green")),
    }
    overall_pass = all(checks.values())

    payload = {
        "eval_id": "agi-bootstrap-cross-domain-transfer-eval-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "gsm_ab": str(gsm_path),
            "tinystories_ab": str(ts_path),
            "tinystories_gates": str(gates_path),
        },
        "domain_metrics": {
            "gsm_hybrid": gsm_hybrid,
            "tinystories_hybrid": ts_hybrid,
        },
        "checks": checks,
        "overall_pass": overall_pass,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved cross-domain transfer evaluation to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
