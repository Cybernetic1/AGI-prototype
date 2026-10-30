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


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate unified AGI bootstrap contract/ledger gates.")
    parser.add_argument("--golden-lock-eval", default="rules/reports/golden_slices_lock_eval_v1.json")
    parser.add_argument("--gsm-cost", default="rules/reports/gsm_cost_diagnostics_v1.json")
    parser.add_argument("--efficiency-phase", default="rules/reports/agi_bootstrap_efficiency_phase_v2.json")
    parser.add_argument("--tinystories-gates", default="rules/reports/agi_bootstrap_tinystories_widening_gates_v1.json")
    parser.add_argument("--cross-domain", default="rules/reports/agi_bootstrap_cross_domain_transfer_v1.json")
    parser.add_argument("--output", default="rules/reports/agi_bootstrap_unified_contract_ledger_v1.json")
    args = parser.parse_args()

    lock_eval = _load_json(Path(args.golden_lock_eval))
    gsm_cost = _load_json(Path(args.gsm_cost))
    phase = _load_json(Path(args.efficiency_phase))
    ts_gates = _load_json(Path(args.tinystories_gates))
    cross = _load_json(Path(args.cross_domain))

    promoted = str(phase.get("promoted_run_id") or "")
    phase_evals = phase.get("evaluations") or []
    promoted_non_default = promoted != "baseline-default-budget"
    quality_green = all(
        bool(item.get("gates", {}).get("solve_non_degrading_vs_promoted_baseline", False))
        and bool(item.get("gates", {}).get("symbolic_non_degrading_vs_promoted_baseline", False))
        for item in phase_evals
        if isinstance(item, dict)
    )

    checks = {
        "gsm_lock_valid": bool(lock_eval.get("ok")),
        "tinystories_widening_green": bool(ts_gates.get("all_gates_green")),
        "cross_domain_transfer_green": bool(cross.get("overall_pass")),
        "gsm_budget_quality_green": quality_green,
        "gsm_annealing_promoted_non_default": promoted_non_default,
    }
    overall_green = all(checks.values())

    payload = {
        "ledger_id": "agi-bootstrap-unified-contract-ledger-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "golden_lock_eval": args.golden_lock_eval,
            "gsm_cost": args.gsm_cost,
            "efficiency_phase": args.efficiency_phase,
            "tinystories_gates": args.tinystories_gates,
            "cross_domain": args.cross_domain,
        },
        "checks": checks,
        "overall_green": overall_green,
        "highlights": {
            "promoted_run_id": promoted,
            "gsm_family_count": gsm_cost.get("family_count"),
            "top_gsm_cost_family": (gsm_cost.get("top_cost_families") or [{}])[0].get("family"),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved unified contract ledger to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
