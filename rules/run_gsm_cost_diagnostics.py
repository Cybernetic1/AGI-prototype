#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_RUNNER = REPO_ROOT / "rules" / "run_bootstrap_baseline.py"
DEFAULT_LOCK = REPO_ROOT / "rules" / "bootstrap_golden_slices_v1.json"
DEFAULT_CONTRACT = REPO_ROOT / "rules" / "bootstrap_invariant_curriculum_widening.json"
GSM_SUITES = {"finality_regression", "template_generalization"}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _resolve_gsm_slices(lock: dict[str, Any], contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = contract.get("slices")
    if not isinstance(rows, list):
        raise ValueError("contract missing slices")
    by_id = {
        str(row.get("id")): row
        for row in rows
        if isinstance(row, dict) and row.get("id") and str(row.get("suite") or "") in GSM_SUITES
    }

    selected = []
    for slice_id in lock.get("golden_slices", []):
        item = by_id.get(str(slice_id))
        if item:
            selected.append(item)
    if not selected:
        raise ValueError("no GSM slices found in lock/contract intersection")
    return selected


def _run_family(
    *,
    python_exec: str,
    family: str,
    max_cycles: int,
    seed: int,
    outdir: Path,
) -> dict[str, Any]:
    out = outdir / f"gsm_cost_diag_{family}.json"
    cmd = [
        python_exec,
        str(BASELINE_RUNNER),
        "--suite",
        "combined_widening",
        "--profiles",
        "hybrid",
        "--max-cycles",
        str(max_cycles),
        "--families",
        family,
        "--seed",
        str(seed),
        "--quiet",
        "--output",
        str(out),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    report = _load_json(out)
    profile = next((p for p in report.get("profiles", []) if p.get("profile") == "hybrid"), None)
    if not isinstance(profile, dict):
        raise ValueError(f"hybrid profile missing in report for family {family}")
    result_rows = profile.get("results") or []
    belief_count = 0
    cycles_executed = 0
    if result_rows and isinstance(result_rows[0], dict):
        belief_count = len(result_rows[0].get("belief_values") or [])
        cycles_executed = int(result_rows[0].get("cycles_executed") or 0)
    return {
        "family": family,
        "total_cases": int(profile.get("total_cases", 0)),
        "solved_cases": int(profile.get("solved_cases", 0)),
        "symbolic_valid_cases": int(profile.get("symbolic_valid_cases", 0)),
        "solved_and_symbolic_valid_cases": int(profile.get("solved_and_symbolic_valid_cases", 0)),
        "solve_rate": float(profile["capability"]["solve_rate"]),
        "symbolic_valid_rate": float(profile["symbolic_validity"]["symbolic_valid_rate"]),
        "wall_time_sec_total": float(profile["compute_cost"]["wall_time_sec_total"]),
        "wall_time_sec_per_case": float(profile["compute_cost"]["wall_time_sec_per_case"]),
        "wall_time_sec_per_valid_solve": profile["compute_cost"]["wall_time_sec_per_valid_solve"],
        "belief_count_proxy": int(belief_count),
        "cycles_executed": int(cycles_executed),
        "family_report_path": str(out),
    }


def _diag_score(row: dict[str, Any]) -> float:
    wall = row.get("wall_time_sec_per_valid_solve")
    if wall is None:
        return math.inf
    return float(wall)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GSM per-family cost diagnostics on locked golden slices.")
    parser.add_argument("--golden-lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-cycles", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--outdir", default="rules/reports/gsm_cost_diagnostics")
    parser.add_argument("--output", default="rules/reports/gsm_cost_diagnostics_v1.json")
    args = parser.parse_args()

    lock_path = Path(args.golden_lock)
    if not lock_path.is_absolute():
        lock_path = REPO_ROOT / lock_path
    contract_path = Path(args.contract)
    if not contract_path.is_absolute():
        contract_path = REPO_ROOT / contract_path

    lock = _load_json(lock_path)
    contract = _load_json(contract_path)
    slices = _resolve_gsm_slices(lock, contract)

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = REPO_ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for slice_row in slices:
        slice_id = str(slice_row.get("id"))
        suite = str(slice_row.get("suite"))
        for family in slice_row.get("families", []):
            fam_id = str(family)
            diag = _run_family(
                python_exec=str(args.python),
                family=fam_id,
                max_cycles=int(args.max_cycles),
                seed=int(args.seed),
                outdir=outdir,
            )
            diag["slice_id"] = slice_id
            diag["suite"] = suite
            rows.append(diag)

    ranked = sorted(rows, key=_diag_score, reverse=True)
    top = ranked[: max(1, int(args.top_k))]

    payload = {
        "diagnostic_id": "gsm-golden-cost-diagnostics-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "golden_lock": str(lock_path),
        "contract": str(contract_path),
        "slice_ids": [str(item.get("id")) for item in slices],
        "seed": int(args.seed),
        "max_cycles": int(args.max_cycles),
        "family_count": len(rows),
        "top_k": int(args.top_k),
        "top_cost_families": top,
        "all_families": ranked,
        "notes": {
            "branch_proxy": "belief_count_proxy uses number of numeric Belief candidates emitted for the case as a branch-factor proxy.",
            "sorting": "descending by wall_time_sec_per_valid_solve; unsolved/invalid rows rank first with +inf score.",
        },
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved GSM cost diagnostics to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
