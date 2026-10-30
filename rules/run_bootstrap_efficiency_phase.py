#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
AB_REPEATED = REPO_ROOT / "rules" / "run_bootstrap_ab_repeated.py"

DEFAULT_CONTRACT = REPO_ROOT / "rules" / "bootstrap_invariant_curriculum_widening.json"
DEFAULT_GOLDEN_LOCK = REPO_ROOT / "rules" / "bootstrap_golden_slices_v1.json"
DEFAULT_GOLDEN_SLICES = [
    "invariant-core-finality",
    "adjacent-template-core",
    "adjacent-template-full",
]
DEFAULT_SEEDS = [11, 23, 37]
DEFAULT_PROFILES = ["lt-heavy", "hybrid", "ilp-heavy"]
DEFAULT_BUDGET_GRID = [
    {"id": "baseline-default-budget", "ilp_max_depth": 0, "ilp_max_fact_matches": 0, "ilp_max_rule_matches": 0, "ilp_max_subgoal_results": 0, "numeric_authority_strict": False},
    {"id": "depth10-subgoal8-authority", "ilp_max_depth": 10, "ilp_max_fact_matches": 0, "ilp_max_rule_matches": 0, "ilp_max_subgoal_results": 8, "numeric_authority_strict": True},
    {"id": "depth8-subgoal6-authority", "ilp_max_depth": 8, "ilp_max_fact_matches": 0, "ilp_max_rule_matches": 0, "ilp_max_subgoal_results": 6, "numeric_authority_strict": True},
]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _golden_families(contract: dict[str, Any], slice_ids: list[str]) -> list[str]:
    families: list[str] = []
    family_seen = set()
    rows = contract.get("slices")
    if not isinstance(rows, list):
        raise ValueError("contract missing slices list")
    by_id = {str(row.get("id")): row for row in rows if isinstance(row, dict) and row.get("id")}
    missing = [sid for sid in slice_ids if sid not in by_id]
    if missing:
        raise ValueError(f"unknown slice ids in contract: {', '.join(missing)}")
    for sid in slice_ids:
        entry = by_id[sid]
        for fam in entry.get("families", []):
            fam_id = str(fam).strip()
            if not fam_id or fam_id in family_seen:
                continue
            family_seen.add(fam_id)
            families.append(fam_id)
    return families


def _hybrid_metrics(report: dict[str, Any]) -> dict[str, float]:
    profiles = report.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("repeated report missing profiles")
    hybrid = next((p for p in profiles if isinstance(p, dict) and p.get("profile") == "hybrid"), None)
    if not isinstance(hybrid, dict):
        raise ValueError("hybrid profile not found in repeated report")
    m = hybrid.get("metrics")
    if not isinstance(m, dict):
        raise ValueError("hybrid metrics missing")
    return {
        "solve_rate_control_mean": float(m["solve_rate"]["control"]["mean"]),
        "solve_rate_treatment_mean": float(m["solve_rate"]["treatment"]["mean"]),
        "symbolic_control_mean": float(m["symbolic_valid_rate"]["control"]["mean"]),
        "symbolic_treatment_mean": float(m["symbolic_valid_rate"]["treatment"]["mean"]),
        "wall_per_valid_control_mean": float(m["wall_time_sec_per_valid_solve"]["control"]["mean"]),
        "wall_per_valid_treatment_mean": float(m["wall_time_sec_per_valid_solve"]["treatment"]["mean"]),
    }


def _run_ab(
    *,
    python_exec: str,
    outdir: Path,
    output_path: Path,
    families: list[str],
    seeds: list[int],
    max_cycles: int,
    profiles: list[str],
    teacher_proposals: str,
    budget: dict[str, Any],
) -> dict[str, Any]:
    cmd = [
        python_exec,
        str(AB_REPEATED),
        "--suite",
        "combined_widening",
        "--families",
        *families,
        "--profiles",
        *profiles,
        "--max-cycles",
        str(max_cycles),
        "--seeds",
        *[str(s) for s in seeds],
        "--teacher-proposals",
        teacher_proposals,
        "--outdir",
        str(outdir),
        "--output",
        str(output_path),
    ]
    if int(budget.get("ilp_max_depth", 0)) > 0:
        cmd.extend(["--ilp-max-depth", str(int(budget["ilp_max_depth"]))])
    if int(budget.get("ilp_max_fact_matches", 0)) > 0:
        cmd.extend(["--ilp-max-fact-matches", str(int(budget["ilp_max_fact_matches"]))])
    if int(budget.get("ilp_max_rule_matches", 0)) > 0:
        cmd.extend(["--ilp-max-rule-matches", str(int(budget["ilp_max_rule_matches"]))])
    if int(budget.get("ilp_max_subgoal_results", 0)) > 0:
        cmd.extend(["--ilp-max-subgoal-results", str(int(budget["ilp_max_subgoal_results"]))])
    if bool(budget.get("numeric_authority_strict", False)):
        cmd.append("--numeric-authority-strict")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    return _load_json(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run focused AGI bootstrap efficiency phase on golden slices.")
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--golden-lock", default=str(DEFAULT_GOLDEN_LOCK))
    parser.add_argument("--golden-slices", nargs="+", default=DEFAULT_GOLDEN_SLICES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES, choices=DEFAULT_PROFILES)
    parser.add_argument("--max-cycles", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--teacher-proposals", default="rules/reports/teacher_proposals_ab_smoke.json")
    parser.add_argument("--outdir", default="rules/reports")
    parser.add_argument("--report-dir", default="rules/reports/efficiency_phase")
    parser.add_argument("--output", default="rules/reports/agi_bootstrap_efficiency_phase_v1.json")
    args = parser.parse_args()

    lock_path = Path(args.golden_lock)
    if not lock_path.is_absolute():
        lock_path = REPO_ROOT / lock_path
    lock_payload = {}
    if lock_path.exists():
        lock_payload = _load_json(lock_path)

    contract_path = Path(args.contract)
    if str(args.contract).strip() == str(DEFAULT_CONTRACT) and lock_payload.get("contract_path"):
        contract_path = Path(str(lock_payload.get("contract_path")))
    if not contract_path.is_absolute():
        contract_path = REPO_ROOT / contract_path
    contract = _load_json(contract_path)
    selected_slices = [str(x) for x in args.golden_slices]
    if (
        selected_slices == DEFAULT_GOLDEN_SLICES
        and isinstance(lock_payload.get("golden_slices"), list)
        and lock_payload.get("golden_slices")
    ):
        selected_slices = [str(x) for x in lock_payload.get("golden_slices")]
    families = _golden_families(contract, selected_slices)

    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    report_dir = REPO_ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = None
    evaluations: list[dict[str, Any]] = []
    promoted_id = None

    for budget in DEFAULT_BUDGET_GRID:
        run_id = str(budget["id"])
        repeated_output = report_dir / f"{run_id}.ab_repeated.json"
        report = _run_ab(
            python_exec=str(args.python),
            outdir=outdir,
            output_path=repeated_output,
            families=families,
            seeds=[int(s) for s in args.seeds],
            max_cycles=int(args.max_cycles),
            profiles=list(args.profiles),
            teacher_proposals=str(args.teacher_proposals),
            budget=budget,
        )
        hybrid = _hybrid_metrics(report)
        if baseline_metrics is None:
            baseline_metrics = dict(hybrid)
            promoted_id = run_id

        solve_non_degrading = hybrid["solve_rate_control_mean"] >= baseline_metrics["solve_rate_control_mean"] - 1e-9
        symbolic_non_degrading = hybrid["symbolic_control_mean"] >= baseline_metrics["symbolic_control_mean"] - 1e-9
        cost_improved = hybrid["wall_per_valid_control_mean"] < baseline_metrics["wall_per_valid_control_mean"] - 1e-9
        promoted = bool(solve_non_degrading and symbolic_non_degrading and cost_improved)
        if promoted:
            promoted_id = run_id
            baseline_metrics = dict(hybrid)

        evaluations.append(
            {
                "run_id": run_id,
                "budget": budget,
                "report_path": str(repeated_output),
                "hybrid_metrics": hybrid,
                "gates": {
                    "solve_non_degrading_vs_promoted_baseline": solve_non_degrading,
                    "symbolic_non_degrading_vs_promoted_baseline": symbolic_non_degrading,
                    "wall_per_valid_improved_vs_promoted_baseline": cost_improved,
                },
                "promoted": promoted,
            }
        )

    promoted_row = next((row for row in evaluations if row["run_id"] == promoted_id), None)
    payload = {
        "phase_id": "agi-bootstrap-focused-efficiency-phase-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract_path),
        "golden_lock": str(lock_path) if lock_path.exists() else None,
        "golden_slices": selected_slices,
        "golden_families": families,
        "seeds": list(args.seeds),
        "profiles": list(args.profiles),
        "max_cycles": int(args.max_cycles),
        "budget_grid": DEFAULT_BUDGET_GRID,
        "evaluations": evaluations,
        "promoted_run_id": promoted_id,
        "promoted_budget": promoted_row["budget"] if promoted_row else {},
        "promoted_report_path": promoted_row["report_path"] if promoted_row else None,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved efficiency phase report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
