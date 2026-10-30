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
CURRICULUM_RUNNER = REPO_ROOT / "rules" / "run_bootstrap_invariant_curriculum.py"
DEFAULT_CONTRACT = REPO_ROOT / "rules" / "bootstrap_invariant_curriculum_widening.json"
DEFAULT_SLICES = ["tinystories-state-tracking-compact", "mixed-narrative-symbolic-compact"]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TinyStories widening gate slices and emit focused report.")
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--slice", nargs="+", default=DEFAULT_SLICES)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-cycles", type=int, default=2)
    parser.add_argument("--outdir", default="rules/reports")
    parser.add_argument("--runner-output", default="rules/reports/agi_bootstrap_tinystories_widening_raw_v1.json")
    parser.add_argument("--output", default="rules/reports/agi_bootstrap_tinystories_widening_gates_v1.json")
    args = parser.parse_args()

    contract_path = Path(args.contract)
    if not contract_path.is_absolute():
        contract_path = REPO_ROOT / contract_path

    runner_output = Path(args.runner_output)
    if not runner_output.is_absolute():
        runner_output = REPO_ROOT / runner_output
    runner_output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.python),
        str(CURRICULUM_RUNNER),
        "--contract",
        str(contract_path),
        "--slice",
        *[str(x) for x in args.slice],
        "--max-cycles",
        str(int(args.max_cycles)),
        "--output",
        str(runner_output),
        "--quiet",
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    raw = _load_json(runner_output)

    slice_rows = []
    for row in raw.get("slices", []):
        if not isinstance(row, dict):
            continue
        gates = row.get("gate_results") or []
        passed = sum(1 for g in gates if isinstance(g, dict) and g.get("passed"))
        blocking_failed = [
            g.get("id")
            for g in gates
            if isinstance(g, dict) and g.get("blocking", True) and not g.get("passed")
        ]
        slice_rows.append(
            {
                "slice_id": row.get("slice_id"),
                "suite": row.get("suite"),
                "case_count": (row.get("baseline_report") or {}).get("case_count"),
                "gates_green": bool(row.get("gates_green")),
                "passed_gate_count": passed,
                "total_gate_count": len(gates),
                "blocking_failures": blocking_failed,
                "report_path": row.get("baseline_report_path"),
            }
        )

    payload = {
        "study_id": "agi-bootstrap-tinystories-widening-gates-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract_path),
        "selected_slices": [str(x) for x in args.slice],
        "max_cycles": int(args.max_cycles),
        "all_gates_green": bool(raw.get("all_gates_green")),
        "halted_on_slice": raw.get("halted_on_slice"),
        "slice_summaries": slice_rows,
        "raw_report_path": str(runner_output),
    }

    out = Path(args.output)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved TinyStories widening gates report to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
