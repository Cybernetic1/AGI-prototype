#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rules.program_runner import list_program_specs, load_program_spec, readiness_check, run_program


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a named cognitive program profile.")
    parser.add_argument("--program", default="gsm8k_solver", help="Program id under rules/programs/<id>.yaml")
    parser.add_argument("--program-file", default=None, help="Explicit program YAML path")
    parser.add_argument("--quick", action="store_true", help="Use quick subset mode where supported.")
    parser.add_argument("--quiet", action="store_true", help="Suppress benchmark detail logs.")
    parser.add_argument("--background", action="store_true", help="Request background-ready gate check before run.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    parser.add_argument("--check-only", action="store_true", help="Run readiness check only; do not execute the program.")
    parser.add_argument("--list", action="store_true", help="List available program specs and exit.")
    args = parser.parse_args()

    if args.list:
        for path in list_program_specs():
            print(path.stem)
        return 0

    spec = load_program_spec(program_id=args.program, path=args.program_file)
    gate = readiness_check(spec, quiet=args.quiet)
    if args.background:
        if not spec.background_allowed:
            print(f"[blocked] program '{spec.name}' does not allow background runs.")
            return 2
        if not gate.get("ready", False):
            print(
                f"[blocked] readiness gate failed for '{spec.name}': "
                f"{gate.get('solved', 0)}/{gate.get('total', 0)} solved; "
                f"requires >= {gate.get('min_solved', 0)}."
            )
            return 3

    if args.check_only:
        print(json.dumps({"program": spec.name, "readiness": gate}, indent=2, ensure_ascii=False))
        return 0 if gate.get("ready", False) else 4

    report = run_program(spec, quiet=args.quiet, quick=args.quick)
    output = {"program": spec.name, "description": spec.description, "readiness": gate, "run": report}
    if args.json:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        mode = report.get("mode")
        if "report" in report:
            r = report["report"]
            print(f"Program {spec.name} ({mode}): {r.get('solved')} / {r.get('total')} solved")
        elif "result" in report:
            r = report["result"]
            print(f"Program {spec.name} ({mode}): solved={r.get('solved')} beliefs={r.get('belief_values')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
