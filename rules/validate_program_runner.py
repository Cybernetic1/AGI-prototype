from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rules.program_runner import list_program_specs, load_program_spec, readiness_check, run_program


def main() -> None:
    specs = list_program_specs()
    names = {p.stem for p in specs}
    assert "gsm8k_solver" in names
    assert "story_recall_eval" in names

    spec = load_program_spec(program_id="gsm8k_solver")
    assert spec.name == "gsm8k_solver"

    gate = readiness_check(spec, quiet=True)
    assert gate.get("enabled") is True
    assert "ready" in gate

    report = run_program(spec, quiet=True, quick=True)
    assert report.get("program") == "gsm8k_solver"
    assert report.get("mode") in {"template_generalization", "family_benchmark", "single_problem"}
    run = report.get("report") or report.get("result")
    assert run is not None

    story_spec = load_program_spec(program_id="story_recall_eval")
    story_gate = readiness_check(story_spec, quiet=True)
    assert story_gate.get("enabled") is False
    print("Program runner validation passed.")


if __name__ == "__main__":
    main()
