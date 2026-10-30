#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_memory.pot_dsl import PoTProgram, PoTStep, execute_program, wm_nodes_from_trace


def _gsm_style_program() -> PoTProgram:
    # A has 3x B; B=10; A gives away 15; then 20% eaten; then buys 8; then 5 spoiled.
    return PoTProgram(
        program_id="gsm-finality-smoke",
        steps=(
            PoTStep("const", "b_qty", (10,)),
            PoTStep("const", "factor", (3,)),
            PoTStep("mul", "a_start", ("$b_qty", "$factor")),
            PoTStep("state_set", "a_state0", ("a", "apples", "$a_start")),
            PoTStep("state_add_delta", "a_after_giveaway", ("a", "apples", -15)),
            PoTStep("state_apply_percent_reduction", "a_after_eat", ("a", "apples", 0.2)),
            PoTStep("state_add_delta", "a_after_buy", ("a", "apples", 8)),
            PoTStep("state_add_delta", "a_after_spoil", ("a", "apples", -5)),
            PoTStep("state_query", "final_a", ("a", "apples")),
        ),
    )


def main() -> None:
    program = _gsm_style_program()
    result = execute_program(program)
    assert result.ok, f"expected program success, got issues={result.issues}"
    final = result.vars.get("final_a")
    assert final is not None
    # 30 - 15 = 15; 20% eaten => 12; +8 => 20; -5 => 15
    assert abs(final - 15.0) < 1e-9, f"expected final 15.0, got {final}"

    wm_nodes = wm_nodes_from_trace(result.trace)
    # We should emit at least one arithmetic WMNode from the executable trace.
    assert len(wm_nodes) >= 1, "expected arithmetic WMNode trace emission"

    print("PoT DSL validation passed.")


if __name__ == "__main__":
    main()
