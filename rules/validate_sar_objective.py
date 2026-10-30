#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "logic_transformer") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "logic_transformer"))

from semantic_ar import SemanticARConfig, score_semantic_ar
from working_memory.symbolic_validation import semantic_candidate_gate


def main() -> None:
    cfg = SemanticARConfig(
        enabled=True,
        weight=0.25,
        lexical_weight=0.5,
        role_weight=0.3,
        prior_weight=0.2,
        gate_fail_penalty=0.05,
        couple_to_symbolic_gate=True,
    )

    valid = score_semantic_ar(
        question="Alice had 10 apples. How many apples does Alice have?",
        concept="Quantity",
        args=["Alice", "apples", 10],
        base_score=0.6,
        config=cfg,
    )
    invalid = score_semantic_ar(
        question="Alice had many apples.",
        concept="Quantity",
        args=["Alice", "apples", "many"],
        base_score=0.6,
        config=cfg,
    )

    assert valid.symbolic_gate_pass is True, "valid quantity should pass symbolic pre-gate"
    assert valid.final_score > valid.base_score, "SAR should boost valid symbolic proposals"
    assert invalid.symbolic_gate_pass is False, "non-numeric quantity should fail symbolic pre-gate"
    assert invalid.final_score <= invalid.base_score, "failed symbolic pre-gate must not receive SAR boost"

    gate_invalid, _ = semantic_candidate_gate("Quantity", {"owner": "Alice", "item": "apples"})
    gate_valid, propagated_value = semantic_candidate_gate(
        "Quantity",
        {"owner": "Alice", "item": "apples", "qty": 10},
    )
    assert gate_invalid is False, "invalid semantic candidate must be rejected by System-2 gate"
    assert gate_valid is True and propagated_value == 10, "valid semantic candidate should be accepted by System-2 gate"

    print("SAR objective smoke validation passed.")


if __name__ == "__main__":
    main()
