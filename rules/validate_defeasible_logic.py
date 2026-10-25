from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forward_engine.defeasible_logic import load_default_defeasible_reasoner
from working_memory.predicate_logic import PredicateFact


def _fact_set(facts):
    return {(f.predicate, tuple(f.args)) for f in facts}


def main() -> None:
    reasoner = load_default_defeasible_reasoner()
    # Scenario: a broad quantity-based conclusion conflicts with a blocked-state
    # exclusion for the same candidate answer.
    reasoner.add_facts(
        [
            PredicateFact("quantity", ("e1",)),
            PredicateFact("owner", ("e1", "bob")),
            PredicateFact("item", ("e1", "apples")),
            PredicateFact("qty", ("e1", 10)),
            PredicateFact("blockedstate", ("b1",)),
            PredicateFact("owner", ("b1", "bob")),
            PredicateFact("item", ("b1", "apples")),
            PredicateFact("qty", ("b1", 10)),
            PredicateFact("query", ("q1",)),
            PredicateFact("owner", ("q1", "bob")),
            PredicateFact("item", ("q1", "apples")),
            PredicateFact("goal", ("q1", "remaining_quantity")),
        ]
    )
    accepted, meta = reasoner.run()
    facts = _fact_set(accepted)
    meta_facts = _fact_set(meta)

    assert ("not_belief", ("answer", 10)) in facts, "Expected blocked-state defeasible rule to win"
    assert ("belief", ("answer", 10)) not in facts, "Fallback quantity answer should be defeated"
    assert ("overrides", ("r_defeasible_not_remaining_from_blocked", "r_defeasible_remaining_from_quantity")) in meta_facts
    print("Defeasible logic validation passed.")


if __name__ == "__main__":
    main()
