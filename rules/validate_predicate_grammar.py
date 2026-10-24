from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_memory.predicate_logic import PredicateFact


def validate_fact(fact: PredicateFact) -> None:
    if not isinstance(fact, PredicateFact):
        raise TypeError(f"Expected PredicateFact, got {type(fact)!r}")
    if len(fact.args) > 2:
        raise ValueError(f"Predicate {fact.predicate} has arity {len(fact.args)} > 2")
    if any(isinstance(arg, (list, tuple, dict)) for arg in fact.args):
        raise ValueError(f"Predicate {fact.predicate} contains nested composite args: {fact.args!r}")


def canonical_bob_example() -> list[PredicateFact]:
    return [
        PredicateFact("has", ("e1",)),
        PredicateFact("owner", ("e1", "bob")),
        PredicateFact("item", ("e1", "apples")),
        PredicateFact("qty", ("e1", 10)),
        PredicateFact("lose", ("e2",)),
        PredicateFact("owner", ("e2", "bob")),
        PredicateFact("item", ("e2", "apples")),
        PredicateFact("qty", ("e2", 3)),
        PredicateFact("query", ("q1",)),
        PredicateFact("owner", ("q1", "bob")),
        PredicateFact("item", ("q1", "apples")),
        PredicateFact("goal", ("q1", "remaining_quantity")),
    ]


def main() -> None:
    facts = canonical_bob_example()
    for fact in facts:
        validate_fact(fact)
    print("Validated canonical predicate grammar on Bob-apples example:")
    for fact in facts:
        print(f"  - {fact}")
    print("All predicate facts are unary/binary only.")


if __name__ == "__main__":
    main()
