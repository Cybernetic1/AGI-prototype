from __future__ import annotations

from typing import Any, Mapping, Tuple


_SEMANTIC_GATE_CONCEPTS = {"Quantity", "Has", "StateTransition", "Ratio", "Percent", "Remaining"}


def semantic_candidate_gate(concept: str, attrs: Mapping[str, Any]) -> Tuple[bool, Any]:
    """Return whether a semantic candidate passes System-2 structural checks.

    The returned value is the numeric payload propagated into validated beliefs when
    the gate passes.
    """
    if concept not in _SEMANTIC_GATE_CONCEPTS:
        return False, None

    owner = attrs.get("owner")
    item = attrs.get("item")
    qty = attrs.get("qty")
    delta = attrs.get("delta")
    factor = attrs.get("factor")
    pct = attrs.get("pct")

    if not owner or not item:
        return False, None
    if concept in {"Quantity", "Has", "Remaining"} and qty is None:
        return False, None
    if concept == "StateTransition" and delta is None:
        return False, None
    if concept == "Ratio" and factor is None:
        return False, None
    if concept == "Percent" and pct is None:
        return False, None

    value = qty if qty is not None else (delta if delta is not None else (factor if factor is not None else pct))
    return True, value
