from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").strip()
    match = _NUM_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _build_name_map(input_props: List[Dict[str, Any]]) -> Dict[str, str]:
    name_map: Dict[str, str] = {}
    for p in input_props:
        if p.get("pred") != "name":
            continue
        args = p.get("args") or []
        if len(args) >= 2:
            name_map[str(args[0])] = str(args[1])
    return name_map


def _resolve_token(token: Any, *, name_map: Dict[str, str]) -> str:
    key = str(token)
    return name_map.get(key, key)


def extract_numeric_bridge(
    *,
    input_props: List[Dict[str, Any]],
    class_label: str,
) -> Dict[str, Any]:
    """Build canonical numeric bridge facts + pre-solve trace signals.

    This is deterministic and intentionally simple: we only map stable primitives
    from LT-exported logical-form facts into a compact numeric fact set.
    """
    name_map = _build_name_map(input_props)
    facts: List[Dict[str, Any]] = []

    query_event = None
    query_target = None
    for p in input_props:
        if p.get("pred") == "query":
            args = p.get("args") or []
            if len(args) >= 2:
                query_event = str(args[0])
                query_target = str(args[1])
                break

    owner = "unknown"
    item = "item"
    verb = ""
    if query_event is not None:
        for p in input_props:
            pred = p.get("pred")
            args = p.get("args") or []
            if len(args) < 2 or str(args[0]) != query_event:
                continue
            if pred == "agent":
                owner = _resolve_token(args[1], name_map=name_map).lower()
            elif pred == "patient":
                item = _resolve_token(args[1], name_map=name_map).lower()
            elif pred == "predicate":
                verb = _resolve_token(args[1], name_map=name_map).lower()

    quantity_rows: List[Tuple[str, float]] = []
    for p in input_props:
        if p.get("pred") != "quantity":
            continue
        args = p.get("args") or []
        if len(args) < 2:
            continue
        qty = _as_float(args[1])
        if qty is None:
            continue
        quantity_rows.append((str(args[0]), qty))
        facts.append(
            {
                "concept": "Quantity",
                "owner": owner,
                "item": item,
                "qty": qty,
                "source_var": str(args[0]),
            }
        )

    ratio_cues = {"half", "double", "twice", "triple", "ratio", "percent"}
    transfer_verbs = {"give", "gave", "send", "sell", "pay", "borrow", "lend", "transfer"}
    loss_verbs = {"lose", "lost", "spent", "spend", "eat", "ate", "use", "used", "leave", "left"}
    rate_verbs = {"earn", "cost", "pay", "sell", "buy"}

    joined_names = " ".join(v.lower() for v in name_map.values())
    has_ratio_cue = any(cue in joined_names for cue in ratio_cues) or verb in ratio_cues
    has_transition_cue = (verb in transfer_verbs) or (verb in loss_verbs) or ("then" in joined_names)
    has_rate_cue = (verb in rate_verbs) or ("per" in joined_names) or ("hour" in joined_names)

    if query_event is not None:
        facts.append(
            {
                "concept": "Query",
                "goal": "numeric_answer",
                "event": query_event,
                "target": query_target or "?ans",
                "owner": owner,
                "item": item,
            }
        )

    signals = {
        "has_query_event": query_event is not None,
        "has_verb": bool(verb),
        "has_owner": owner != "unknown",
        "has_item": item != "item",
        "has_quantity": len(quantity_rows) > 0,
        "has_two_quantities": len(quantity_rows) >= 2,
        "has_ratio_cue": has_ratio_cue,
        "has_transition_cue": has_transition_cue,
        "has_rate_cue": has_rate_cue,
    }

    missing: List[str] = []
    if not signals["has_query_event"]:
        missing.append("query_event")
    if not signals["has_quantity"]:
        missing.append("quantity")
    if class_label in {"transfer", "loss"} and not signals["has_transition_cue"]:
        missing.append("transition_cue")
    if class_label == "comparison" and not (signals["has_ratio_cue"] or signals["has_two_quantities"]):
        missing.append("comparison_cue")
    if class_label == "creation" and not (signals["has_rate_cue"] or signals["has_quantity"]):
        missing.append("creation_numeric_cue")

    return {
        "bridge_facts": facts,
        "signals": signals,
        "missing_primitives": missing,
        "resolved": {
            "owner": owner,
            "item": item,
            "verb": verb,
            "query_event": query_event,
            "query_target": query_target,
        },
    }
