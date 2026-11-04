from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ContextCuePolicy:
    cues: set[str]
    allowed_families: set[str]
    allowed_predicates: set[str]
    max_body_len: int
    rationale: list[str]


_FAMILY_NEIGHBORS: dict[str, set[str]] = {
    "remaining_quantity": {"remaining_quantity", "state_transition", "latest_valid_state", "ratio_compose"},
    "state_transition": {"state_transition", "latest_valid_state", "remaining_quantity", "sum_total"},
    "latest_valid_state": {"latest_valid_state", "state_transition", "remaining_quantity"},
    "ratio_compose": {"ratio_compose", "sum_total", "remaining_quantity"},
    "sum_total": {"sum_total", "ratio_compose", "remaining_quantity"},
}

METARULE_LIBRARY: dict[str, list[dict[str, Any]]] = {
    "remaining_quantity": [
        {
            "family": "remaining_quantity",
            "head": ("remaining_quantity", ["?owner", "?item", "?ans"]),
            "body": [("quantity", ["?owner", "?item", "?base"]), ("state_transition", ["?owner", "?item", "?delta", "?after"])],
            "calc": "base + delta",
            "target": "?ans",
            "rationale": "remaining total is current state plus the delta",
        },
        {
            "family": "remaining_quantity",
            "head": ("remaining_quantity", ["?owner", "?item", "?ans"]),
            "body": [("quantity", ["?owner", "?item", "?base"]), ("percent", ["?owner", "?item", "?pct"])],
            "calc": "base * (1 - pct)",
            "target": "?ans",
            "rationale": "percent loss from a remaining base quantity",
        },
    ],
    "state_transition": [
        {
            "family": "state_transition",
            "head": ("state_transition", ["?owner", "?item", "?delta", "?after"]),
            "body": [("quantity", ["?owner", "?item", "?before"]), ("event_delta", ["?owner", "?item", "?delta"])],
            "calc": "before + delta",
            "target": "?after",
            "rationale": "same owner/item updates follow additive state transitions",
        },
        {
            "family": "state_transition",
            "head": ("state_transition", ["?owner", "?item", "?delta", "?after"]),
            "body": [("before_state", ["?owner", "?item", "?before"]), ("after_state", ["?owner", "?item", "?after"])],
            "calc": "after - before",
            "target": "?delta",
            "rationale": "derive delta from ordered state change",
        },
    ],
    "ratio_compose": [
        {
            "family": "ratio_compose",
            "head": ("ratio_compose", ["?owner", "?item", "?ans"]),
            "body": [("ratio", ["?owner", "?item", "?factor"]), ("quantity", ["?owner", "?item", "?qty"])],
            "calc": "qty * factor",
            "target": "?ans",
            "rationale": "multiply a ratio by a current quantity",
        },
    ],
    "sum_total": [
        {
            "family": "sum_total",
            "head": ("sum_total", ["?owner", "?item", "?ans"]),
            "body": [("quantity", ["?owner", "?item", "?lhs"]), ("quantity", ["?owner", "?other", "?rhs"])],
            "calc": "lhs + rhs",
            "target": "?ans",
            "rationale": "sum across related quantity slots",
        },
    ],
    "percent_of_total": [
        {
            "family": "percent_of_total",
            "head": ("percent_of_total", ["?owner", "?item", "?ans"]),
            "body": [("percent", ["?owner", "?item", "?pct"]), ("total", ["?owner", "?item", "?whole"])],
            "calc": "whole * pct",
            "target": "?ans",
            "rationale": "take a percent of a known total",
        },
    ],
    "part_whole_balance": [
        {
            "family": "part_whole_balance",
            "head": ("part_whole_balance", ["?part", "?whole", "?ans"]),
            "body": [("part_quantity", ["?part", "?whole", "?part_qty"]), ("combined_total", ["?part", "?whole", "?whole_qty"])],
            "calc": "whole_qty - part_qty",
            "target": "?ans",
            "rationale": "balance the missing remainder of a whole",
        },
    ],
    "temporal_causal_chain": [
        {
            "family": "temporal_causal_chain",
            "head": ("temporal_causal_chain", ["?owner", "?item", "?ans"]),
            "body": [("before_state", ["?owner", "?item", "?before"]), ("state_transition", ["?owner", "?item", "?delta", "?after"])],
            "calc": "after - before",
            "target": "?ans",
            "rationale": "reason over ordered events and their net causal delta",
        },
    ],
    "profit_cost": [
        {
            "family": "profit_cost",
            "head": ("profit_cost", ["?seller", "?product", "?ans"]),
            "body": [("cost", ["?seller", "?product", "?cost"]), ("percent", ["?seller", "?product", "?pct"])],
            "calc": "cost * (1 + pct)",
            "target": "?ans",
            "rationale": "sell price includes percent profit over base cost",
        },
    ],
    "average_speed": [
        {
            "family": "average_speed",
            "head": ("average_speed", ["?trip", "?ans"]),
            "body": [("distance", ["?trip", "?dist"]), ("duration", ["?trip", "?time"])],
            "calc": "dist / time",
            "target": "?ans",
            "rationale": "compute average speed from total distance and total time",
        },
    ],
}


def _family_predicates() -> dict[str, set[str]]:
    return {
        "remaining_quantity": {"remaining_quantity", "quantity", "percent", "state", "final_state"},
        "state_transition": {"state_transition", "state", "quantity", "final_state"},
        "latest_valid_state": {"state", "final_state", "quantity", "state_transition"},
        "ratio_compose": {"ratio", "quantity", "combined_total", "remaining_quantity"},
        "sum_total": {"combined_total", "quantity", "state", "remaining_quantity"},
        "rate_product": {"rate", "quantity", "rate_times_duration", "value_per_unit"},
        "unit_convert": {"unitconvert", "unit_convert", "quantity"},
        "percent_of_total": {"percent_of_total", "percent", "quantity", "combined_total"},
        "ratio_of_total": {"ratio_of_total", "quantity", "combined_total"},
        "fraction_of_total": {"fraction_of_total", "quantity", "combined_total"},
        "percent_growth": {"percent_growth", "percent", "quantity"},
        "percent_sequence": {"percent_sequence", "percent", "quantity"},
        "average_speed": {"average_speed", "rate", "quantity"},
        "simple_interest": {"simple_interest", "percent", "quantity"},
        "profit_cost": {"profit_cost", "quantity", "combined_total"},
        "part_whole_balance": {"part_whole_balance", "part_quantity", "combined_total", "quantity"},
        "temporal_causal_chain": {"temporal_causal_chain", "causal_delta", "state_transition", "state"},
        "unit_compatibility": {"unit_compatibility", "unit_rate", "unitconvert", "quantity"},
    }


def extract_context_cues(*, question: str, family: str) -> set[str]:
    tokens = [tok.strip(".,!?;:").lower() for tok in str(question or "").split()]
    cues: set[str] = {str(family or "").strip().lower()}
    token_set = set(tokens)
    if "%" in str(question or "") or "percent" in token_set or "discount" in token_set:
        cues.add("percent")
    if {"after", "then", "before"} & token_set:
        cues.add("temporal")
    if {"buy", "bought", "sold", "sell", "gave", "give", "donate", "spoil", "ate", "eaten"} & token_set:
        cues.add("transition")
    if {"left", "remaining", "remain"} & token_set:
        cues.add("remaining")
    if {"times", "ratio", "twice", "thrice"} & token_set:
        cues.add("ratio")
    if {"total", "altogether", "sum", "combined"} & token_set:
        cues.add("total")
    return cues


def build_context_policy(*, question: str, family: str) -> ContextCuePolicy:
    cues = extract_context_cues(question=question, family=family)
    fam = str(family or "").strip().lower()
    allowed_families = set(_FAMILY_NEIGHBORS.get(fam, set()))
    if not allowed_families:
        allowed_families = {fam} if fam else {"state_transition", "remaining_quantity", "ratio_compose"}
    if "percent" in cues:
        allowed_families |= {"remaining_quantity", "percent_of_total", "percent_growth"}
    if "ratio" in cues:
        allowed_families |= {"ratio_compose", "ratio_of_total"}
    if "total" in cues:
        allowed_families |= {"sum_total", "fraction_of_total"}
    if "transition" in cues or "temporal" in cues:
        allowed_families |= {"state_transition", "latest_valid_state", "temporal_causal_chain"}

    pred_map = _family_predicates()
    allowed_predicates: set[str] = set()
    for family_id in allowed_families:
        allowed_predicates |= pred_map.get(family_id, set())
    if not allowed_predicates:
        allowed_predicates = {"quantity", "state", "state_transition", "remaining_quantity", "ratio", "percent", "final_state"}

    max_body_len = 4 if "transition" in cues or "temporal" in cues else 3
    rationale = [
        f"family={fam or 'unknown'}",
        f"cues={sorted(cues)}",
        f"allowed_families={sorted(allowed_families)}",
        f"max_body_len={max_body_len}",
    ]
    return ContextCuePolicy(
        cues=cues,
        allowed_families=allowed_families,
        allowed_predicates=allowed_predicates,
        max_body_len=max_body_len,
        rationale=rationale,
    )


def broaden_metarule_candidates(*, question: str, family: str) -> list[dict[str, Any]]:
    """Return a broader, contextually relevant family library for rule generation.

    This widens the search beyond a single immediate family to include adjacent story
    semantics such as temporal causality, part-whole balancing, and rate-based totals.
    """
    cues = extract_context_cues(question=question, family=family)
    selected: list[dict[str, Any]] = []
    for fam in sorted(METARULE_LIBRARY):
        is_relevant = False
        fam_key = fam.lower()
        if fam_key == str(family or "").strip().lower():
            is_relevant = True
        elif "percent" in cues and fam_key.startswith("percent"):
            is_relevant = True
        elif "ratio" in cues and fam_key.startswith("ratio"):
            is_relevant = True
        elif "remaining" in cues and fam_key in {"remaining_quantity", "state_transition", "part_whole_balance"}:
            is_relevant = True
        elif "transition" in cues and fam_key in {"state_transition", "temporal_causal_chain"}:
            is_relevant = True
        elif "total" in cues and fam_key in {"sum_total", "part_whole_balance", "percent_of_total", "profit_cost"}:
            is_relevant = True
        elif any(token in cues for token in {"rate", "time", "speed"}) and fam_key in {"average_speed", "profit_cost"}:
            is_relevant = True
        if is_relevant:
            selected.extend(METARULE_LIBRARY.get(fam, []))
    return selected


def candidate_matches_policy(candidate: dict[str, Any], policy: ContextCuePolicy) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    family = str(candidate.get("family") or "").strip().lower()
    if family and family not in policy.allowed_families:
        reasons.append(f"family_not_allowed:{family}")

    body = candidate.get("body") or []
    if len(body) > int(policy.max_body_len):
        reasons.append(f"body_too_long:{len(body)}>{policy.max_body_len}")

    preds: set[str] = set()
    head = candidate.get("head")
    if isinstance(head, (tuple, list)) and len(head) >= 1:
        preds.add(str(head[0]).strip().lower())
    for item in body:
        if isinstance(item, (tuple, list)) and len(item) >= 1:
            preds.add(str(item[0]).strip().lower())
    unknown = sorted(p for p in preds if p and p not in policy.allowed_predicates)
    if unknown:
        reasons.append(f"predicates_not_allowed:{','.join(unknown)}")
    return len(reasons) == 0, reasons
