from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import json
from pathlib import Path

from ilp_learner.metagol_nds import (
    MetaRuleLearner,
    evaluate_candidate_constraints,
    generate_synthetic_family_examples,
    resolve_agi_transfer_toggle,
)

# GSM-focused heuristics for "stuck/uncertain" detection.
# We keep them small and explicit so they can be relaxed later for broader AGI settings.
MIN_CONFIDENT_SCORE = 0.80
MIN_TOP2_MARGIN = 0.08


@dataclass
class RankedMetaRule:
    score: float
    coverage: float
    simplicity: float
    symbolic_validity: float
    constraint_validity: float
    constraint_report: Dict[str, Any]
    candidate: Dict[str, Any]


def generate_synthetic_family_examples_with_negatives(
    include_agi_transfer_families: bool | None = None,
) -> Dict[str, Dict[str, List[Tuple[str, List[Any], float]]]]:
    include_transfer = resolve_agi_transfer_toggle(include_agi_transfer_families)
    positives = generate_synthetic_family_examples(
        include_agi_transfer_families=include_transfer
    )
    negatives: Dict[str, List[Tuple[str, List[Any], float]]] = {
        "rate_product": [
            ("combined_total", ["alice", "apples", 20.0], 20.0),
            ("remaining_quantity", ["alice", "apples", 0.75], 15.0),
        ],
        "sum_total": [
            ("rate_times_duration", ["seller", "hours", 5.0], 90.0),
            ("ratio_compose", ["alice", "bob", "apples", 2.0], 20.0),
        ],
        "unit_convert": [
            ("ratio_compose", ["carol", "dave", "apples", 3.0], 30.0),
            ("remaining_quantity", ["bob", "books", 0.8], 24.0),
        ],
        "percent_of_total": [
            ("rate_times_duration", ["worker", "hours", 4.0], 72.0),
            ("combined_total", ["team", "points", 55.0], 55.0),
        ],
        "ratio_of_total": [
            ("combined_total", ["alice", "apples", 20.0], 20.0),
            ("remaining_quantity", ["alice", "apples", 0.75], 15.0),
        ],
        "fraction_of_total": [
            ("ratio_compose", ["alice", "bob", "apples", 2.0], 20.0),
            ("remaining_quantity", ["bob", "books", 0.8], 24.0),
        ],
        "percent_growth": [
            ("combined_total", ["team", "points", 55.0], 55.0),
            ("ratio_compose", ["carol", "dave", "apples", 3.0], 30.0),
        ],
        "percent_sequence": [
            ("percent_growth", ["town", "residents", 600.0], 600.0),
            ("remaining_quantity", ["charlie", "tickets", 0.2], 6.0),
        ],
        "average_speed": [
            ("ratio_compose", ["erin", "frank", "oranges", 0.5], 5.0),
            ("percent_growth", ["school", "students", 240.0], 240.0),
        ],
        "simple_interest": [
            ("combined_total", ["store", "items", 18.0], 18.0),
            ("ratio_compose", ["alice", "bob", "apples", 2.0], 20.0),
        ],
        "profit_cost": [
            ("percent_of_total", ["box", "marble", 10.0], 10.0),
            ("state", ["alice", "apples", 8.0], 8.0),
        ],
        "ratio_compose": [
            ("remaining_quantity", ["alice", "apples", 0.75], 15.0),
            ("state", ["alice", "apples", 8.0], 8.0),
        ],
        "remaining_quantity": [
            ("ratio_compose", ["alice", "bob", "apples", 2.0], 20.0),
            ("final_state", ["alice", "apples", 17.0], 17.0),
        ],
        "latest_valid_state": [
            ("ratio_compose", ["carol", "dave", "apples", 3.0], 30.0),
            ("remaining_quantity", ["bob", "books", 0.8], 24.0),
        ],
        "state_transition": [
            ("remaining_quantity", ["charlie", "tickets", 0.2], 6.0),
            ("ratio_compose", ["erin", "frank", "oranges", 0.5], 5.0),
        ],
    }
    if include_transfer:
        negatives.update(
            {
                "part_whole_balance": [
                    ("unit_compatibility", [3.0, "hour", "minute", 180.0], 180.0),
                    ("ratio_compose", ["alice", "bob", "apples", 2.0], 20.0),
                ],
                "temporal_causal_chain": [
                    ("part_whole_balance", ["part_crate", "whole_crate", 18.0], 18.0),
                    ("unit_compatibility", [2.0, "day", "hour", 48.0], 48.0),
                ],
                "unit_compatibility": [
                    ("part_whole_balance", ["part_bundle", "whole_bundle", 12.0], 12.0),
                    ("temporal_causal_chain", ["before_shift", "after_shift", "cause_loss", 9.0], 9.0),
                ],
            }
        )
    merged: Dict[str, Dict[str, List[Tuple[str, List[Any], float]]]] = {}
    for family, rows in positives.items():
        merged[family] = {"positive": rows, "negative": negatives.get(family, [])}
    return merged


def infer_family_from_examples(
    examples: List[Tuple[str, List[Any], float]],
    include_agi_transfer_families: bool | None = None,
) -> str:
    include_transfer = resolve_agi_transfer_toggle(include_agi_transfer_families)
    names = [str(x[0]).lower() for x in examples if isinstance(x[0], str)]
    for _, args, _ in examples:
        for arg in args or []:
            if isinstance(arg, str):
                names.append(arg.lower())
    text = " ".join(names)
    if include_transfer:
        if "part_whole_balance" in text or ("part" in text and ("whole" in text or "total" in text)):
            return "part_whole_balance"
        if "temporal_causal_chain" in text or (("before" in text or "after" in text) and ("cause" in text or "delta" in text)):
            return "temporal_causal_chain"
        if "unit_compatibility" in text or ("hour" in text and "minute" in text):
            return "unit_compatibility"
    if "ratio_of_total" in text:
        return "ratio_of_total"
    if "fraction_of_total" in text:
        return "fraction_of_total"
    if "percent_growth" in text:
        return "percent_growth"
    if "percent_sequence" in text:
        return "percent_sequence"
    if "average_speed" in text:
        return "average_speed"
    if "simple_interest" in text:
        return "simple_interest"
    if "profit_cost" in text:
        return "profit_cost"
    if "rate" in text or "hour" in text or "duration" in text or "speed" in text:
        return "rate_product"
    if "combined" in text or "total" in text or "sum" in text or "altogether" in text:
        return "sum_total"
    if "unit_convert" in text or "convert" in text or "dozen" in text:
        return "unit_convert"
    if "percent_of_total" in text:
        return "percent_of_total"
    if "ratio" in text or "times" in text:
        return "ratio_compose"
    if "remaining" in text or "percent" in text or "discount" in text:
        return "remaining_quantity"
    if "final_state" in text or "latest" in text or "donat" in text or "spoil" in text:
        return "latest_valid_state"
    return "state_transition"


def _allowed_predicates() -> set[str]:
    return {
        "quantity",
        "remaining_quantity",
        "state",
        "final_state",
        "ratio",
        "percent",
        "state_transition",
        "rate",
        "rate_times_duration",
        "value_per_unit",
        "combined_total",
        "unitconvert",
        "unit_convert",
        "percent_of_total",
        "ratio_of_total",
        "fraction_of_total",
        "percent_growth",
        "percent_sequence",
        "average_speed",
        "simple_interest",
        "profit_cost",
        "part_whole_balance",
        "part_quantity",
        "temporal_causal_chain",
        "causal_delta",
        "unit_compatibility",
        "unit_rate",
    }


def _candidate_family(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("family") or "").strip().lower()


def rank_candidates_for_family(
    family: str,
    candidates: List[Dict[str, Any]],
    positives: List[Tuple[str, List[Any], float]],
    negatives: List[Tuple[str, List[Any], float]],
) -> List[RankedMetaRule]:
    ranked: List[RankedMetaRule] = []
    allowed = _allowed_predicates()

    neg_total = max(1, len(negatives))
    neg_pred_names = {str(x[0]).lower() for x in negatives}

    for c in candidates:
        c_family = _candidate_family(c)
        coverage = 1.0 if c_family == family else 0.0
        if coverage <= 0:
            continue
        body = c.get("body") or []
        body_len = len(body)
        simplicity = 1.0 / max(1, body_len)

        preds = {str(c.get("head", ("", []))[0]).lower()}
        preds.update(str(b[0]).lower() for b in body if isinstance(b, tuple) and len(b) >= 1)
        symbolic_validity = 1.0 if all(p in allowed for p in preds) else 0.0
        if symbolic_validity <= 0:
            continue
        constraint_ok, constraint_report = evaluate_candidate_constraints(c, positives)
        if not constraint_ok:
            continue
        constraint_validity = float(constraint_report.get("score", 0.0))

        neg_overlap = len(preds & neg_pred_names) / float(neg_total)
        score = (
            (0.55 * coverage)
            + (0.2 * simplicity)
            + (0.2 * symbolic_validity)
            + (0.2 * constraint_validity)
            - (0.2 * neg_overlap)
        )
        ranked.append(
            RankedMetaRule(
                score=score,
                coverage=coverage,
                simplicity=simplicity,
                symbolic_validity=symbolic_validity,
                constraint_validity=constraint_validity,
                constraint_report=constraint_report,
                candidate=c,
            )
        )

    # Deterministic ranking (not randomized): higher score first.
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


def _guidance_flags(ranked: List[RankedMetaRule]) -> Dict[str, Any]:
    """Decide whether to request human guidance for this candidate set.

    We call it "uncertain" when either:
    - the best score is weak, or
    - top candidates are too close to distinguish confidently.
    """
    if not ranked:
        return {"needs_human_guidance": True, "reason": "no_candidates", "score": 0.0, "margin": 0.0}
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    margin = float(best.score - second.score) if second is not None else float(best.score)
    low_score = float(best.score) < MIN_CONFIDENT_SCORE
    low_margin = margin < MIN_TOP2_MARGIN
    if low_score:
        return {
            "needs_human_guidance": True,
            "reason": "low_score",
            "score": float(best.score),
            "margin": margin,
        }
    if low_margin:
        return {
            "needs_human_guidance": True,
            "reason": "low_margin",
            "score": float(best.score),
            "margin": margin,
        }
    return {"needs_human_guidance": False, "reason": "confident", "score": float(best.score), "margin": margin}


def run_metarule_learning_cycle(
    top_n: int = 2,
    include_agi_transfer_families: bool | None = None,
) -> List[Dict[str, Any]]:
    include_transfer = resolve_agi_transfer_toggle(include_agi_transfer_families)
    learner = MetaRuleLearner(include_agi_transfer_families=include_transfer)
    dataset = generate_synthetic_family_examples_with_negatives(
        include_agi_transfer_families=include_transfer
    )
    accepted: List[Dict[str, Any]] = []
    for family, group in dataset.items():
        positives = group.get("positive", [])
        negatives = group.get("negative", [])
        proposed = learner.propose(positives)
        ranked = rank_candidates_for_family(family, proposed, positives, negatives)
        for row in ranked[: max(1, top_n)]:
            item = dict(row.candidate)
            item["ranking"] = {
                "score": row.score,
                "coverage": row.coverage,
                "simplicity": row.simplicity,
                "symbolic_validity": row.symbolic_validity,
                "constraint_validity": row.constraint_validity,
                "constraint_checks": row.constraint_report.get("checks", []),
            }
            accepted.append(item)
    return accepted


def propose_ranked_for_examples(
    examples: List[Tuple[str, List[Any], float]],
    top_n: int = 3,
    include_agi_transfer_families: bool | None = None,
) -> List[Dict[str, Any]]:
    if not examples:
        return []
    include_transfer = resolve_agi_transfer_toggle(include_agi_transfer_families)
    # Meta-template-based search:
    # 1) infer family from the examples
    # 2) generate a small seeded candidate pool
    # 3) rank deterministically by symbolic heuristics
    # This is not BFS and does not use random branching.
    family = infer_family_from_examples(
        examples,
        include_agi_transfer_families=include_transfer,
    )
    data = generate_synthetic_family_examples_with_negatives(
        include_agi_transfer_families=include_transfer
    )
    negatives = data.get(family, {}).get("negative", [])
    learner = MetaRuleLearner(include_agi_transfer_families=include_transfer)
    proposed = learner.propose(examples)
    ranked = rank_candidates_for_family(family, proposed, examples, negatives)
    guidance = _guidance_flags(ranked)
    selected: List[Dict[str, Any]] = []
    for row in ranked[: max(1, top_n)]:
        item = dict(row.candidate)
        item["ranking"] = {
            "score": row.score,
            "coverage": row.coverage,
            "simplicity": row.simplicity,
            "symbolic_validity": row.symbolic_validity,
            "constraint_validity": row.constraint_validity,
            "constraint_checks": row.constraint_report.get("checks", []),
            "family_inferred": family,
            "needs_human_guidance": guidance["needs_human_guidance"],
            "guidance_reason": guidance["reason"],
            "top2_margin": guidance["margin"],
        }
        selected.append(item)
    return selected


def export_synthetic_training_set(
    path: str | Path,
    include_agi_transfer_families: bool | None = None,
) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_family_examples_with_negatives(
        include_agi_transfer_families=include_agi_transfer_families
    )
    rows = []
    for family, group in data.items():
        for pred, args, expected in group.get("positive", []):
            rows.append({"label": 1, "family": family, "pred": pred, "args": args, "expected": expected})
        for pred, args, expected in group.get("negative", []):
            rows.append({"label": 0, "family": family, "pred": pred, "args": args, "expected": expected})
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
