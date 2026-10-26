from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import json
from pathlib import Path

from ilp_learner.metagol_nds import MetaRuleLearner, generate_synthetic_family_examples


@dataclass
class RankedMetaRule:
    score: float
    coverage: float
    simplicity: float
    symbolic_validity: float
    candidate: Dict[str, Any]


def generate_synthetic_family_examples_with_negatives() -> Dict[str, Dict[str, List[Tuple[str, List[Any], float]]]]:
    positives = generate_synthetic_family_examples()
    negatives: Dict[str, List[Tuple[str, List[Any], float]]] = {
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
    merged: Dict[str, Dict[str, List[Tuple[str, List[Any], float]]]] = {}
    for family, rows in positives.items():
        merged[family] = {"positive": rows, "negative": negatives.get(family, [])}
    return merged


def infer_family_from_examples(examples: List[Tuple[str, List[Any], float]]) -> str:
    names = [str(x[0]).lower() for x in examples if isinstance(x[0], str)]
    text = " ".join(names)
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

    pos_total = max(1, len(positives))
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

        neg_overlap = len(preds & neg_pred_names) / float(neg_total)
        score = (0.6 * coverage) + (0.25 * simplicity) + (0.25 * symbolic_validity) - (0.2 * neg_overlap)
        ranked.append(
            RankedMetaRule(
                score=score,
                coverage=coverage,
                simplicity=simplicity,
                symbolic_validity=symbolic_validity,
                candidate=c,
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


def run_metarule_learning_cycle(top_n: int = 2) -> List[Dict[str, Any]]:
    learner = MetaRuleLearner()
    dataset = generate_synthetic_family_examples_with_negatives()
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
            }
            accepted.append(item)
    return accepted


def propose_ranked_for_examples(examples: List[Tuple[str, List[Any], float]], top_n: int = 3) -> List[Dict[str, Any]]:
    if not examples:
        return []
    family = infer_family_from_examples(examples)
    data = generate_synthetic_family_examples_with_negatives()
    negatives = data.get(family, {}).get("negative", [])
    learner = MetaRuleLearner()
    proposed = learner.propose(examples)
    ranked = rank_candidates_for_family(family, proposed, examples, negatives)
    selected: List[Dict[str, Any]] = []
    for row in ranked[: max(1, top_n)]:
        item = dict(row.candidate)
        item["ranking"] = {
            "score": row.score,
            "coverage": row.coverage,
            "simplicity": row.simplicity,
            "symbolic_validity": row.symbolic_validity,
            "family_inferred": family,
        }
        selected.append(item)
    return selected


def export_synthetic_training_set(path: str | Path) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_family_examples_with_negatives()
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
