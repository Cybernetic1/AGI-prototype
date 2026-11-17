from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIFIED_ONTOLOGY_PATH = REPO_ROOT / "rules" / "reports" / "unified_ontology_v1.json"
DEFAULT_SEMANTIC_AR_PATH = REPO_ROOT / "rules" / "reports" / "semantic_ar_worked_examples_v1.json"

_SYMBOL_ALIASES = {
    "has": "have",
    "remaining": "remain",
    "state_transition": "change",
    "event": "event",
    "intent": "intend",
    "belief": "believe",
    "revisedbelief": "revise",
    "transfer": "transfer",
    "finalstate": "state",
    "quantity": "quantity",
    "ratio": "ratio",
    "total": "total",
    "percent": "percent",
    "compare": "compare",
    "gain": "gain",
    "lose": "lose",
    "buy": "buy",
    "give": "give",
    "apple": "apple",
}


def normalize_seed_symbol(symbol: str) -> str:
    text = str(symbol or "").strip()
    if not text:
        return ""
    # Split CamelCase words before normalizing punctuation so names like
    # "StateTransition" map to the same canonical lemma as the ontology entry.
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


def canonical_seed_alias(symbol: str) -> str:
    norm = normalize_seed_symbol(symbol)
    return _SYMBOL_ALIASES.get(norm, norm)


def load_unified_ontology(path: str | Path = DEFAULT_UNIFIED_ONTOLOGY_PATH) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _scan_candidates(payload: dict[str, Any], lemma: str) -> list[dict[str, Any]]:
    lemma_index = payload.get("lemma_index", {}) if isinstance(payload, dict) else {}
    if not isinstance(lemma_index, dict):
        return []
    row = lemma_index.get(lemma)
    if not isinstance(row, dict):
        return []
    candidates = [{
        "lemma": lemma,
        "event_label": row.get("event_label", "neutral"),
        "event_confidence": float(row.get("event_confidence", 0.0) or 0.0),
        "sources": list(row.get("sources", []) or []),
    }]
    return candidates


def score_symbol_alignment(symbol: str, ontology_payload: dict[str, Any]) -> dict[str, Any]:
    norm = normalize_seed_symbol(symbol)
    alias = canonical_seed_alias(symbol)
    candidates = []
    for key in (norm, alias):
        if key:
            direct = _scan_candidates(ontology_payload, key)
            if direct:
                candidates.extend(direct)
    if candidates:
        chosen = candidates[0]
        return {
            "symbol": symbol,
            "normalized": norm,
            "alias": alias,
            "chosen_concept": chosen["event_label"],
            "confidence": min(1.0, max(0.0, float(chosen["event_confidence"]) + 0.1)),
            "sources": chosen["sources"],
            "review_required": False,
            "alternatives": candidates[1:],
        }

    # If the symbol is not directly in the ontology, keep the canonical alias as a soft anchor.
    # This preserves the semantic family while flagging its final mapping for review.
    return {
        "symbol": symbol,
        "normalized": norm,
        "alias": alias,
        "chosen_concept": "neutral",
        "confidence": 0.4,
        "sources": [],
        "review_required": True,
        "alternatives": [],
    }


def _iter_predicates(rule_text: str) -> list[str]:
    matches = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", rule_text)
    return [m for m in matches if m]


def align_seed_rule(rule_text: str, ontology_payload: dict[str, Any]) -> dict[str, Any]:
    predicates = _iter_predicates(rule_text)
    aligned_predicates = [score_symbol_alignment(pred, ontology_payload) for pred in predicates]
    review_required = any(item["review_required"] for item in aligned_predicates)
    confidence = min(1.0, sum(item["confidence"] for item in aligned_predicates) / max(1, len(aligned_predicates)))
    return {
        "rule": rule_text,
        "predicates": predicates,
        "alignments": aligned_predicates,
        "confidence": confidence,
        "needs_human_review": review_required,
    }


def align_semantic_ar_examples(payload: dict[str, Any], ontology_payload: dict[str, Any]) -> dict[str, Any]:
    examples = payload.get("examples", []) if isinstance(payload, dict) else []
    aligned_examples: list[dict[str, Any]] = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        aligned = {
            "example_id": ex.get("example_id"),
            "category": ex.get("category"),
            "ontology_alignment": {
                "trace": [],
                "target": score_symbol_alignment(str(ex.get("target") or "").split("(", 1)[0], ontology_payload),
                "seed_rules": [],
            },
        }
        for step in ex.get("trace") or []:
            if not isinstance(step, dict):
                continue
            prop = str(step.get("proposition") or "")
            pred = prop.split("(", 1)[0]
            aligned["ontology_alignment"]["trace"].append({
                "step": step.get("step"),
                "predicate": pred,
                "alignment": score_symbol_alignment(pred, ontology_payload),
            })
        for rule in ex.get("seed_rules") or []:
            aligned["ontology_alignment"]["seed_rules"].append(align_seed_rule(str(rule), ontology_payload))
        aligned_examples.append(aligned)

    summary = {
        "total_examples": len(aligned_examples),
        "review_required_examples": sum(1 for ex in aligned_examples if any(
            item["needs_human_review"] for item in ex["ontology_alignment"]["seed_rules"]
        )),
    }
    result = dict(payload)
    result["ontology_alignment_summary"] = summary
    result["examples"] = aligned_examples
    return result


def align_semantic_ar_file(input_path: str | Path, ontology_path: str | Path = DEFAULT_UNIFIED_ONTOLOGY_PATH, output_path: str | Path | None = None) -> dict[str, Any]:
    in_path = Path(input_path)
    ontology_payload = load_unified_ontology(ontology_path)
    payload = json.loads(in_path.read_text(encoding="utf-8")) if in_path.exists() else {}
    aligned = align_semantic_ar_examples(payload, ontology_payload)
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(aligned, indent=2, ensure_ascii=False), encoding="utf-8")
    return aligned
