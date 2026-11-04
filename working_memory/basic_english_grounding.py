from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set

from working_memory.ltm import LongTermMemory
from working_memory.semantic_schema import SemanticGraphSchema

# Small, intentionally conservative Basic English-style vocabulary.
# The goal is not broad coverage, but a stable core that can serve as the
# bootstrap layer before more specialized GSM semantics are learned.
BASIC_ENGLISH_CORE: Set[str] = {
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "before", "between", "buy", "bought", "by", "can", "cost",
    "do", "each", "eat", "few", "for", "from", "gave", "give", "go", "had", "have",
    "he", "her", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "left", "less", "many", "more", "need", "not", "of", "on", "one", "or", "our",
    "out", "part", "percent", "remain", "remained", "remaining", "remainder", "same",
    "sell", "she", "some", "spend", "take", "that", "the", "their", "them", "then",
    "there", "they", "this", "time", "to", "total", "up", "use", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "whole", "with", "would", "you",
    "your",
}

# A tiny concept lexicon that maps weakly-understood terms to the canonical
# semantic categories already used by the reasoning stack.
TERM_GROUNDING: Dict[str, Dict[str, Any]] = {
    "left": {"concept": "remaining_quantity", "gloss": "the amount that remains after some change", "kind": "state"},
    "remaining": {"concept": "remaining_quantity", "gloss": "what stays after subtracting something; remain", "kind": "state"},
    "remain": {"concept": "remaining_quantity", "gloss": "to remain after a change", "kind": "state"},
    "remained": {"concept": "remaining_quantity", "gloss": "to remain after a change", "kind": "state"},
    "remainder": {"concept": "remaining_quantity", "gloss": "the remainder after some part is taken away", "kind": "state"},
    "whole": {"concept": "whole_quantity", "gloss": "the complete amount before part is removed", "kind": "quantity"},
    "part": {"concept": "part_whole", "gloss": "a section of a larger whole", "kind": "quantity"},
    "percent": {"concept": "percent", "gloss": "a proportion of 100", "kind": "percent"},
    "after": {"concept": "temporal_after", "gloss": "later in a sequence", "kind": "context"},
    "before": {"concept": "temporal_before", "gloss": "earlier in a sequence", "kind": "context"},
    "more": {"concept": "comparison_more", "gloss": "greater in amount", "kind": "operator"},
    "less": {"concept": "comparison_less", "gloss": "smaller in amount", "kind": "operator"},
    "buy": {"concept": "acquire", "gloss": "to get something by paying money", "kind": "state_transition"},
    "sell": {"concept": "transfer", "gloss": "to give something in exchange for money", "kind": "state_transition"},
    "give": {"concept": "transfer", "gloss": "to pass something to another person", "kind": "state_transition"},
    "eat": {"concept": "consume", "gloss": "to use something up by eating it", "kind": "state_transition"},
    "spoil": {"concept": "loss", "gloss": "to become unusable or damaged", "kind": "state_transition"},
    "total": {"concept": "sum_total", "gloss": "the complete combined amount", "kind": "quantity"},
    "cost": {"concept": "price", "gloss": "the money paid for something", "kind": "quantity"},
    "price": {"concept": "price", "gloss": "the amount of money for a unit or item", "kind": "quantity"},
    "each": {"concept": "per_unit", "gloss": "one item or one share at a time", "kind": "rate"},
    "share": {"concept": "partition", "gloss": "a part of something divided among people", "kind": "quantity"},
    "need": {"concept": "requirement", "gloss": "a necessary amount or condition", "kind": "context"},
    "because": {"concept": "cause", "gloss": "for an explanatory reason", "kind": "context"},
    "then": {"concept": "sequence", "gloss": "next in an ordered event chain", "kind": "context"},
}

POS_KIND_DEFAULTS: Dict[str, str] = {
    "verb": "state_transition",
    "noun": "entity",
    "adjective": "operator",
    "adverb": "context",
    "preposition": "context",
    "conjunction": "context",
    "pronoun": "entity",
    "determiner": "context",
    "number": "quantity",
}


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in re.findall(r"[A-Za-z%]+", str(text or ""))]


def _normalize_term(term: str) -> str:
    token = str(term or "").strip().lower()
    aliases = {
        "bought": "buy",
        "had": "have",
        "remained": "remaining",
        "remainder": "remaining",
        "remains": "remaining",
        "remaining": "remaining",
        "left": "remaining",
    }
    return aliases.get(token, token)


def _normalize_pos(pos: str) -> str:
    token = str(pos or "").strip().lower()
    aliases = {
        "v": "verb",
        "n": "noun",
        "adj": "adjective",
        "adv": "adverb",
        "prep": "preposition",
        "conj": "conjunction",
        "pron": "pronoun",
        "det": "determiner",
        "num": "number",
    }
    return aliases.get(token, token)


def _concept_from_pos_and_gloss(term: str, pos: str, gloss: str) -> tuple[str, str, float]:
    token = _normalize_term(term)
    pos_norm = _normalize_pos(pos)
    gloss_norm = str(gloss or "").strip().lower()
    cue = f"{token} {gloss_norm}"

    if token in TERM_GROUNDING:
        existing = TERM_GROUNDING[token]
        return str(existing.get("concept", "unknown_lexeme")), str(existing.get("kind", "entity")), 0.95

    if pos_norm == "verb":
        if any(k in cue for k in {"give", "send", "donate", "transfer", "share out"}):
            return "transfer", "state_transition", 0.9
        if any(k in cue for k in {"buy", "purchase", "acquire", "get", "obtain"}):
            return "acquire", "state_transition", 0.9
        if any(k in cue for k in {"eat", "consume", "spend", "lose", "destroy", "waste"}):
            return "loss", "state_transition", 0.88
        if any(k in cue for k in {"make", "create", "build", "produce"}):
            return "creation", "state_transition", 0.85
        return f"action_{token}", "state_transition", 0.62

    if pos_norm == "noun":
        if any(k in cue for k in {"money", "cost", "price", "pay"}):
            return "price", "quantity", 0.86
        if any(k in cue for k in {"percent", "proportion", "fraction"}):
            return "percent", "percent", 0.86
        if any(k in cue for k in {"total", "sum", "combined"}):
            return "sum_total", "quantity", 0.84
        if any(k in cue for k in {"amount", "number", "quantity", "count"}):
            return "quantity", "quantity", 0.82
        if any(k in cue for k in {"time", "sequence", "order"}):
            return "temporal_marker", "context", 0.75
        return f"entity_{token}", "entity", 0.58

    if pos_norm == "adjective":
        if any(k in cue for k in {"more", "greater", "larger", "bigger"}):
            return "comparison_more", "operator", 0.84
        if any(k in cue for k in {"less", "fewer", "smaller"}):
            return "comparison_less", "operator", 0.84
        return f"modifier_{token}", "operator", 0.56

    if pos_norm == "number":
        return "quantity", "quantity", 0.8

    if pos_norm in {"preposition", "conjunction", "adverb", "determiner"}:
        return f"context_{token}", "context", 0.52

    return "unknown_lexeme", POS_KIND_DEFAULTS.get(pos_norm, "entity"), 0.4


def generate_pos_grounding_candidate(*, word: str, pos: str, gloss: str) -> Dict[str, Any]:
    """Generate a semantic-mapping candidate from a PoS-tagged lexicon entry."""
    term = _normalize_term(word)
    concept, kind, confidence = _concept_from_pos_and_gloss(term=term, pos=pos, gloss=gloss)
    return {
        "term": term,
        "pos": _normalize_pos(pos),
        "concept": concept,
        "kind": kind,
        "gloss": str(gloss or ""),
        "confidence": float(confidence),
        "source": "pos_automation",
    }


def ingest_pos_lexicon(entries: Iterable[Dict[str, Any]], *, register: bool = False, overwrite: bool = False) -> Dict[str, Any]:
    """Build semantic candidates from PoS-tagged lexicon entries.

    Entries are dictionaries with keys like: word, pos, gloss.
    """
    candidates: List[Dict[str, Any]] = []
    registered = 0
    skipped = 0
    for row in entries:
        word = str((row or {}).get("word") or "").strip()
        pos = str((row or {}).get("pos") or "").strip()
        gloss = str((row or {}).get("gloss") or "").strip()
        if not word or not pos:
            skipped += 1
            continue
        candidate = generate_pos_grounding_candidate(word=word, pos=pos, gloss=gloss)
        candidates.append(candidate)
        if register:
            term = str(candidate.get("term") or "").strip().lower()
            if not term:
                continue
            if term in TERM_GROUNDING and not overwrite:
                continue
            TERM_GROUNDING[term] = {
                "concept": candidate.get("concept", "unknown_lexeme"),
                "gloss": candidate.get("gloss", ""),
                "kind": candidate.get("kind", "entity"),
            }
            BASIC_ENGLISH_CORE.add(term)
            registered += 1
    return {
        "count": len(candidates),
        "registered": registered,
        "skipped": skipped,
        "candidates": candidates,
    }


def propose_noun_facts(*, noun: str, gloss: str = "", max_facts: int = 5) -> List[Dict[str, Any]]:
    """Generate candidate noun attributes that can be human-vetted before LTM write."""
    token = _normalize_term(noun)
    gloss_norm = str(gloss or "").strip().lower()
    candidates: List[Dict[str, Any]] = []

    def add(attr: str, value: str, conf: float) -> None:
        if not attr or not value:
            return
        if any(f["attribute"] == attr and f["value"] == value for f in candidates):
            return
        candidates.append(
            {
                "attribute": attr,
                "value": value,
                "confidence": float(conf),
                "source": "noun_pos_bootstrap",
            }
        )

    add("isa", "object", 0.5)
    if "fruit" in gloss_norm or token in {"apple", "orange", "banana", "pear", "grape"}:
        add("isa", "fruit", 0.9)
        add("taste", "sweet", 0.66)
        add("edible", "true", 0.92)
        add("color", "red", 0.6 if token == "apple" else 0.5)
    if "vehicle" in gloss_norm or token in {"car", "bus", "truck"}:
        add("isa", "vehicle", 0.88)
        add("moves_by", "transport", 0.75)
    if any(k in gloss_norm for k in {"person", "human", "student", "boy", "girl", "man", "woman"}):
        add("isa", "person", 0.88)
        add("can_possess", "true", 0.75)
    if any(k in gloss_norm for k in {"money", "dollar", "coin", "cash"}):
        add("isa", "currency", 0.85)
        add("used_for", "payment", 0.72)
    if any(k in gloss_norm for k in {"container", "box", "bag", "basket"}):
        add("can_contain", "items", 0.74)
    if any(k in gloss_norm for k in {"food", "eat", "edible"}):
        add("edible", "true", 0.86)
    if any(k in gloss_norm for k in {"red", "green", "blue", "yellow"}):
        for color in ("red", "green", "blue", "yellow"):
            if color in gloss_norm:
                add("color", color, 0.76)

    if len(candidates) < 5:
        add("lexeme", token, 0.55)
    return candidates[: max(1, int(max_facts))]


def vet_noun_facts(
    *,
    noun: str,
    candidates: List[Dict[str, Any]],
    decision_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Vet candidate noun facts interactively via callback.

    decision_callback receives (noun, candidate_dict) and returns True/False.
    If callback is omitted, defaults to keep high-confidence facts (>= 0.7).
    """
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for cand in candidates:
        keep = bool(float(cand.get("confidence", 0.0)) >= 0.7)
        if decision_callback is not None:
            keep = bool(decision_callback(noun, cand))
        if keep:
            accepted.append(cand)
        else:
            rejected.append(cand)
    return {"noun": _normalize_term(noun), "accepted": accepted, "rejected": rejected}


def curate_noun_knowledge(
    *,
    noun: str,
    gloss: str = "",
    decision_callback: Optional[Any] = None,
    ltm: Optional[LongTermMemory] = None,
    max_facts: int = 5,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Propose, vet, and persist noun facts to LTM."""
    candidates = propose_noun_facts(noun=noun, gloss=gloss, max_facts=max_facts)
    vet = vet_noun_facts(noun=noun, candidates=candidates, decision_callback=decision_callback)
    memory = ltm or LongTermMemory.default()
    store = memory.upsert_noun_knowledge(
        noun=noun,
        facts=vet["accepted"],
        source="basic_english_noun_curation",
        vetted_by="human" if decision_callback is not None else "auto_threshold",
        overwrite=overwrite,
    )
    return {
        "noun": _normalize_term(noun),
        "candidates": candidates,
        "accepted": vet["accepted"],
        "rejected": vet["rejected"],
        "store": store,
    }


def extract_unknown_terms(text: str, *, seed_words: Optional[Iterable[str]] = None) -> List[str]:
    """Return terms that are not covered by the basic vocabulary.

    The result is normalized to a small canonical form so that variants like
    "remained" and "remainder" map onto the same semantic concept.
    """
    vocab = set(BASIC_ENGLISH_CORE)
    if seed_words is not None:
        vocab |= {str(w).lower() for w in seed_words}
    unknown = []
    for tok in _tokenize(text):
        normalized = _normalize_term(tok)
        if normalized not in vocab and normalized not in unknown:
            unknown.append(normalized)
    return unknown


def ground_term(term: str) -> Dict[str, Any]:
    """Map a term to a canonical semantic concept for graph grounding."""
    token = _normalize_term(term)
    if not token:
        return {"term": "", "concept": "unknown", "gloss": "empty token", "kind": "context"}
    if token in TERM_GROUNDING:
        entry = dict(TERM_GROUNDING[token])
        entry["term"] = token
        return entry
    # Fallback: treat as an underspecified entity or proposition. This keeps the
    # system fail-closed and easy to inspect without hallucinating a specific fact.
    return {
        "term": token,
        "concept": "unknown_lexeme",
        "gloss": f"not yet grounded in the Basic-English vocabulary; requires dictionary lookup and semantic composition",
        "kind": "entity",
    }


def bootstrap_grounding(text: str, *, seed_words: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Build a compact semantic grounding summary for a passage.

    The returned structure is intentionally small, explicit, and inspectable so it
    can be used as the first stage of a larger NL->semantic graph pipeline.
    """
    unknown = extract_unknown_terms(text, seed_words=seed_words)
    anchored = []
    seen_terms = set()
    for tok in _tokenize(text):
        normalized = _normalize_term(tok)
        if normalized in TERM_GROUNDING and normalized not in seen_terms:
            anchored.append(ground_term(normalized))
            seen_terms.add(normalized)
    grounded = [ground_term(w) for w in unknown]
    grounded = anchored + grounded
    seen_grounded = set()
    deduped_grounded = []
    for item in grounded:
        key = item.get("term")
        if key and key in seen_grounded:
            continue
        if key:
            seen_grounded.add(key)
        deduped_grounded.append(item)

    graph_nodes: Dict[str, Dict[str, Any]] = {}
    node_id = 0
    for item in deduped_grounded:
        node_id += 1
        graph_nodes[f"g{node_id}"] = {
            "kind": item.get("kind", "entity"),
            "id": f"g{node_id}",
            "name": item.get("term", "unknown"),
            "concept": item.get("concept", "unknown_lexeme"),
            "gloss": item.get("gloss", ""),
        }

    semantic_graph = {
        "nodes": graph_nodes,
        "edges": [],
        "unknown_terms": unknown,
        "grounded_terms": deduped_grounded,
        "schema": SemanticGraphSchema.clamp_predicates(),
    }
    return semantic_graph
