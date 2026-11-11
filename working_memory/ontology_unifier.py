from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIFIED_ONTOLOGY_PATH = REPO_ROOT / "rules" / "reports" / "unified_ontology_v1.json"

EVENT_LABELS = {"transfer", "creation", "loss", "comparison", "neutral"}

_WORDNET_LEXNAME_LABELS = {
    "verb.possession": "transfer",
    "verb.change": "creation",
    "verb.consumption": "loss",
    "verb.contact": "loss",
    "verb.competition": "comparison",
    "verb.stative": "comparison",
}

_VERBNET_CLASS_HINTS = {
    "give": "transfer",
    "send": "transfer",
    "transfer": "transfer",
    "obtain": "creation",
    "get": "creation",
    "create": "creation",
    "build": "creation",
    "destroy": "loss",
    "remove": "loss",
    "consume": "loss",
    "lose": "loss",
    "cost": "comparison",
    "price": "comparison",
    "estimate": "comparison",
}

_FRAMENET_HINTS = {
    "commerce_buy": "creation",
    "commerce_sell": "transfer",
    "giving": "transfer",
    "transfer": "transfer",
    "acquiring": "creation",
    "getting": "creation",
    "using": "loss",
    "destroying": "loss",
    "change_position_on_a_scale": "comparison",
    "comparing": "comparison",
}


def _normalize_lemma(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _event_label_from_wordnet_lexname(lexname: str) -> str:
    return _WORDNET_LEXNAME_LABELS.get(str(lexname or "").strip().lower(), "neutral")


def _event_label_from_hint(text: str, mapping: Dict[str, str]) -> str:
    lowered = str(text or "").strip().lower()
    for hint, label in mapping.items():
        if hint in lowered:
            return label
    return "neutral"


def import_wordnet() -> list[dict[str, Any]]:
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    try:
        synsets = wn.all_synsets(pos=wn.VERB)
    except Exception:
        return []
    for syn in synsets:
        label = _event_label_from_wordnet_lexname(getattr(syn, "lexname", lambda: "")())
        syn_name = str(getattr(syn, "name", lambda: "")())
        for lemma_obj in syn.lemmas():
            lemma = _normalize_lemma(lemma_obj.name())
            if not lemma:
                continue
            rows.append(
                {
                    "source": "wordnet",
                    "lemma": lemma,
                    "sense_id": syn_name,
                    "event_label": label,
                }
            )
    return rows


def import_verbnet() -> list[dict[str, Any]]:
    try:
        from nltk.corpus import verbnet as vn
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    try:
        class_ids = vn.classids()
    except Exception:
        return []
    for class_id in class_ids:
        label = _event_label_from_hint(str(class_id), _VERBNET_CLASS_HINTS)
        try:
            lemmas = vn.lemmas(class_id)
        except Exception:
            lemmas = []
        for lemma_raw in lemmas:
            lemma = _normalize_lemma(lemma_raw)
            if not lemma:
                continue
            rows.append(
                {
                    "source": "verbnet",
                    "lemma": lemma,
                    "sense_id": str(class_id),
                    "event_label": label,
                }
            )
    return rows


def import_framenet(*, verb_focused: bool = True) -> list[dict[str, Any]]:
    try:
        from nltk.corpus import framenet as fn
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    try:
        frames = fn.frames()
    except Exception:
        return []
    for frame in frames:
        frame_name = str(getattr(frame, "name", "") or "")
        label = _event_label_from_hint(frame_name, _FRAMENET_HINTS)
        lex_units = getattr(frame, "lexUnit", {}) or {}
        for lu_name, _ in lex_units.items():
            lu = str(lu_name or "")
            if verb_focused and not lu.endswith(".v"):
                continue
            lemma = _normalize_lemma(lu.rsplit(".", 1)[0])
            if not lemma:
                continue
            rows.append(
                {
                    "source": "framenet",
                    "lemma": lemma,
                    "sense_id": frame_name,
                    "event_label": label,
                }
            )
    return rows


def _build_lemma_index(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in records:
        lemma = _normalize_lemma(str(row.get("lemma") or ""))
        if not lemma:
            continue
        source = str(row.get("source") or "unknown")
        sense_id = str(row.get("sense_id") or "")
        label = str(row.get("event_label") or "neutral")
        if label not in EVENT_LABELS:
            label = "neutral"

        slot = index.setdefault(
            lemma,
            {
                "lemma": lemma,
                "sources": [],
                "wordnet_senses": [],
                "verbnet_classes": [],
                "framenet_frames": [],
                "event_votes": {k: 0 for k in sorted(EVENT_LABELS)},
            },
        )
        if source not in slot["sources"]:
            slot["sources"].append(source)
        if source == "wordnet" and sense_id and sense_id not in slot["wordnet_senses"]:
            slot["wordnet_senses"].append(sense_id)
        elif source == "verbnet" and sense_id and sense_id not in slot["verbnet_classes"]:
            slot["verbnet_classes"].append(sense_id)
        elif source == "framenet" and sense_id and sense_id not in slot["framenet_frames"]:
            slot["framenet_frames"].append(sense_id)
        slot["event_votes"][label] = int(slot["event_votes"].get(label, 0)) + 1

    for slot in index.values():
        votes = slot["event_votes"]
        total = sum(int(v) for v in votes.values())
        if total <= 0:
            slot["event_label"] = "neutral"
            slot["event_confidence"] = 0.0
            continue
        best_label, best_votes = max(votes.items(), key=lambda item: (int(item[1]), item[0]))
        slot["event_label"] = best_label
        slot["event_confidence"] = float(best_votes) / float(total)
    return index


def _align_edges(index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for lemma, row in index.items():
        sources = set(row.get("sources", []))
        if {"wordnet", "verbnet"} <= sources:
            edges.append(
                {
                    "from": f"lemma:{lemma}:wordnet",
                    "to": f"lemma:{lemma}:verbnet",
                    "relation": "same_lemma",
                    "confidence": min(1.0, 0.5 + 0.5 * float(row.get("event_confidence", 0.0))),
                }
            )
        if {"wordnet", "framenet"} <= sources:
            edges.append(
                {
                    "from": f"lemma:{lemma}:wordnet",
                    "to": f"lemma:{lemma}:framenet",
                    "relation": "same_lemma",
                    "confidence": min(1.0, 0.5 + 0.5 * float(row.get("event_confidence", 0.0))),
                }
            )
        if {"verbnet", "framenet"} <= sources:
            edges.append(
                {
                    "from": f"lemma:{lemma}:verbnet",
                    "to": f"lemma:{lemma}:framenet",
                    "relation": "same_lemma",
                    "confidence": min(1.0, 0.5 + 0.5 * float(row.get("event_confidence", 0.0))),
                }
            )
    return edges


def build_unified_ontology(*, include_wordnet: bool = True, include_verbnet: bool = True, include_framenet: bool = True, framenet_verb_focused: bool = True) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if include_wordnet:
        records.extend(import_wordnet())
    if include_verbnet:
        records.extend(import_verbnet())
    if include_framenet:
        records.extend(import_framenet(verb_focused=framenet_verb_focused))

    lemma_index = _build_lemma_index(records)
    edges = _align_edges(lemma_index)
    source_counts = defaultdict(int)
    for row in records:
        source_counts[str(row.get("source") or "unknown")] += 1

    return {
        "version": "v1",
        "record_count": len(records),
        "lemma_count": len(lemma_index),
        "source_counts": dict(sorted(source_counts.items())),
        "lemma_index": dict(sorted(lemma_index.items())),
        "alignment_edges": edges,
        "settings": {
            "include_wordnet": bool(include_wordnet),
            "include_verbnet": bool(include_verbnet),
            "include_framenet": bool(include_framenet),
            "framenet_verb_focused": bool(framenet_verb_focused),
        },
    }


def save_unified_ontology(payload: dict[str, Any], path: Path = DEFAULT_UNIFIED_ONTOLOGY_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_unified_ontology(path: Path = DEFAULT_UNIFIED_ONTOLOGY_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def export_embedding_edges(payload: dict[str, Any]) -> list[tuple[str, str, str, float]]:
    rows: list[tuple[str, str, str, float]] = []
    lemma_index = payload.get("lemma_index", {}) if isinstance(payload, dict) else {}
    if isinstance(lemma_index, dict):
        for lemma, data in lemma_index.items():
            label = str((data or {}).get("event_label") or "neutral")
            confidence = float((data or {}).get("event_confidence") or 0.0)
            rows.append((f"lemma:{lemma}", f"event_label:{label}", "has_event_label", confidence))
            for src in (data or {}).get("sources", []) or []:
                rows.append((f"lemma:{lemma}", f"source:{src}", "in_source", 1.0))
    for edge in payload.get("alignment_edges", []) or []:
        if not isinstance(edge, dict):
            continue
        rows.append(
            (
                str(edge.get("from") or ""),
                str(edge.get("to") or ""),
                str(edge.get("relation") or "aligned"),
                float(edge.get("confidence") or 0.0),
            )
        )
    return rows


def export_ltm_triples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    triples: list[dict[str, Any]] = []
    lemma_index = payload.get("lemma_index", {}) if isinstance(payload, dict) else {}
    if isinstance(lemma_index, dict):
        for lemma, data in lemma_index.items():
            row = data if isinstance(data, dict) else {}
            label = str(row.get("event_label") or "neutral")
            confidence = float(row.get("event_confidence", 0.0) or 0.0)
            triples.append(
                {
                    "head": str(lemma),
                    "relation": "event_label",
                    "tail": label,
                    "confidence": confidence,
                    "source": "unified_ontology",
                }
            )
    for edge in payload.get("alignment_edges", []) or []:
        if not isinstance(edge, dict):
            continue
        from_node = str(edge.get("from") or "")
        to_node = str(edge.get("to") or "")
        relation = str(edge.get("relation") or "aligned")
        confidence = float(edge.get("confidence", 0.0) or 0.0)
        from_parts = from_node.split(":")
        to_parts = to_node.split(":")
        if len(from_parts) >= 2 and len(to_parts) >= 2 and from_parts[0] == "lemma" and to_parts[0] == "lemma":
            triples.append(
                {
                    "head": str(from_parts[1]),
                    "relation": relation,
                    "tail": str(to_parts[1]),
                    "confidence": confidence,
                    "source": "unified_ontology",
                }
            )
    return triples
