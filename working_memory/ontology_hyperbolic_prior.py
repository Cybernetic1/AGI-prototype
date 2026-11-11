from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from logic_transformer.lorentz_math import project_to_hyperboloid

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIFIED_ONTOLOGY_PATH = REPO_ROOT / "rules" / "reports" / "unified_ontology_v1.json"
DEFAULT_ONTOLOGY_PRIOR_PATH = REPO_ROOT / "rules" / "reports" / "ontology_hyperbolic_prior_v1.json"

_LABEL_TO_ANCHOR = {
    "transfer": 0,
    "creation": 1,
    "loss": 2,
    "comparison": 3,
    "neutral": 4,
}


def _hash_unit_vector(token: str, dim: int) -> torch.Tensor:
    digest = hashlib.sha1(str(token).encode("utf-8")).digest()
    values = []
    while len(values) < dim:
        for b in digest:
            values.append((float(b) / 255.0) - 0.5)
            if len(values) >= dim:
                break
        digest = hashlib.sha1(digest).digest()
    vec = torch.tensor(values[:dim], dtype=torch.float32)
    norm = torch.linalg.norm(vec).item()
    if norm <= 1e-9:
        vec = torch.ones(dim, dtype=torch.float32) / float(dim) ** 0.5
    else:
        vec = vec / norm
    return vec


def _anchor_vector(label: str, dim: int) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    idx = _LABEL_TO_ANCHOR.get(str(label or "neutral").lower(), _LABEL_TO_ANCHOR["neutral"]) % max(1, dim)
    vec[idx] = 1.0
    return vec


def build_hyperbolic_prior_from_unified(unified_payload: Dict[str, Any], *, dim: int = 16) -> Dict[str, Any]:
    lemma_index = unified_payload.get("lemma_index") if isinstance(unified_payload, dict) else {}
    if not isinstance(lemma_index, dict):
        lemma_index = {}
    rows: Dict[str, Dict[str, Any]] = {}
    for lemma, data in lemma_index.items():
        if not isinstance(data, dict):
            continue
        label = str(data.get("event_label") or "neutral").strip().lower()
        confidence = float(data.get("event_confidence") or 0.0)
        # Blend deterministic ontology anchor with lemma-specific lexical identity.
        base = 0.8 * _anchor_vector(label, dim) + 0.2 * _hash_unit_vector(str(lemma), dim)
        base = base * (0.6 + 0.4 * max(0.0, min(1.0, confidence)))
        hyper = project_to_hyperboloid(base.unsqueeze(0)).squeeze(0)
        rows[str(lemma)] = {
            "label": label,
            "confidence": confidence,
            "vector": [float(x) for x in hyper.detach().cpu().tolist()],
            "sources": list(data.get("sources") or []),
        }

    return {
        "version": "v1",
        "kind": "ontology_hyperbolic_prior",
        "dim": int(dim),
        "lemma_count": len(rows),
        "lemmas": rows,
    }


def save_ontology_prior(payload: Dict[str, Any], path: Path = DEFAULT_ONTOLOGY_PRIOR_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_ontology_prior(path: Path = DEFAULT_ONTOLOGY_PRIOR_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def lemma_prior_score(prior_payload: Dict[str, Any], lemma: str) -> Tuple[float, str, float]:
    if not isinstance(prior_payload, dict):
        return 0.0, "neutral", 0.0
    lemmas = prior_payload.get("lemmas")
    if not isinstance(lemmas, dict):
        return 0.0, "neutral", 0.0
    row = lemmas.get(str(lemma).strip().lower())
    if not isinstance(row, dict):
        return 0.0, "neutral", 0.0
    confidence = float(row.get("confidence") or 0.0)
    label = str(row.get("label") or "neutral")
    # Keep score in [0,1], with a mild floor for known ontology lemmas.
    score = min(1.0, max(0.0, 0.5 + 0.5 * confidence))
    return score, label, confidence
