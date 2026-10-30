from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import re
from pathlib import Path
from typing import Any, Mapping

from working_memory.semantic_schema import SemanticGraphSchema


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9_]+", text.lower()) if tok}


@dataclass(frozen=True)
class SemanticARConfig:
    enabled: bool = False
    weight: float = 0.2
    lexical_weight: float = 0.5
    role_weight: float = 0.3
    prior_weight: float = 0.2
    gate_fail_penalty: float = 0.0
    couple_to_symbolic_gate: bool = True
    emit_metadata: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | "SemanticARConfig" | None) -> "SemanticARConfig":
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, Mapping):
            return cls()
        return cls(
            enabled=bool(payload.get("enabled", False)),
            weight=_clamp(payload.get("weight", 0.2)),
            lexical_weight=max(0.0, float(payload.get("lexical_weight", 0.5))),
            role_weight=max(0.0, float(payload.get("role_weight", 0.3))),
            prior_weight=max(0.0, float(payload.get("prior_weight", 0.2))),
            gate_fail_penalty=_clamp(payload.get("gate_fail_penalty", 0.0)),
            couple_to_symbolic_gate=bool(payload.get("couple_to_symbolic_gate", True)),
            emit_metadata=bool(payload.get("emit_metadata", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_semantic_ar_config(path: str | Path) -> SemanticARConfig:
    src = Path(path)
    payload = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"SAR config must be a JSON object: {src}")
    if isinstance(payload.get("config"), Mapping):
        payload = payload["config"]
    return SemanticARConfig.from_mapping(payload)


@dataclass(frozen=True)
class SemanticARScore:
    base_score: float
    final_score: float
    sar_bonus: float
    raw_sar_score: float
    gated_sar_score: float
    symbolic_gate_pass: bool | None
    symbolic_gate_reason: str
    lexical_alignment: float
    role_coverage: float
    concept_prior: float

    def as_metadata(self) -> dict[str, Any]:
        return {
            "sar_base_score": self.base_score,
            "sar_final_score": self.final_score,
            "sar_bonus": self.sar_bonus,
            "sar_raw_score": self.raw_sar_score,
            "sar_gated_score": self.gated_sar_score,
            "sar_gate_pass": self.symbolic_gate_pass,
            "sar_gate_reason": self.symbolic_gate_reason,
            "sar_lexical_alignment": self.lexical_alignment,
            "sar_role_coverage": self.role_coverage,
            "sar_concept_prior": self.concept_prior,
        }


def _expected_role_count(concept: str) -> int:
    expected = {
        "quantity": 3,
        "agent": 2,
        "patient": 2,
        "recipient": 2,
        "name": 1,
        "event": 2,
        "predicate": 2,
    }
    return expected.get(concept.lower(), 2)


def _role_coverage_score(concept: str, args: list[Any]) -> float:
    if not args:
        return 0.0
    filled = 0
    for arg in args:
        if arg is None:
            continue
        text = str(arg).strip()
        if text and text.lower() not in {"none", "null", "nan", "?"}:
            filled += 1
    expected = max(1, _expected_role_count(concept))
    return _clamp(filled / float(expected))


def _lexical_alignment_score(question: str, concept: str, args: list[Any]) -> float:
    q_tokens = _tokenize(question or "")
    proposal_tokens = _tokenize(" ".join([concept] + [str(a) for a in args]))
    if not q_tokens or not proposal_tokens:
        return 0.0
    overlap = q_tokens.intersection(proposal_tokens)
    return _clamp(len(overlap) / float(len(proposal_tokens)))


def _concept_prior_score(question: str, concept: str) -> float:
    q = question.lower()
    c = concept.lower()
    if c == "quantity":
        if re.search(r"\d", q) and any(k in q for k in ("how many", "remaining", "left", "total", "have")):
            return 1.0
        if re.search(r"\d", q):
            return 0.8
        return 0.4
    if c in {"agent", "patient", "recipient", "name"}:
        return 0.7 if re.search(r"\b[a-z]+\b", q) else 0.3
    if c in {"event", "predicate"}:
        if any(k in q for k in ("bought", "gave", "donated", "spoiled", "ate", "used")):
            return 0.8
        return 0.5
    return 0.5


def _semantic_gate_check(concept: str, args: list[Any]) -> tuple[bool | None, str]:
    concept_lower = concept.lower()
    if concept_lower == "quantity":
        text_args = [str(a).strip() for a in args if str(a).strip()]
        owner = text_args[0] if text_args else "entity"
        value = next((_as_float(a) for a in args if _as_float(a) is not None), None)
        payload = {
            "kind": "quantity",
            "id": f"sar_q_{owner}",
            "entity_id": f"ent_{owner}",
            "value": value,
            "unit": "count",
        }
        ok, issues = SemanticGraphSchema.validate_graph_fragment(payload)
        if not ok:
            return False, f"schema:{'; '.join(issues)}"
        if value is None:
            return False, "non_numeric_quantity"
        return True, "schema_ok"

    if concept_lower in {"agent", "patient", "recipient", "name", "entity"}:
        text_args = [str(a).strip() for a in args if str(a).strip()]
        if not text_args:
            return False, "missing_entity_name"
        payload = {
            "kind": "entity",
            "id": f"ent_{text_args[0]}",
            "name": text_args[0],
            "kind": concept_lower,
        }
        ok, issues = SemanticGraphSchema.validate_graph_fragment(payload)
        if not ok:
            return False, f"schema:{'; '.join(issues)}"
        return True, "schema_ok"

    return None, "not_applicable"


def score_semantic_ar(
    *,
    question: str,
    concept: str,
    args: list[Any],
    base_score: float,
    config: Mapping[str, Any] | SemanticARConfig | None = None,
) -> SemanticARScore:
    cfg = SemanticARConfig.from_mapping(config)
    base = _clamp(base_score)

    lexical = _lexical_alignment_score(question, concept, args)
    role_cov = _role_coverage_score(concept, args)
    prior = _concept_prior_score(question, concept)

    weight_sum = max(1e-9, cfg.lexical_weight + cfg.role_weight + cfg.prior_weight)
    raw_sar = _clamp(
        (cfg.lexical_weight * lexical + cfg.role_weight * role_cov + cfg.prior_weight * prior) / weight_sum
    )
    gate_pass, gate_reason = _semantic_gate_check(concept, args)

    if cfg.couple_to_symbolic_gate and gate_pass is not True:
        gated_sar = 0.0
    else:
        gated_sar = raw_sar

    sar_bonus = 0.0
    if cfg.enabled:
        sar_bonus += cfg.weight * gated_sar
        if cfg.couple_to_symbolic_gate and gate_pass is False:
            sar_bonus -= cfg.gate_fail_penalty

    final = _clamp(base + sar_bonus)
    if cfg.couple_to_symbolic_gate and gate_pass is False:
        final = min(final, base)

    return SemanticARScore(
        base_score=base,
        final_score=final,
        sar_bonus=sar_bonus,
        raw_sar_score=raw_sar,
        gated_sar_score=gated_sar,
        symbolic_gate_pass=gate_pass,
        symbolic_gate_reason=gate_reason,
        lexical_alignment=lexical,
        role_coverage=role_cov,
        concept_prior=prior,
    )
