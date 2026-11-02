from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from ilp_learner.metagol_nds import MetaRuleLearner
from ilp_learner.metarule_loop import propose_ranked_for_examples
from rules.contextual_metarules import build_context_policy, candidate_matches_policy
from rules.ilp_rule_utility import utility_components
from rules.teacher_ollama import request_ollama_chat_json


@dataclass
class ILPPipelineConfig:
    candidate_top_n: int = 6
    prescore_min: float = 0.45
    max_candidates_per_family: int = 3
    include_agi_transfer_families: bool = True
    llm_validate_max: int = 2
    llm_latency_budget_sec: float = 12.0
    llm_timeout_sec: int = 20
    llm_enabled: bool = False
    ollama_url: str = ""
    ollama_model: str = ""


@dataclass
class CandidateRecord:
    candidate_id: str
    family: str
    signature: str
    candidate: dict[str, Any]
    prescore: float
    dropped_reason: str = ""
    llm_valid: bool | None = None
    llm_score: float = 0.0
    llm_reason: str = ""
    llm_latency_sec: float = 0.0


def _normalized_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def candidate_signature(candidate: dict[str, Any]) -> str:
    core = {
        "family": candidate.get("family"),
        "head": candidate.get("head"),
        "body": candidate.get("body"),
        "calc": candidate.get("calc"),
        "target": candidate.get("target"),
    }
    return hashlib.sha1(_normalized_json(core).encode("utf-8")).hexdigest()


def _candidate_predicate_set(candidate: dict[str, Any]) -> set[str]:
    preds: set[str] = set()
    head = candidate.get("head")
    if isinstance(head, tuple) and len(head) >= 1:
        preds.add(str(head[0]).strip().lower())
    for body_item in candidate.get("body") or []:
        if isinstance(body_item, tuple) and len(body_item) >= 1:
            preds.add(str(body_item[0]).strip().lower())
    return {p for p in preds if p}


def candidate_to_raw_rule(candidate: dict[str, Any]) -> str:
    family = str(candidate.get("family") or "").strip().lower()
    if family == "latest_valid_state":
        return "Latest valid state wins: prefer the later state for the same owner and item."
    if family == "remaining_quantity":
        return "If remaining quantity is needed, then compute remaining quantity as base times one minus percent."
    if family == "state_transition":
        return "When buy, donate, spoil, or give events change state, update state for same owner and item by applying delta."
    head = candidate.get("head")
    body = candidate.get("body")
    calc = candidate.get("calc")
    return (
        f"Rule family {family}. If {body}, then derive {head} "
        f"using calc={calc}."
    )


def _prescore_candidate(
    candidate: dict[str, Any],
    *,
    seen_signatures: set[str],
    family_counts: dict[str, int],
    max_candidates_per_family: int,
    utility_bonus: dict[str, float] | None = None,
) -> tuple[float, str]:
    ranking = candidate.get("ranking") if isinstance(candidate.get("ranking"), dict) else {}
    base = float(ranking.get("score", 0.0) or 0.0)
    coverage = float(ranking.get("coverage", 0.0) or 0.0)
    simplicity = float(ranking.get("simplicity", 0.0) or 0.0)
    symbolic = float(ranking.get("symbolic_validity", 0.0) or 0.0)
    constraint = float(ranking.get("constraint_validity", 0.0) or 0.0)

    family = str(candidate.get("family") or "").strip().lower()
    signature = candidate_signature(candidate)
    duplicate_penalty = 0.25 if signature in seen_signatures else 0.0
    budget_penalty = 0.20 if family_counts.get(family, 0) >= max_candidates_per_family else 0.0

    preds = _candidate_predicate_set(candidate)
    complexity_penalty = 0.08 if len(preds) >= 4 else 0.0
    utility = utility_bonus or {}
    rl_bonus = float(utility.get("bonus", 0.0) or 0.0)
    rl_value = float(utility.get("value", 0.0) or 0.0)

    score = (
        (0.50 * base)
        + (0.15 * coverage)
        + (0.10 * simplicity)
        + (0.15 * symbolic)
        + (0.10 * constraint)
        + (0.10 * rl_bonus)
        + (0.05 * rl_value)
        - duplicate_penalty
        - budget_penalty
        - complexity_penalty
    )
    if budget_penalty > 0:
        return score, "per_family_budget"
    return score, ""


def build_examples_from_question(*, question: str, family: str) -> list[tuple[str, list[Any], float]]:
    tokens = [tok.strip(".,!?").lower() for tok in str(question or "").split()]
    has_percent = any("%" in tok or tok in {"percent", "discount"} for tok in tokens)
    has_rate = any(tok in {"per", "hour", "day", "minute", "speed", "each"} for tok in tokens)
    has_remaining = any(tok in {"left", "remain", "remaining"} for tok in tokens)

    examples: list[tuple[str, list[Any], float]] = []
    if has_percent:
        examples.append(("percent", ["owner", "item", "pct"], 0.1))
    if has_rate:
        examples.append(("rate", ["item", "rate"], 1.0))
    if has_remaining:
        examples.append(("remaining_quantity", ["owner", "item", "ans"], 1.0))
    examples.append((str(family or "state_transition"), ["owner", "item", "ans"], 1.0))
    return examples


def generate_ranked_candidates(*, question: str, family: str, cfg: ILPPipelineConfig) -> list[dict[str, Any]]:
    examples = build_examples_from_question(question=question, family=family)
    policy = build_context_policy(question=question, family=family)
    candidates: list[dict[str, Any]] = []

    ranked = propose_ranked_for_examples(
        examples,
        top_n=max(2, int(cfg.candidate_top_n)),
        include_agi_transfer_families=bool(cfg.include_agi_transfer_families),
    )
    for item in ranked:
        cand = dict(item)
        ok, reasons = candidate_matches_policy(cand, policy)
        if not ok:
            cand["policy_drop_reasons"] = reasons
            continue
        candidates.append(cand)

    learner = MetaRuleLearner(include_agi_transfer_families=bool(cfg.include_agi_transfer_families))
    for rule in learner.grammar:
        cand = rule.instantiate()
        ok, reasons = candidate_matches_policy(cand, policy)
        if not ok:
            continue
        cand.setdefault("ranking", {})
        cand["ranking"].setdefault("score", 0.45)
        cand["ranking"].setdefault("coverage", 0.6 if cand.get("family") == family else 0.45)
        cand["ranking"].setdefault("simplicity", 1.0 / max(1, len(cand.get("body") or [])))
        cand["ranking"].setdefault("symbolic_validity", 1.0)
        cand["ranking"].setdefault("constraint_validity", 0.8)
        cand["ranking"]["context_policy"] = {
            "cues": sorted(policy.cues),
            "allowed_families": sorted(policy.allowed_families),
            "max_body_len": policy.max_body_len,
        }
        candidates.append(cand)

    dedup: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        dedup.setdefault(candidate_signature(cand), cand)
    scored = list(dedup.values())
    scored.sort(
        key=lambda c: float((c.get("ranking") or {}).get("score", 0.0) or 0.0),
        reverse=True,
    )
    return scored[: max(1, int(cfg.candidate_top_n))]


def prescore_and_filter(
    candidates: list[dict[str, Any]],
    *,
    cfg: ILPPipelineConfig,
    utility_state: dict[str, Any] | None = None,
) -> tuple[list[CandidateRecord], list[CandidateRecord]]:
    accepted: list[CandidateRecord] = []
    dropped: list[CandidateRecord] = []
    seen_signatures: set[str] = set()
    family_counts: dict[str, int] = {}
    index = 0
    for candidate in candidates:
        index += 1
        family = str(candidate.get("family") or "unknown").strip().lower()
        signature = candidate_signature(candidate)
        utility = utility_components(utility_state or {"meta": {}, "rules": {}}, signature)
        candidate["rl_utility"] = utility
        prescore, budget_reason = _prescore_candidate(
            candidate,
            seen_signatures=seen_signatures,
            family_counts=family_counts,
            max_candidates_per_family=max(1, int(cfg.max_candidates_per_family)),
            utility_bonus=utility,
        )
        rec = CandidateRecord(
            candidate_id=f"ilp-candidate-{index:03d}",
            family=family,
            signature=signature,
            candidate=candidate,
            prescore=float(prescore),
        )
        if budget_reason:
            rec.dropped_reason = budget_reason
            dropped.append(rec)
            continue
        if prescore < float(cfg.prescore_min):
            rec.dropped_reason = f"prescore<{float(cfg.prescore_min):.3f}"
            dropped.append(rec)
            continue
        seen_signatures.add(signature)
        family_counts[family] = family_counts.get(family, 0) + 1
        accepted.append(rec)
    accepted.sort(key=lambda r: r.prescore, reverse=True)
    return accepted, dropped


def _parse_validation_result(payload: Any) -> tuple[bool, float, str]:
    if not isinstance(payload, dict):
        return False, 0.0, "validator_non_dict_response"
    valid_keys = ("valid", "accepted", "ok", "pass")
    valid = any(bool(payload.get(k)) for k in valid_keys if k in payload)
    score = 0.0
    for key in ("score", "confidence", "probability"):
        if key in payload:
            try:
                score = float(payload.get(key))
            except Exception:
                score = 0.0
            break
    reason = str(payload.get("reason") or payload.get("rationale") or "").strip()
    return bool(valid), float(score), reason


def _default_validator_call(
    *,
    raw_rule: str,
    family: str,
    question: str,
    cfg: ILPPipelineConfig,
) -> tuple[dict[str, Any], str]:
    system_prompt = (
        "You are a strict symbolic rule validator for an ILP pipeline. "
        "Judge only structural and semantic validity, not final GSM correctness. "
        "Return JSON only with keys: valid (bool), score (0..1), reason (short)."
    )
    user_prompt = (
        f"family: {family}\n"
        f"question: {question}\n"
        f"candidate_rule: {raw_rule}\n"
        "Return JSON only."
    )
    payload, err = request_ollama_chat_json(
        ollama_url=str(cfg.ollama_url),
        model=str(cfg.ollama_model),
        payload={
            "model": str(cfg.ollama_model),
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout_sec=int(cfg.llm_timeout_sec),
    )
    if err:
        return {}, err
    if not isinstance(payload, dict):
        return {}, "validator_non_dict_payload"
    return payload, ""


def selective_llm_validate(
    records: list[CandidateRecord],
    *,
    question: str,
    cfg: ILPPipelineConfig,
    cache: dict[str, dict[str, Any]] | None = None,
    validator_call: Callable[..., tuple[dict[str, Any], str]] | None = None,
) -> tuple[list[CandidateRecord], list[CandidateRecord], dict[str, Any]]:
    cache_store = cache if cache is not None else {}
    accepted: list[CandidateRecord] = []
    rejected: list[CandidateRecord] = []
    metrics = {
        "validated": 0,
        "cache_hits": 0,
        "latency_total_sec": 0.0,
        "latency_budget_exhausted": False,
    }
    if not records:
        return accepted, rejected, metrics

    if not cfg.llm_enabled:
        for rec in records:
            rec.llm_valid = True
            rec.llm_score = 0.0
            rec.llm_reason = "llm_disabled_bypass"
            accepted.append(rec)
        return accepted, rejected, metrics

    call_fn = validator_call or _default_validator_call
    max_validate = max(0, int(cfg.llm_validate_max))
    latency_budget = max(0.0, float(cfg.llm_latency_budget_sec))

    for index, rec in enumerate(records):
        if index >= max_validate:
            rec.llm_valid = False
            rec.llm_reason = "llm_topk_limit"
            rejected.append(rec)
            continue
        if metrics["latency_total_sec"] >= latency_budget:
            metrics["latency_budget_exhausted"] = True
            rec.llm_valid = False
            rec.llm_reason = "llm_latency_budget_exhausted"
            rejected.append(rec)
            continue

        if rec.signature in cache_store:
            cached = cache_store[rec.signature]
            rec.llm_valid = bool(cached.get("valid", False))
            rec.llm_score = float(cached.get("score", 0.0) or 0.0)
            rec.llm_reason = str(cached.get("reason") or "cache_hit")
            metrics["cache_hits"] += 1
            metrics["validated"] += 1
            if rec.llm_valid:
                accepted.append(rec)
            else:
                rejected.append(rec)
            continue

        raw_rule = candidate_to_raw_rule(rec.candidate)
        start = time.monotonic()
        payload, err = call_fn(
            raw_rule=raw_rule,
            family=rec.family,
            question=question,
            cfg=cfg,
        )
        latency = max(0.0, time.monotonic() - start)
        metrics["latency_total_sec"] += latency
        rec.llm_latency_sec = latency
        metrics["validated"] += 1

        if err:
            rec.llm_valid = False
            rec.llm_score = 0.0
            rec.llm_reason = err
            rejected.append(rec)
            continue

        valid, score, reason = _parse_validation_result(payload)
        rec.llm_valid = bool(valid)
        rec.llm_score = float(score)
        rec.llm_reason = str(reason or "llm_validated")
        cache_store[rec.signature] = {
            "valid": rec.llm_valid,
            "score": rec.llm_score,
            "reason": rec.llm_reason,
        }
        if rec.llm_valid:
            accepted.append(rec)
        else:
            rejected.append(rec)
    return accepted, rejected, metrics


def run_ilp_learning_stage(
    *,
    question: str,
    family: str,
    cfg: ILPPipelineConfig,
    validation_cache: dict[str, dict[str, Any]] | None = None,
    utility_state: dict[str, Any] | None = None,
    validator_call: Callable[..., tuple[dict[str, Any], str]] | None = None,
) -> dict[str, Any]:
    generated = generate_ranked_candidates(question=question, family=family, cfg=cfg)
    prescored, dropped = prescore_and_filter(generated, cfg=cfg, utility_state=utility_state)
    llm_ok, llm_reject, llm_metrics = selective_llm_validate(
        prescored,
        question=question,
        cfg=cfg,
        cache=validation_cache,
        validator_call=validator_call,
    )
    return {
        "config": {
            "candidate_top_n": cfg.candidate_top_n,
            "prescore_min": cfg.prescore_min,
            "max_candidates_per_family": cfg.max_candidates_per_family,
            "include_agi_transfer_families": cfg.include_agi_transfer_families,
            "llm_validate_max": cfg.llm_validate_max,
            "llm_latency_budget_sec": cfg.llm_latency_budget_sec,
            "llm_enabled": cfg.llm_enabled,
        },
        "counts": {
            "generated": len(generated),
            "passed_prescore": len(prescored),
            "dropped_prescore": len(dropped),
            "llm_accepted": len(llm_ok),
            "llm_rejected": len(llm_reject),
        },
        "llm_metrics": llm_metrics,
        "generated_candidates": [
            {
                "family": str(c.get("family") or ""),
                "name": str(c.get("name") or ""),
                "head": c.get("head"),
                "body": c.get("body"),
                "calc": c.get("calc"),
                "target": c.get("target"),
                "ranking": c.get("ranking"),
            }
            for c in generated
        ],
        "prescored_candidates": [
            {
                "candidate_id": rec.candidate_id,
                "family": rec.family,
                "prescore": rec.prescore,
                "signature": rec.signature,
                "rl_utility": rec.candidate.get("rl_utility"),
            }
            for rec in prescored
        ],
        "accepted_candidates": [
            {
                "candidate_id": rec.candidate_id,
                "family": rec.family,
                "prescore": rec.prescore,
                "signature": rec.signature,
                "llm_score": rec.llm_score,
                "llm_reason": rec.llm_reason,
                "candidate": rec.candidate,
            }
            for rec in llm_ok
        ],
        "dropped_candidates": [
            {
                "candidate_id": rec.candidate_id,
                "family": rec.family,
                "prescore": rec.prescore,
                "dropped_reason": rec.dropped_reason or rec.llm_reason,
                "candidate": rec.candidate,
            }
            for rec in (dropped + llm_reject)
        ],
    }
