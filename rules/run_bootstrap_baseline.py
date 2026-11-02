#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "logic_transformer") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "logic_transformer"))

from cognitive_loop import (
    FINALITY_REGRESSION_FAMILIES,
    GENERALIZED_TEMPLATE_FAMILIES,
    MIXED_NARRATIVE_SYMBOLIC_FAMILIES,
    TINYSTORIES_STATE_TRACKING_FAMILIES,
    _compile_gsm8k_problem_to_wm,
    _extract_numeric_value,
    system1_step,
    system2_step,
    system3_step,
)
from rules.ilp_candidate_pipeline import (
    ILPPipelineConfig,
    candidate_to_raw_rule,
    run_ilp_learning_stage,
)
from rules.ilp_rule_utility import apply_reward_updates, load_utility_state, save_utility_state
from semantic_ar import SemanticARConfig, load_semantic_ar_config
from rules.teacher_ollama import request_ollama_teacher_proposals
from working_memory.pot_dsl import execute_program, propose_program_for_question, wm_nodes_from_trace
from working_memory.working_memory import WMNode, WorkingMemory


@dataclass(frozen=True)
class ProfileSpec:
    profile_id: str
    description: str
    systems: tuple[str, ...]


@dataclass
class TeacherIngestionConfig:
    enabled: bool
    source: str
    teacher_id: str
    teacher_model: str
    ab_arm: str
    low_volume_mode: bool
    max_total: int
    max_rules: int
    max_facts: int
    max_counterfactuals: int
    max_rule_chars: int
    greedy_by_score: bool
    min_score: float
    static_proposals: Any
    ollama_url: str
    ollama_model: str
    ollama_timeout_sec: int
    ollama_cache: dict[str, Any]


@dataclass
class ILPAssistConfig:
    enabled: bool
    pipeline: ILPPipelineConfig
    validation_cache: dict[str, dict[str, Any]]
    utility_state: dict[str, Any]
    utility_path: str
    register_proposals: bool
    max_registered_per_case: int


PROFILE_SPECS: dict[str, ProfileSpec] = {
    "lt-heavy": ProfileSpec(
        profile_id="lt-heavy",
        description="System-1 proposal + System-2 validation/reasoning only",
        systems=("system1", "system2"),
    ),
    "hybrid": ProfileSpec(
        profile_id="hybrid",
        description="Full tri-system loop (System-1 + System-2 + System-3)",
        systems=("system1", "system2", "system3"),
    ),
    "ilp-heavy": ProfileSpec(
        profile_id="ilp-heavy",
        description="System-1 grounding + System-3 abduction/revision emphasis",
        systems=("system1", "system3"),
    ),
}

DEFAULT_OUTPUT = Path("rules/reports/agi_bootstrap_baseline_v1.json")
DEFAULT_COMMAND = (
    "python rules/run_bootstrap_baseline.py "
    "--suite combined --max-cycles 2 "
    "--output rules/reports/agi_bootstrap_baseline_v1.json"
)

METRIC_DEFINITIONS = {
    "capability": "solve_rate = solved_cases / total_cases",
    "symbolic_validity": (
        "symbolic_valid_rate = cases_with_numeric_reasoning_belief / total_cases; "
        "a numeric reasoning belief is a Belief(value=number) not emitted by gsm8k_compiler"
    ),
    "sample_efficiency": (
        "valid_solves_per_sample = solved_and_symbolic_valid_cases / total_cases; "
        "samples_per_valid_solve = total_cases / solved_and_symbolic_valid_cases"
    ),
    "compute_cost": (
        "wall_time_sec_total, wall_time_sec_per_case, "
        "wall_time_sec_per_valid_solve"
    ),
}


def _suite_cases(name: str) -> list[dict[str, Any]]:
    if name == "template_generalization":
        return list(GENERALIZED_TEMPLATE_FAMILIES)
    if name == "finality_regression":
        return list(FINALITY_REGRESSION_FAMILIES)
    if name == "tinystories_state_tracking":
        return list(TINYSTORIES_STATE_TRACKING_FAMILIES)
    if name == "mixed_narrative_symbolic":
        return list(MIXED_NARRATIVE_SYMBOLIC_FAMILIES)
    if name == "combined":
        return list(FINALITY_REGRESSION_FAMILIES) + list(GENERALIZED_TEMPLATE_FAMILIES)
    if name == "combined_widening":
        return (
            list(FINALITY_REGRESSION_FAMILIES)
            + list(GENERALIZED_TEMPLATE_FAMILIES)
            + list(TINYSTORIES_STATE_TRACKING_FAMILIES)
            + list(MIXED_NARRATIVE_SYMBOLIC_FAMILIES)
        )
    raise ValueError(f"unsupported suite: {name}")


def _system_output(
    system_name: str,
    current_wm_snapshot: set,
    cycle: int,
    question: str,
    sar_config: dict[str, Any],
):
    if system_name == "system1":
        return system1_step(current_wm_snapshot, cycle, question, True, sar_config)
    if system_name == "system2":
        return system2_step(current_wm_snapshot, cycle, None, True)
    if system_name == "system3":
        return system3_step(current_wm_snapshot, cycle, None, True)
    raise ValueError(f"unsupported system: {system_name}")


def _build_sar_config(args: argparse.Namespace) -> dict[str, Any]:
    if str(getattr(args, "sar_config_file", "") or "").strip():
        src = Path(str(args.sar_config_file))
        if not src.is_absolute():
            src = REPO_ROOT / src
        loaded = load_semantic_ar_config(src).to_dict()
        if args.sar_enabled:
            loaded["enabled"] = True
        if args.sar_emit_metadata:
            loaded["emit_metadata"] = True
        return SemanticARConfig.from_mapping(loaded).to_dict()
    return SemanticARConfig.from_mapping(
        {
            "enabled": args.sar_enabled,
            "weight": args.sar_weight,
            "lexical_weight": args.sar_lexical_weight,
            "role_weight": args.sar_role_weight,
            "prior_weight": args.sar_prior_weight,
            "gate_fail_penalty": args.sar_gate_fail_penalty,
            "couple_to_symbolic_gate": not args.sar_disable_symbolic_gate_coupling,
            "emit_metadata": args.sar_emit_metadata,
        }
    ).to_dict()


def _load_teacher_proposals(path: str) -> Any:
    candidate = str(path or "").strip()
    if not candidate:
        return {}
    src = Path(candidate)
    if not src.is_absolute():
        src = REPO_ROOT / src
    if not src.exists():
        raise FileNotFoundError(f"teacher proposal file not found: {src}")
    payload = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(payload, (dict, list)):
        raise ValueError("teacher proposal payload must be a JSON object or list")
    return payload





def _apply_reasoner_budget_env(args: argparse.Namespace) -> dict[str, Any]:
    env_map = {
        "AGI_ILP_MAX_DEPTH": int(args.ilp_max_depth),
        "AGI_ILP_MAX_FACT_MATCHES": int(args.ilp_max_fact_matches),
        "AGI_ILP_MAX_RULE_MATCHES": int(args.ilp_max_rule_matches),
        "AGI_ILP_MAX_SUBGOAL_RESULTS": int(args.ilp_max_subgoal_results),
    }
    for key, value in env_map.items():
        if value > 0:
            os.environ[key] = str(value)
        else:
            os.environ.pop(key, None)
    os.environ["AGI_NUMERIC_AUTHORITY_STRICT"] = "1" if args.numeric_authority_strict else "0"
    return {
        "strict_numeric_authority": bool(args.numeric_authority_strict),
        "ilp_max_depth": int(args.ilp_max_depth),
        "ilp_max_fact_matches": int(args.ilp_max_fact_matches),
        "ilp_max_rule_matches": int(args.ilp_max_rule_matches),
        "ilp_max_subgoal_results": int(args.ilp_max_subgoal_results),
    }


def _run_case(
    *,
    question: str,
    family: str,
    expected_answer: Any,
    systems: tuple[str, ...],
    max_cycles: int,
    sar_config: dict[str, Any],
    pot_enable: bool,
    teacher_config: TeacherIngestionConfig | None = None,
    ilp_assist_config: ILPAssistConfig | None = None,
) -> dict[str, Any]:
    wm = _compile_gsm8k_problem_to_wm(question, expected_answer=expected_answer)
    ilp_stage_report: dict[str, Any] = {"enabled": False, "reason": "not requested"}
    ilp_registered_rule_ids: list[str] = []
    if ilp_assist_config is not None and ilp_assist_config.enabled:
        ilp_stage_report = run_ilp_learning_stage(
            question=question,
            family=family,
            cfg=ilp_assist_config.pipeline,
            validation_cache=ilp_assist_config.validation_cache,
            utility_state=ilp_assist_config.utility_state,
        )
        if ilp_assist_config.register_proposals:
            accepted = list(ilp_stage_report.get("accepted_candidates") or [])
            kept = accepted[: max(0, int(ilp_assist_config.max_registered_per_case))]
            logged = 0
            for row in kept:
                candidate = row.get("candidate") if isinstance(row, dict) else None
                if not isinstance(candidate, dict):
                    continue
                signature = str(row.get("candidate_id") or "candidate").replace(":", "_")
                rule_id = f"ilp_{family}_{signature}".replace("-", "_").replace(" ", "_")
                raw_rule = candidate_to_raw_rule(candidate)
                compiled_payload = {
                    "rule_id": rule_id,
                    "family": str(candidate.get("family") or family),
                    "condition": f"body={candidate.get('body')}",
                    "effect": f"head={candidate.get('head')}; calc={candidate.get('calc')}",
                    "priority": 50,
                }
                entry = wm.rule_registry.register_proposal(
                    rule_id=rule_id,
                    raw_rule=raw_rule,
                    source="ilp_generator",
                    compiled_payload=compiled_payload,
                    proposal_kind="ilp_rule",
                    runtime_authority="proposal_only",
                    activation_policy="proposal_only",
                    evidence={
                        "case_family": family,
                        "question": question,
                        "prescore": row.get("prescore"),
                        "llm_score": row.get("llm_score"),
                    },
                    audit={
                        "pipeline": "ilp_first_latency_adaptive",
                        "llm_reason": row.get("llm_reason"),
                    },
                )
                wm.facts[WMNode.create(
                    "ILPRuleProposal",
                    layer="meta",
                    source="ilp_generator",
                    provenance_id=rule_id,
                    rule_id=rule_id,
                    family=compiled_payload.get("family"),
                    status=entry.get("status", "proposed"),
                    runtime_authority="proposal_only",
                )] = wm.default_ttl
                ilp_registered_rule_ids.append(rule_id)
                logged += 1
            ilp_stage_report["registered_proposals"] = logged
    teacher_report = {
        "enabled": bool(teacher_config.enabled) if teacher_config else False,
        "ab_arm": teacher_config.ab_arm if teacher_config else "control",
        "reason": "not requested",
    }
    if teacher_config is not None and (teacher_config.enabled or teacher_config.static_proposals):
        key = f"{family}::{question}"
        proposals = teacher_config.static_proposals
        teacher_fetch_error = ""
        if teacher_config.ollama_url and teacher_config.ollama_model:
            if key in teacher_config.ollama_cache:
                proposals = teacher_config.ollama_cache[key]
            else:
                proposals, teacher_fetch_error = request_ollama_teacher_proposals(
                    ollama_url=teacher_config.ollama_url,
                    model=teacher_config.ollama_model,
                    question=question,
                    family=family,
                    timeout_sec=teacher_config.ollama_timeout_sec,
                )
                teacher_config.ollama_cache[key] = proposals
        teacher_report = wm.ingest_teacher_proposals(
            proposals,
            source=str(teacher_config.source),
            teacher_id=str(teacher_config.teacher_id),
            teacher_model=str(teacher_config.teacher_model),
            ab_arm=str(teacher_config.ab_arm),
            enable_ingestion=bool(teacher_config.enabled),
            low_volume_mode=bool(teacher_config.low_volume_mode),
            max_total=int(teacher_config.max_total),
            max_rules=int(teacher_config.max_rules),
            max_facts=int(teacher_config.max_facts),
            max_counterfactuals=int(teacher_config.max_counterfactuals),
            max_rule_chars=int(teacher_config.max_rule_chars),
            greedy_by_score=bool(teacher_config.greedy_by_score),
            min_score=float(teacher_config.min_score),
        )
        if teacher_fetch_error:
            teacher_report["fetch_error"] = teacher_fetch_error
    pot_report = {
        "enabled": bool(pot_enable),
        "program_id": None,
        "applied": False,
        "execution_ok": False,
        "issue_count": 0,
    }
    if pot_enable:
        proposal = propose_program_for_question(question=question, family=family)
        if proposal is not None:
            pot_report["program_id"] = proposal.program_id
            pot_exec = execute_program(proposal)
            pot_report["execution_ok"] = bool(pot_exec.ok)
            pot_report["issue_count"] = len(pot_exec.issues)
            if pot_exec.ok:
                wm.update(wm_nodes_from_trace(pot_exec.trace))
                pot_report["applied"] = True

    prior_fact_count = len(wm.facts)
    prior_belief_count = sum(1 for fact in wm.facts if fact.concept == "Belief")

    cycles_executed = 0
    for cycle in range(1, max_cycles + 1):
        cycles_executed = cycle
        current_wm_snapshot = set(wm.facts.keys())
        for system_name in systems:
            out = _system_output(system_name, current_wm_snapshot, cycle, question, sar_config)
            wm.update(out)
        wm.validate_equation_graph()
        wm.decay()

        fact_count_after = len(wm.facts)
        belief_count_after = sum(1 for fact in wm.facts if fact.concept == "Belief")
        if cycle > 1 and fact_count_after == prior_fact_count and belief_count_after == prior_belief_count:
            break
        prior_fact_count = fact_count_after
        prior_belief_count = belief_count_after

    belief_values = []
    has_numeric_reasoning_belief = False
    for fact in wm.facts:
        if fact.concept != "Belief":
            continue
        numeric_value = _extract_numeric_value(fact.to_dict().get("value"))
        if numeric_value is None:
            continue
        belief_values.append(float(numeric_value))
        if fact.source != "gsm8k_compiler" and fact.layer in {"reasoning", "meta"}:
            has_numeric_reasoning_belief = True

    expected = _extract_numeric_value(expected_answer)
    solved = False
    if expected is not None:
        solved = any(abs(v - expected) < 1e-9 for v in belief_values)
    solved_and_valid = bool(solved and has_numeric_reasoning_belief)
    if ilp_assist_config is not None and ilp_assist_config.enabled:
        base_reward = 1.0 if solved_and_valid else (0.35 if has_numeric_reasoning_belief else -0.35)
        reward_updates: list[dict[str, Any]] = []
        accepted_signatures = {
            str(item.get("signature") or "")
            for item in (ilp_stage_report.get("accepted_candidates") or [])
            if isinstance(item, dict)
        }
        prescored_signatures = {
            str(item.get("signature") or "")
            for item in (ilp_stage_report.get("prescored_candidates") or [])
            if isinstance(item, dict)
        }
        for signature in sorted(s for s in prescored_signatures if s):
            reward = base_reward if signature in accepted_signatures else (0.05 if has_numeric_reasoning_belief else -0.1)
            reward_updates.append({"signature": signature, "reward": reward})
        update_report = apply_reward_updates(ilp_assist_config.utility_state, reward_updates)
        ilp_stage_report["rl_reward"] = {
            "base_reward": base_reward,
            "updates": update_report.get("applied", 0),
        }
    if ilp_registered_rule_ids:
        for rule_id in ilp_registered_rule_ids:
            wm.rule_registry.record_accounting_event(
                rule_id,
                applied=1,
                wins=1 if solved_and_valid else 0,
                defeats=0 if solved_and_valid else 1,
                recall_accepted=1 if has_numeric_reasoning_belief else 0,
                recall_rejected=0 if has_numeric_reasoning_belief else 1,
                event="ilp_case_feedback",
                details={
                    "family": family,
                    "solved": bool(solved),
                    "symbolic_valid": bool(has_numeric_reasoning_belief),
                },
            )
        ilp_stage_report["registered_rule_ids"] = ilp_registered_rule_ids
        ilp_stage_report["delayed_feedback"] = {
            "solved_and_symbolic_valid": solved_and_valid,
            "applied_updates": len(ilp_registered_rule_ids),
        }

    return {
        "expected": expected,
        "belief_values": belief_values,
        "solved": bool(solved),
        "symbolic_valid": bool(has_numeric_reasoning_belief),
        "cycles_executed": cycles_executed,
        "pot": pot_report,
        "teacher_ingestion": teacher_report,
        "ilp_learning_stage": ilp_stage_report,
    }


def _summarize_profile(*, profile: ProfileSpec, rows: list[dict[str, Any]], duration_sec: float) -> dict[str, Any]:
    total = len(rows)
    solved = sum(1 for row in rows if row["solved"])
    symbolic_valid = sum(1 for row in rows if row["symbolic_valid"])
    valid_solves = sum(1 for row in rows if row["solved"] and row["symbolic_valid"])

    def _safe_div(numerator: float, denominator: float) -> float:
        return float(numerator) / float(denominator) if denominator else 0.0

    per_case = _safe_div(duration_sec, total)
    per_valid_solve = _safe_div(duration_sec, valid_solves) if valid_solves else None
    samples_per_valid_solve = _safe_div(total, valid_solves) if valid_solves else None

    return {
        "profile": profile.profile_id,
        "description": profile.description,
        "systems": list(profile.systems),
        "total_cases": total,
        "solved_cases": solved,
        "symbolic_valid_cases": symbolic_valid,
        "solved_and_symbolic_valid_cases": valid_solves,
        "capability": {
            "solve_rate": _safe_div(solved, total),
        },
        "symbolic_validity": {
            "symbolic_valid_rate": _safe_div(symbolic_valid, total),
        },
        "sample_efficiency": {
            "valid_solves_per_sample": _safe_div(valid_solves, total),
            "samples_per_valid_solve": samples_per_valid_solve,
        },
        "compute_cost": {
            "wall_time_sec_total": duration_sec,
            "wall_time_sec_per_case": per_case,
            "wall_time_sec_per_valid_solve": per_valid_solve,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the AGI bootstrap baseline contract across LT-heavy, hybrid, and ILP-heavy profiles."
    )
    parser.add_argument(
        "--suite",
        choices=[
            "template_generalization",
            "finality_regression",
            "tinystories_state_tracking",
            "mixed_narrative_symbolic",
            "combined",
            "combined_widening",
        ],
        default="combined",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["lt-heavy", "hybrid", "ilp-heavy"],
        choices=sorted(PROFILE_SPECS.keys()),
    )
    parser.add_argument("--max-cycles", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on suite cases; 0 means full suite.")
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Optional family ids to run inside the selected suite (preserves suite order).",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--sar-enabled", action="store_true", help="Enable semantic-AR proposal shaping in System 1.")
    parser.add_argument("--sar-config-file", default="", help="Optional JSON file containing a trained SAR config.")
    parser.add_argument("--sar-weight", type=float, default=0.2, help="SAR bonus weight added to proposal scores.")
    parser.add_argument("--sar-lexical-weight", type=float, default=0.5, help="Weight for lexical alignment inside SAR.")
    parser.add_argument("--sar-role-weight", type=float, default=0.3, help="Weight for role-coverage inside SAR.")
    parser.add_argument("--sar-prior-weight", type=float, default=0.2, help="Weight for concept prior inside SAR.")
    parser.add_argument("--sar-gate-fail-penalty", type=float, default=0.0, help="Penalty applied when symbolic pre-gate fails.")
    parser.add_argument("--sar-disable-symbolic-gate-coupling", action="store_true", help="Ablation: allow SAR bonus without symbolic pre-gate pass.")
    parser.add_argument("--sar-emit-metadata", action="store_true", help="Attach SAR score metadata to System-1 parse facts.")
    parser.add_argument("--pot-enable", action="store_true", help="Enable PoT DSL proposal execution and WM trace injection.")
    parser.add_argument("--teacher-proposals", default="", help="Optional JSON file containing teacher rule/fact/counterfactual candidates.")
    parser.add_argument("--teacher-enable-ingestion", action="store_true", help="Enable teacher-LLM proposal ingestion preflight.")
    parser.add_argument("--teacher-ab-arm", choices=["control", "treatment"], default="control", help="A/B arm for teacher proposal ingestion.")
    parser.add_argument("--teacher-low-volume", dest="teacher_low_volume", action="store_true", default=True, help="Enable strict low-volume proposal caps.")
    parser.add_argument("--teacher-no-low-volume", dest="teacher_low_volume", action="store_false", help="Disable strict low-volume proposal caps.")
    parser.add_argument("--teacher-id", default="bootstrap-teacher-llm")
    parser.add_argument("--teacher-model", default="unknown")
    parser.add_argument("--teacher-max-total", type=int, default=3)
    parser.add_argument("--teacher-max-rules", type=int, default=1)
    parser.add_argument("--teacher-max-facts", type=int, default=1)
    parser.add_argument("--teacher-max-counterfactuals", type=int, default=1)
    parser.add_argument("--teacher-max-rule-chars", type=int, default=280)
    parser.add_argument("--teacher-greedy-by-score", dest="teacher_greedy_by_score", action="store_true", default=True, help="Greedily rank teacher proposals by score/confidence before applying volume caps.")
    parser.add_argument("--teacher-no-greedy-by-score", dest="teacher_greedy_by_score", action="store_false", help="Preserve input order for teacher proposal ingestion.")
    parser.add_argument("--teacher-min-score", type=float, default=0.0, help="Minimum teacher proposal score/confidence to consider for ingestion.")
    parser.add_argument("--teacher-ollama-url", default="", help="Optional Ollama /api/chat URL for dynamic teacher proposals.")
    parser.add_argument("--teacher-ollama-model", default="", help="Ollama model tag to query for teacher proposals.")
    parser.add_argument("--teacher-ollama-timeout-sec", type=int, default=180, help="HTTP timeout in seconds for Ollama teacher calls.")
    parser.add_argument("--ilp-enable-learning-stage", action="store_true", help="Enable ILP-first candidate generation and selective LLM validation stage per case.")
    parser.add_argument("--ilp-prescore-min", type=float, default=0.45, help="Minimum pre-score for ILP candidates before LLM validation.")
    parser.add_argument("--ilp-candidate-top-n", type=int, default=6, help="Top-N ILP candidates to generate per case before filtering.")
    parser.add_argument("--ilp-max-candidates-per-family", type=int, default=3, help="Per-family cap after pre-scoring.")
    parser.add_argument("--ilp-include-agi-transfer-families", action="store_true", help="Include transfer-oriented metarule families in ILP candidate generation.")
    parser.add_argument("--ilp-llm-validate-max", type=int, default=2, help="Maximum ILP candidates to LLM-validate per case.")
    parser.add_argument("--ilp-llm-latency-budget-sec", type=float, default=12.0, help="Per-case latency budget for ILP LLM validation.")
    parser.add_argument("--ilp-llm-timeout-sec", type=int, default=20, help="Timeout for each ILP LLM validation request.")
    parser.add_argument("--ilp-llm-enable-validator", action="store_true", help="Enable remote LLM validation for ILP candidates.")
    parser.add_argument("--ilp-llm-ollama-url", default="", help="Optional Ollama /api/chat URL for ILP candidate validation (falls back to teacher URL).")
    parser.add_argument("--ilp-llm-ollama-model", default="", help="Optional Ollama model for ILP candidate validation (falls back to teacher model).")
    parser.add_argument("--ilp-register-proposals", action="store_true", help="Register accepted ILP candidates in rule registry as proposal-only entries.")
    parser.add_argument("--ilp-max-registered-per-case", type=int, default=2, help="Max accepted ILP proposals to register per case.")
    parser.add_argument("--ilp-utility-path", default="rules/reports/ilp_rule_utility_v1.json", help="Path to persisted ILP rule utility state used for RL-style scoring.")
    parser.add_argument("--seed", type=int, default=0, help="Optional deterministic seed; 0 keeps default runtime behavior.")
    parser.add_argument("--ilp-max-depth", type=int, default=0, help="Optional ILP reasoner max depth; 0 keeps default.")
    parser.add_argument("--ilp-max-fact-matches", type=int, default=0, help="Optional cap on matched facts per predicate; 0 keeps default.")
    parser.add_argument("--ilp-max-rule-matches", type=int, default=0, help="Optional cap on matched rules per predicate; 0 keeps default.")
    parser.add_argument("--ilp-max-subgoal-results", type=int, default=0, help="Optional cap on retained subgoal bindings; 0 keeps default.")
    parser.add_argument("--numeric-authority-strict", action="store_true", help="Restrict numeric quantity inference to owner/item authority neighborhood.")
    args = parser.parse_args()
    sar_config = _build_sar_config(args)
    reasoner_budget = _apply_reasoner_budget_env(args)

    if args.seed:
        random.seed(int(args.seed))
        try:
            import numpy as np  # type: ignore
            np.random.seed(int(args.seed))
        except Exception:
            pass
        try:
            import torch  # type: ignore
            torch.manual_seed(int(args.seed))
        except Exception:
            pass

    teacher_proposals = _load_teacher_proposals(args.teacher_proposals)
    teacher_cfg = TeacherIngestionConfig(
        enabled=bool(args.teacher_enable_ingestion),
        source="teacher_llm",
        teacher_id=str(args.teacher_id),
        teacher_model=str(args.teacher_model or args.teacher_ollama_model or "unknown"),
        ab_arm=str(args.teacher_ab_arm),
        low_volume_mode=bool(args.teacher_low_volume),
        max_total=int(args.teacher_max_total),
        max_rules=int(args.teacher_max_rules),
        max_facts=int(args.teacher_max_facts),
        max_counterfactuals=int(args.teacher_max_counterfactuals),
        max_rule_chars=int(args.teacher_max_rule_chars),
        greedy_by_score=bool(args.teacher_greedy_by_score),
        min_score=float(args.teacher_min_score),
        static_proposals=teacher_proposals,
        ollama_url=str(args.teacher_ollama_url),
        ollama_model=str(args.teacher_ollama_model),
        ollama_timeout_sec=int(args.teacher_ollama_timeout_sec),
        ollama_cache={},
    )
    ilp_ollama_url = str(args.ilp_llm_ollama_url or args.teacher_ollama_url or "")
    ilp_ollama_model = str(args.ilp_llm_ollama_model or args.teacher_ollama_model or "")
    utility_path_obj = Path(str(args.ilp_utility_path))
    if not utility_path_obj.is_absolute():
        utility_path_obj = REPO_ROOT / utility_path_obj
    utility_state = load_utility_state(utility_path_obj)
    ilp_cfg = ILPAssistConfig(
        enabled=bool(args.ilp_enable_learning_stage),
        pipeline=ILPPipelineConfig(
            candidate_top_n=max(1, int(args.ilp_candidate_top_n)),
            prescore_min=float(args.ilp_prescore_min),
            max_candidates_per_family=max(1, int(args.ilp_max_candidates_per_family)),
            include_agi_transfer_families=bool(args.ilp_include_agi_transfer_families),
            llm_validate_max=max(0, int(args.ilp_llm_validate_max)),
            llm_latency_budget_sec=max(0.0, float(args.ilp_llm_latency_budget_sec)),
            llm_timeout_sec=max(3, int(args.ilp_llm_timeout_sec)),
            llm_enabled=bool(args.ilp_llm_enable_validator),
            ollama_url=ilp_ollama_url,
            ollama_model=ilp_ollama_model,
        ),
        validation_cache={},
        utility_state=utility_state,
        utility_path=str(utility_path_obj),
        register_proposals=bool(args.ilp_register_proposals),
        max_registered_per_case=max(0, int(args.ilp_max_registered_per_case)),
    )

    suite_rows = _suite_cases(args.suite)
    if args.families:
        requested_families = [str(item).strip() for item in args.families if str(item).strip()]
        requested_set = set(requested_families)
        available_families = {row.get("family") for row in suite_rows if row.get("family")}
        missing_families = sorted(requested_set.difference(available_families))
        if missing_families:
            raise ValueError(
                f"unknown families for suite '{args.suite}': {', '.join(missing_families)}"
            )
        suite_rows = [row for row in suite_rows if row.get("family") in requested_set]
    if args.limit > 0:
        suite_rows = suite_rows[: args.limit]
    if not suite_rows:
        raise ValueError("selected suite has no cases")

    profile_reports = []
    for profile_id in args.profiles:
        profile = PROFILE_SPECS[profile_id]
        if not args.quiet:
            print(f"[{profile.profile_id}] running {len(suite_rows)} cases with systems={profile.systems} ...")
        started = time.perf_counter()
        results = []
        for row in suite_rows:
            question = row.get("question")
            expected = row.get("expected")
            if not question:
                continue
            run = _run_case(
                question=question,
                family=str(row.get("family", "unknown")),
                expected_answer=expected,
                systems=profile.systems,
                max_cycles=max(1, args.max_cycles),
                sar_config=sar_config,
                pot_enable=bool(args.pot_enable),
                teacher_config=teacher_cfg,
                ilp_assist_config=ilp_cfg,
            )
            run["family"] = row.get("family", "unknown")
            run["question"] = question
            results.append(run)
        duration_sec = time.perf_counter() - started
        profile_report = _summarize_profile(profile=profile, rows=results, duration_sec=duration_sec)
        profile_report["results"] = results
        teacher_events = [r.get("teacher_ingestion", {}) for r in results if isinstance(r.get("teacher_ingestion"), dict)]
        if teacher_events:
            agg = {"submitted": 0, "processed": 0, "accepted": 0, "rejected": 0, "dropped_by_volume": 0}
            for ev in teacher_events:
                counts = ev.get("counts", {}) if isinstance(ev.get("counts"), dict) else {}
                for key in agg:
                    agg[key] += int(counts.get(key, 0) or 0)
            profile_report["teacher_ingestion_summary"] = {
                "enabled": bool(args.teacher_enable_ingestion),
                "ab_arm": str(args.teacher_ab_arm),
                "counts": agg,
                "dynamic_ollama": bool(args.teacher_ollama_url and args.teacher_ollama_model),
            }
        ilp_events = [r.get("ilp_learning_stage", {}) for r in results if isinstance(r.get("ilp_learning_stage"), dict)]
        if ilp_events:
            ilp_counts = {
                "generated": 0,
                "passed_prescore": 0,
                "dropped_prescore": 0,
                "llm_accepted": 0,
                "llm_rejected": 0,
                "validated": 0,
                "cache_hits": 0,
            }
            latency_total = 0.0
            latency_budget_exhausted = 0
            for ev in ilp_events:
                counts = ev.get("counts", {}) if isinstance(ev.get("counts"), dict) else {}
                metrics = ev.get("llm_metrics", {}) if isinstance(ev.get("llm_metrics"), dict) else {}
                for key in ("generated", "passed_prescore", "dropped_prescore", "llm_accepted", "llm_rejected"):
                    ilp_counts[key] += int(counts.get(key, 0) or 0)
                ilp_counts["validated"] += int(metrics.get("validated", 0) or 0)
                ilp_counts["cache_hits"] += int(metrics.get("cache_hits", 0) or 0)
                latency_total += float(metrics.get("latency_total_sec", 0.0) or 0.0)
                if bool(metrics.get("latency_budget_exhausted", False)):
                    latency_budget_exhausted += 1
            profile_report["ilp_learning_summary"] = {
                "enabled": bool(args.ilp_enable_learning_stage),
                "llm_validator_enabled": bool(args.ilp_llm_enable_validator),
                "counts": ilp_counts,
                "latency_total_sec": latency_total,
                "latency_budget_exhausted_cases": latency_budget_exhausted,
            }
        profile_reports.append(profile_report)

    teacher_report: dict[str, Any] = {
        "enabled": bool(args.teacher_enable_ingestion),
        "ab_arm": str(args.teacher_ab_arm),
        "low_volume_mode": bool(args.teacher_low_volume),
        "dynamic_ollama": bool(args.teacher_ollama_url and args.teacher_ollama_model),
    }
    total_counts = {"submitted": 0, "processed": 0, "accepted": 0, "rejected": 0, "dropped_by_volume": 0}
    for prof in profile_reports:
        summary = prof.get("teacher_ingestion_summary", {})
        counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
        for key in total_counts:
            total_counts[key] += int(counts.get(key, 0) or 0)
    teacher_report["counts"] = total_counts
    ilp_report: dict[str, Any] = {
        "enabled": bool(args.ilp_enable_learning_stage),
        "llm_validator_enabled": bool(args.ilp_llm_enable_validator),
        "llm_validator_remote": bool(ilp_ollama_url and ilp_ollama_model),
        "register_proposals": bool(args.ilp_register_proposals),
    }
    ilp_total_counts = {
        "generated": 0,
        "passed_prescore": 0,
        "dropped_prescore": 0,
        "llm_accepted": 0,
        "llm_rejected": 0,
        "validated": 0,
        "cache_hits": 0,
    }
    ilp_latency_total = 0.0
    for prof in profile_reports:
        summary = prof.get("ilp_learning_summary", {})
        counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
        for key in ilp_total_counts:
            ilp_total_counts[key] += int(counts.get(key, 0) or 0)
        ilp_latency_total += float(summary.get("latency_total_sec", 0.0) or 0.0)
    ilp_report["counts"] = ilp_total_counts
    ilp_report["latency_total_sec"] = ilp_latency_total
    ilp_report["utility_path"] = ilp_cfg.utility_path
    ilp_report["utility_total_updates"] = int((ilp_cfg.utility_state.get("meta", {}) or {}).get("total_updates", 0) or 0)
    save_utility_state(ilp_cfg.utility_path, ilp_cfg.utility_state)

    output = {
        "contract_id": "agi-bootstrap-baseline-contract-v1",
        "suite": args.suite,
        "requested_families": list(args.families or []),
        "families": [row.get("family", "unknown") for row in suite_rows],
        "max_cycles": max(1, args.max_cycles),
        "case_count": len(suite_rows),
        "seed": int(args.seed),
        "baseline_command": DEFAULT_COMMAND,
        "proposal_shaping": {
            "semantic_ar": sar_config,
            "pot_dsl": {
                "enabled": bool(args.pot_enable),
                "mode": "proposal_trace_injection",
                "runtime_authority": "proposal_only",
            },
            "symbolic_validator_authority": "System-2 semantic validation and WM equation-graph checks remain authoritative.",
        },
        "search_budget": reasoner_budget,
        "ilp_learning_stage": ilp_report,
        "teacher_proposal_ingestion": teacher_report,
        "metric_definitions": METRIC_DEFINITIONS,
        "profiles": profile_reports,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    for report in profile_reports:
        print(
            f"{report['profile']}: solved={report['solved_cases']}/{report['total_cases']} "
            f"symbolic_valid={report['symbolic_valid_cases']}/{report['total_cases']} "
            f"time={report['compute_cost']['wall_time_sec_total']:.2f}s"
        )
    print(f"Saved baseline report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
