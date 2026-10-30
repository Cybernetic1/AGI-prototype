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
from semantic_ar import SemanticARConfig, load_semantic_ar_config
from working_memory.pot_dsl import execute_program, propose_program_for_question, wm_nodes_from_trace
from working_memory.working_memory import WorkingMemory


@dataclass(frozen=True)
class ProfileSpec:
    profile_id: str
    description: str
    systems: tuple[str, ...]


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
) -> dict[str, Any]:
    wm = _compile_gsm8k_problem_to_wm(question, expected_answer=expected_answer)
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

    return {
        "expected": expected,
        "belief_values": belief_values,
        "solved": bool(solved),
        "symbolic_valid": bool(has_numeric_reasoning_belief),
        "cycles_executed": cycles_executed,
        "pot": pot_report,
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
    teacher_report = {
        "enabled": bool(args.teacher_enable_ingestion),
        "ab_arm": str(args.teacher_ab_arm),
        "low_volume_mode": bool(args.teacher_low_volume),
        "reason": "not requested",
    }
    if args.teacher_enable_ingestion or args.teacher_proposals:
        teacher_wm = WorkingMemory(default_ttl=3)
        teacher_report = teacher_wm.ingest_teacher_proposals(
            teacher_proposals,
            source="teacher_llm",
            teacher_id=str(args.teacher_id),
            teacher_model=str(args.teacher_model),
            ab_arm=str(args.teacher_ab_arm),
            enable_ingestion=bool(args.teacher_enable_ingestion),
            low_volume_mode=bool(args.teacher_low_volume),
            max_total=int(args.teacher_max_total),
            max_rules=int(args.teacher_max_rules),
            max_facts=int(args.teacher_max_facts),
            max_counterfactuals=int(args.teacher_max_counterfactuals),
            max_rule_chars=int(args.teacher_max_rule_chars),
        )
        if not args.quiet:
            print(
                "teacher-ingestion: "
                f"accepted={teacher_report.get('counts', {}).get('accepted', 0)} "
                f"rejected={teacher_report.get('counts', {}).get('rejected', 0)} "
                f"arm={teacher_report.get('ab_arm')}"
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
            )
            run["family"] = row.get("family", "unknown")
            run["question"] = question
            results.append(run)
        duration_sec = time.perf_counter() - started
        profile_report = _summarize_profile(profile=profile, rows=results, duration_sec=duration_sec)
        profile_report["results"] = results
        profile_reports.append(profile_report)

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
